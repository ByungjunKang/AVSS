import os
import glob
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

class VoxCeleb2MixTSEDataset(Dataset):
    def __init__(self, root_dir, chunk_len_sec=3.0, audio_sr=16000, video_fps=25):
        """
        root_dir: 'audio_10w'와 'asd_scores' 폴더가 포함된 최상위 경로
        """
        super().__init__()
        self.root_dir = root_dir
        self.chunk_len_sec = chunk_len_sec
        self.audio_sr = audio_sr
        self.video_fps = video_fps
        
        self.audio_chunk_len = int(audio_sr * chunk_len_sec) # 48000
        self.video_chunk_len = int(video_fps * chunk_len_sec) # 75
        self.samples_per_frame = int(audio_sr / video_fps) # 640
        
        # 경로 셋팅
        self.mix_dir = os.path.join(root_dir, 'audio_10w/wav_16k/min/tr/mix')
        self.s1_dir = os.path.join(root_dir, 'audio_10w/wav_16k/min/tr/s1')
        self.s2_dir = os.path.join(root_dir, 'audio_10w/wav_16k/min/tr/s2')
        self.asd_dir = os.path.join(root_dir, 'asd_scores')
        
        # 전체 믹스 파일 리스트업
        self.mix_files = glob.glob(os.path.join(self.mix_dir, '*.wav'))
        print(f"Total mixture files found: {len(self.mix_files)}")

    def __len__(self):
        return len(self.mix_files)

    def _parse_filename(self, filename):
        """
        파일명에서 s1과 s2의 ASD npy 파일명을 추출합니다.
        파일명 형식: s1id_s1folder_s1file_s1snr_s2id_s2folder_s2file_s2snr.wav
        """
        name_no_ext = filename.replace('.wav', '')
        parts = name_no_ext.split('_')
        
        # [주의] VoxCeleb2의 Youtube ID(folder)에 '_'가 포함될 수 있는 엣지 케이스를 
        # 방어하기 위해 더 견고한 파싱이 필요할 수 있습니다. 
        # 여기서는 가장 스탠다드하게 split('_')로 8조각이 났다고 가정합니다.
        s1_npy_name = f"{parts[0]}_{parts[1]}_{parts[2]}.npy"
        s2_npy_name = f"{parts[4]}_{parts[5]}_{parts[6]}.npy"
        
        return s1_npy_name, s2_npy_name

    def __getitem__(self, idx):
        mix_path = self.mix_files[idx]
        filename = os.path.basename(mix_path)
        
        s1_path = os.path.join(self.s1_dir, filename)
        s2_path = os.path.join(self.s2_dir, filename)
        
        # 1. 오디오 로드 (1, T_audio)
        mix_wav, _ = torchaudio.load(mix_path)
        s1_wav, _ = torchaudio.load(s1_path)
        s2_wav, _ = torchaudio.load(s2_path)
        
        s1_npy_name, s2_npy_name = self._parse_filename(filename)
        
        # 2. Dynamic Target Switching (TSE 패러다임의 핵심)
        # 50% 확률로 S1 또는 S2를 Target으로 지정
        if random.random() < 0.5:
            target_wav = s1_wav
            interf_wav = s2_wav
            asd_path = os.path.join(self.asd_dir, s1_npy_name)
        else:
            target_wav = s2_wav
            interf_wav = s1_wav
            asd_path = os.path.join(self.asd_dir, s2_npy_name)
            
        # 3번 채널(Noise Sink)용 빈 텐서 (2-mix 환경이므로 0)
        noise_wav = torch.zeros_like(target_wav)
            
        # 3. ASD Score 로드 (1, T_video)
        try:
            target_asd = np.load(asd_path)
            target_asd = torch.from_numpy(target_asd).float().unsqueeze(0)
        except Exception:
            # 파일이 깨졌을 경우의 Fallback
            target_asd = torch.zeros(1, mix_wav.shape[-1] // self.samples_per_frame)

        # 4. Synchronized Random Cropping (동기화 자르기)
        total_audio_len = mix_wav.shape[-1]
        total_video_len = target_asd.shape[-1]
        
        if total_audio_len < self.audio_chunk_len or total_video_len < self.video_chunk_len:
            # Padding 로직
            mix_wav = self._pad(mix_wav, self.audio_chunk_len)
            target_wav = self._pad(target_wav, self.audio_chunk_len)
            interf_wav = self._pad(interf_wav, self.audio_chunk_len)
            noise_wav = self._pad(noise_wav, self.audio_chunk_len)
            target_asd = self._pad(target_asd, self.video_chunk_len)
            start_sample = 0
            start_frame = 0
        else:
            # Random Crop 로직
            max_start_frame = total_video_len - self.video_chunk_len
            start_frame = random.randint(0, max_start_frame)
            start_sample = start_frame * self.samples_per_frame
            
        mix_wav = mix_wav[:, start_sample : start_sample + self.audio_chunk_len]
        target_wav = target_wav[:, start_sample : start_sample + self.audio_chunk_len]
        interf_wav = interf_wav[:, start_sample : start_sample + self.audio_chunk_len]
        noise_wav = noise_wav[:, start_sample : start_sample + self.audio_chunk_len]
        target_asd = target_asd[:, start_frame : start_frame + self.video_chunk_len]
        
        # 5. TIGER 호환성을 위한 Sources 텐서 병합 (3, T_audio)
        # TIGER는 보통 sources 변수 하나로 타겟들을 묶어서 받습니다.
        sources_wav = torch.cat([target_wav, interf_wav, noise_wav], dim=0)

        # 최종 반환: Mixture, Sources(Target/Interf/Noise), 1D Score
        return mix_wav, sources_wav, target_asd

    def _pad(self, tensor, target_len):
        pad_len = target_len - tensor.shape[-1]
        if pad_len > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_len))
        return tensor
