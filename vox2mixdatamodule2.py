import os
import glob
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def print_(message: str):
    print(message)

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

class VoxCeleb2MixTSEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,       # 예: 'audio_10w/wav_16k/min/tr'
        asd_dir: str,        # 예: 'asd_scores'
        sample_rate: int = 16000,
        video_fps: int = 25,
        segment: float = 3.0,
        normalize_audio: bool = False,
    ) -> None:
        super().__init__()
        self.EPS = 1e-8
        self.data_dir = data_dir
        self.asd_dir = asd_dir
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.normalize_audio = normalize_audio
        
        self.mix_dir = os.path.join(data_dir, 'mix')
        self.s1_dir = os.path.join(data_dir, 's1')
        self.s2_dir = os.path.join(data_dir, 's2')
        
        if segment is None:
            self.seg_len = None # Valid / Test 시 전체 길이 추론
        else:
            self.seg_len = int(segment * sample_rate) # Train 시 Chunking
            self.vid_seg_len = int(segment * video_fps)
            
        self.samples_per_frame = int(sample_rate / video_fps)
        self.test = self.seg_len is None

        # 믹스 파일 리스트업
        self.mix_files = glob.glob(os.path.join(self.mix_dir, '*.wav'))
        print_(f"Loaded {len(self.mix_files)} files from {self.mix_dir}")

    def __len__(self):
        return len(self.mix_files)

    def _parse_filename(self, filename):
        name_no_ext = filename.replace('.wav', '')
        parts = name_no_ext.split('_')
        # 파일명 컨벤션에 따른 파싱 (s1id_s1folder_s1file_s1snr_...)
        s1_npy_name = f"{parts[0]}_{parts[1]}_{parts[2]}.npy"
        s2_npy_name = f"{parts[4]}_{parts[5]}_{parts[6]}.npy"
        return s1_npy_name, s2_npy_name

    def __getitem__(self, idx: int):
        mix_path = self.mix_files[idx]
        filename = os.path.basename(mix_path)
        
        s1_path = os.path.join(self.s1_dir, filename)
        s2_path = os.path.join(self.s2_dir, filename)
        s1_npy_name, s2_npy_name = self._parse_filename(filename)

        # 1. Target 결정 로직
        # Test/Valid 시에는 평가의 일관성을 위해 무조건 S1을 타겟으로 고정합니다.
        # Train 시에는 50% 확률로 동적 스위칭(Data Augmentation 효과)
        if self.test or np.random.rand() < 0.5:
            target_path, interf_path = s1_path, s2_path
            asd_path = os.path.join(self.asd_dir, s1_npy_name)
        else:
            target_path, interf_path = s2_path, s1_path
            asd_path = os.path.join(self.asd_dir, s2_npy_name)

        # 2. ASD Score 로드
        try:
            target_asd = np.load(asd_path)
            target_asd = torch.from_numpy(target_asd).float()
        except Exception:
            # 안전망: 파일이 없으면 임시로 0 텐서 생성 (길이는 추후 맞춤)
            target_asd = torch.zeros(1000)

        # 3. 오디오 메타데이터 읽기 및 크롭(Crop) 계산
        info = sf.info(mix_path)
        total_audio_frames = info.frames
        
        if self.test or total_audio_frames <= self.seg_len:
            rand_start_audio = 0
            stop = None
            start_frame_video = 0
            # Test 시 비디오 길이를 오디오 실제 길이에 맞춤
            target_asd = target_asd[:total_audio_frames // self.samples_per_frame]
        else:
            # Train 시 Random Crop
            max_start_frame = len(target_asd) - self.vid_seg_len
            if max_start_frame > 0:
                start_frame_video = np.random.randint(0, max_start_frame)
            else:
                start_frame_video = 0
                
            rand_start_audio = start_frame_video * self.samples_per_frame
            stop = rand_start_audio + self.seg_len
            
            # 비디오 크롭
            target_asd = target_asd[start_frame_video : start_frame_video + self.vid_seg_len]

        # 4. 오디오 로드 (지정된 start~stop 구간만 메모리에 올려 속도 최적화)
        x, _ = sf.read(mix_path, start=rand_start_audio, stop=stop, dtype="float32")
        tgt, _ = sf.read(target_path, start=rand_start_audio, stop=stop, dtype="float32")
        intf, _ = sf.read(interf_path, start=rand_start_audio, stop=stop, dtype="float32")
        
        mixture = torch.from_numpy(x).unsqueeze(0) # (1, T)
        target = torch.from_numpy(tgt).unsqueeze(0)
        interf = torch.from_numpy(intf).unsqueeze(0)
        noise = torch.zeros_like(target) # 2-mix이므로 배경 소음은 0
        
        # 5. 패딩 로직 (Train 시 짧은 오디오 방어)
        if not self.test and mixture.shape[-1] < self.seg_len:
            pad_len = self.seg_len - mixture.shape[-1]
            mixture = torch.nn.functional.pad(mixture, (0, pad_len))
            target = torch.nn.functional.pad(target, (0, pad_len))
            interf = torch.nn.functional.pad(interf, (0, pad_len))
            noise = torch.nn.functional.pad(noise, (0, pad_len))
            
            vid_pad_len = self.vid_seg_len - target_asd.shape[-1]
            if vid_pad_len > 0:
                target_asd = torch.nn.functional.pad(target_asd, (0, vid_pad_len))

        # 6. Sources 병합 (Target, Interference, Noise Sink)
        sources = torch.cat([target, interf, noise], dim=0) # (3, T)
        target_asd = target_asd.unsqueeze(0) # (1, T_video)

        # 7. 오디오 정규화 (TIGER 순정 로직)
        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

        # [중요] TIGER 루프 호환을 위해 3개의 리턴값 반환
        return mixture, sources, target_asd


class VoxCeleb2MixDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        asd_dir: str,
        sample_rate: int = 16000,
        video_fps: int = 25,
        segment: float = 3.0,
        normalize_audio: bool = False,
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.asd_dir = asd_dir
        
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.segment = segment
        self.normalize_audio = normalize_audio
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = VoxCeleb2MixTSEDataset(
            data_dir=self.train_dir,
            asd_dir=self.asd_dir,
            sample_rate=self.sample_rate,
            video_fps=self.video_fps,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
        )
        self.data_val = VoxCeleb2MixTSEDataset(
            data_dir=self.valid_dir,
            asd_dir=self.asd_dir,
            sample_rate=self.sample_rate,
            video_fps=self.video_fps,
            segment=None, # Valid는 전체 길이로 평가
            normalize_audio=self.normalize_audio,
        )
        self.data_test = VoxCeleb2MixTSEDataset(
            data_dir=self.test_dir,
            asd_dir=self.asd_dir,
            sample_rate=self.sample_rate,
            video_fps=self.video_fps,
            segment=None, # Test는 전체 길이로 평가
            normalize_audio=self.normalize_audio,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
