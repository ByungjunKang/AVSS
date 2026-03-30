import numpy as np
import torch
import soundfile as sf
import os
import glob
from torch.utils.data import Dataset

class VoxCeleb2MixTSEDataset(Dataset):
    def __init__(self, data_dir, asd_dir, sample_rate=16000, video_fps=25, segment=3.0, normalize_audio=False):
        super().__init__()
        # ... (기존 초기화 코드 동일) ...
        
    def __getitem__(self, idx):
        # ... (기존 파일 로드 로직: mix, target, interf, asd 파일 경로 가져오기) ...

        # [솔루션 2] Turn-taking / Silence 데이터 증강 로직
        # 전체 데이터의 50% 확률로 타겟 화자에게 '침묵 구간'을 생성
        apply_turn_taking = np.random.rand() < 0.5
        
        # 증강이 적용될 시간 구간 (3초 segment 중 1.0 ~ 2.0초 구간)
        silence_start_sec = 1.0
        silence_end_sec = 2.0
        
        silence_start_audio = int(silence_start_sec * self.sample_rate)
        silence_end_audio = int(silence_end_sec * self.sample_rate)
        
        silence_start_video = int(silence_start_sec * self.video_fps)
        silence_end_video = int(silence_end_sec * self.video_fps)

        # 1. 오디오 로드 및 증강 적용
        # sf.read(..., start=..., stop=...) 로 청크 로드 로직은 동일
        x, _ = sf.read(mix_path, start=..., stop=..., dtype="float32")
        tgt, _ = sf.read(target_path, start=..., stop=..., dtype="float32")
        intf, _ = sf.read(interf_path, start=..., stop=..., dtype="float32")
        
        if apply_turn_taking:
            # 🚀 [핵심] 타겟 화자가 1~2초 구간에 입을 닫음 (오디오 삭제)
            tgt[silence_start_audio : silence_end_audio] = 0.0
            
            # [선택사항] 만약 Caster(제3자) 난입을 더 강조하고 싶다면,
            # 타겟이 침묵하는 구간에 Interference 채널의 볼륨을 키워주거나 
            # 외부 소음을 이 구간에만 더 강하게 섞어줄 수 있습니다.
            # intf[silence_start_audio : silence_end_audio] *= 1.5 
            
            # 믹스처 재합성 (2-mix 환경이므로 타겟이 0이 되면 간섭음만 남음)
            x = tgt + intf

        mixture = torch.from_numpy(x).unsqueeze(0)
        target = torch.from_numpy(tgt).unsqueeze(0)
        interf = torch.from_numpy(intf).unsqueeze(0)
        sources = torch.cat([target, interf], dim=0) # (2, T)

        # 2. ASD Score 로드 및 증강 적용
        target_asd = np.load(asd_path)
        
        if apply_turn_taking:
            # 🚀 [핵심] 오디오와 싱크를 맞춰 비전 힌트도 '침묵(0)'으로 깎음
            # 원래 -5~5 범위의 로짓 값이므로 완벽한 Inactive(-5.0 이하) 값으로 채움
            target_asd[silence_start_video : silence_end_video] = -10.0 # 시그모이드 취하면 완벽한 0

        target_asd = torch.from_numpy(target_asd).float()
        
        # ... (이하 패딩 및 정규화 로직 동일) ...
        # target_asd = target_asd.unsqueeze(0) # (1, T_video)

        return mixture, sources, target_asd
