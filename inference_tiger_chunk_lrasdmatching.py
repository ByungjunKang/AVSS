import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from scipy.optimize import linear_sum_assignment
import os

# --- 추론 전 설정 ---
sr = 16000
fps = 25
chunk_len_sec = 3.0
chunk_samples = int(chunk_len_sec * sr)       # 48000
chunk_frames = int(chunk_len_sec * fps)       # 75

stride_samples = chunk_samples // 2           # 24000
stride_frames = chunk_frames // 2             # 37 (보통 정수 배율을 사용함)
samples_per_frame = sr // fps                 # 640

device = 'cuda' if torch.cuda.is_available() else 'cpu'
window = torch.hann_window(chunk_samples).to(device)

# --- 입력 데이터 가이드 ---
# mix shape: (1, 1, T_audio)
# global_asd_scores shape: (N_tracks, T_video) -> LR-ASD에서 뽑은 전체 영상의 1D Score
N_tracks = global_asd_scores.shape[0]
num_sources = 2 # TIGER 모델이 뱉는 음성 소스 개수 (Noise 채널이 있다면 3으로 설정)
# -------------------------

T_audio_total = mix.shape[-1]
T_video_total = global_asd_scores.shape[-1]

# 1. Padding (오디오와 ASD Score 모두 스트라이드에 맞게 패딩)
pad_samples = chunk_samples - ((T_audio_total - chunk_samples) % stride_samples)
if pad_samples == chunk_samples: pad_samples = 0
mix_padded = F.pad(mix, (0, pad_samples))
T_audio_padded = mix_padded.shape[-1]

pad_frames = chunk_frames - ((T_video_total - chunk_frames) % stride_frames)
if pad_frames == chunk_frames: pad_frames = 0
asd_padded = F.pad(global_asd_scores, (0, pad_frames))

# 2. 누적 버퍼 생성 (최종 결과물은 LR-ASD의 N_tracks 개수만큼 생성됨)
est_sources = torch.zeros(1, N_tracks, T_audio_padded).to(device)
window_sum = torch.zeros(1, 1, T_audio_padded).to(device)

# 글로벌 ASD 텐서를 GPU로 이동
asd_padded = asd_padded.to(device)

model.eval()
with torch.no_grad():
    # 오디오 샘플과 비디오 프레임 인덱스를 동시에 순회
    for s_idx, f_idx in zip(
        range(0, T_audio_padded - chunk_samples + 1, stride_samples),
        range(0, asd_padded.shape[-1] - chunk_frames + 1, stride_frames)
    ):
        # 1. Chunk 자르기
        mix_chunk = mix_padded[..., s_idx : s_idx + chunk_samples]
        asd_chunk = asd_padded[:, f_idx : f_idx + chunk_frames] # (N_tracks, chunk_frames)
        
        # 2. 모델 추론 (TIGER)
        # est_chunk shape: (1, num_sources, chunk_samples)
        est_chunk = model(mix_chunk)
        
        # --- [추가] Chunk-level Permutation Matching (LR-ASD 기반) ---
        
        # A. TIGER 출력 오디오의 프레임별 RMS 에너지 계산 (25fps로 다운샘플링)
        # (1, num_sources, chunk_frames, samples_per_frame)
        wav_reshaped = est_chunk.view(num_sources, chunk_frames, samples_per_frame)
        energy_chunk = torch.sqrt(torch.mean(wav_reshaped**2, dim=-1) + 1e-9) # (num_sources, chunk_frames)
        
        # B. 상관관계(Pearson Correlation)를 위한 정규화 (Z-score)
        # 에너지 정규화
        e_mean = energy_chunk.mean(dim=-1, keepdim=True)
        e_std = energy_chunk.std(dim=-1, keepdim=True) + 1e-9
        e_norm = (energy_chunk - e_mean) / e_std
        
        # ASD Score 정규화
        a_mean = asd_chunk.mean(dim=-1, keepdim=True)
        a_std = asd_chunk.std(dim=-1, keepdim=True) + 1e-9
        a_norm = (asd_chunk - a_mean) / a_std
        
        # C. Correlation Matrix 계산 (N_tracks x num_sources)
        # 내적 후 N-1 로 나누어 상관계수 행렬 도출
        corr_matrix = torch.matmul(a_norm, e_norm.transpose(0, 1)) / (chunk_frames - 1)
        
        # D. Hungarian Method로 최적의 짝 찾기 (Scipy 사용을 위해 CPU로 이동)
        corr_np = corr_matrix.cpu().numpy()
        # Cost를 최소화하는 함수이므로 음수(-)를 취해 Maximum Bipartite Matching 수행
        row_ind, col_ind = linear_sum_assignment(-corr_np)
        
        # E. 찾아낸 순서(col_ind)대로 현재 청크의 오디오 채널 재배열
        # row_ind는 LR-ASD의 트랙 순서(0, 1, ...), col_ind는 그에 매칭된 TIGER 채널 인덱스
        matched_chunk = torch.zeros(1, N_tracks, chunk_samples).to(device)
        for r, c in zip(row_ind, col_ind):
            # 만약 상관계수가 너무 낮다면(예: 0.05 미만) 할당하지 않는 방어 로직을 추가할 수도 있습니다.
            matched_chunk[0, r, :] = est_chunk[0, c, :]
            
        # ---------------------------------------------------------------
        
        # 3. 윈도우(가중치) 곱하기 및 누적
        matched_chunk_windowed = matched_chunk * window
        est_sources[..., s_idx : s_idx + chunk_samples] += matched_chunk_windowed
        window_sum[..., s_idx : s_idx + chunk_samples] += window

# 4. 정규화 및 패딩 제거
est_sources = est_sources / (window_sum + 1e-8)
est_sources = est_sources[..., :T_audio_total]

# 5. 화자별 wav 파일 저장
output_dir = "separated_outputs"
os.makedirs(output_dir, exist_ok=True)

for track_id in range(N_tracks):
    # est_sources shape: (1, N_tracks, T_total) -> (1, T_total)로 슬라이싱
    wav_tensor = est_sources[0, track_id, :].unsqueeze(0).cpu()
    save_path = os.path.join(output_dir, f"track_{track_id}_separated.wav")
    torchaudio.save(save_path, wav_tensor, sr)
    print(f"Saved {save_path}")
