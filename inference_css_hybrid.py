import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from scipy.optimize import linear_sum_assignment

# -------------------------------------------------------------------------
# 1. Hybrid Matcher & Golden Feature Buffer 클래스
# -------------------------------------------------------------------------
class GoldenFeatureBuffer:
    def __init__(self, sr=16000, n_mfcc=20, ema_alpha=0.8):
        self.mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc)
        self.golden_profiles = {}
        self.ema_alpha = ema_alpha

    def _get_vocal_profile(self, wav_tensor):
        mfcc = self.mfcc_transform(wav_tensor)
        return mfcc.mean(dim=-1)

    def update(self, track_id, wav_tensor):
        new_profile = self._get_vocal_profile(wav_tensor)
        if track_id not in self.golden_profiles:
            self.golden_profiles[track_id] = new_profile
        else:
            self.golden_profiles[track_id] = (self.ema_alpha * self.golden_profiles[track_id]) + \
                                             ((1 - self.ema_alpha) * new_profile)

    def get_similarity(self, track_id, wav_tensor):
        if track_id not in self.golden_profiles:
            return 0.0
        current_profile = self._get_vocal_profile(wav_tensor)
        prof_a = current_profile.unsqueeze(0)
        prof_b = self.golden_profiles[track_id].unsqueeze(0)
        return max(0.0, F.cosine_similarity(prof_a, prof_b).item())


import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torchaudio.transforms as T

# (GoldenFeatureBuffer 클래스는 기존과 완벽히 동일하므로 생략)

class AdvancedHybridMatcher:
    def __init__(self, sr=16000, overlap_sec=1.0, video_fps=25):
        self.sr = sr
        self.video_fps = video_fps
        self.overlap_samples = int(overlap_sec * sr)
        self.golden_buffer = GoldenFeatureBuffer(sr=sr)
        self.history = {}
        
        self.golden_vis_thresh = 0.9  
        self.active_rms_thresh = 0.01 

    def _calc_rms(self, wav):
        return torch.sqrt(torch.mean(wav**2, dim=-1) + 1e-9)

    # 🚀 [추가됨] 진정한 Audio-Visual Correlation 계산 로직
    def _calc_av_correlation(self, wav, asd_probs):
        """오디오 에너지 엔벨로프와 ASD 확률 궤적 간의 피어슨 상관계수 계산"""
        frame_len = self.sr // self.video_fps # 16000 / 25 = 640 samples per frame
        T_video = asd_probs.shape[0]
        
        # 1. 오디오를 비디오 프레임 길이에 맞게 패딩/크롭
        req_samples = T_video * frame_len
        if wav.shape[0] < req_samples:
            wav_padded = F.pad(wav, (0, req_samples - wav.shape[0]))
        else:
            wav_padded = wav[:req_samples]
            
        # 2. 프레임 단위로 쪼개서 RMS 에너지 추출 (T_video,)
        wav_frames = wav_padded.view(T_video, frame_len)
        audio_env = self._calc_rms(wav_frames)
        
        # 3. Pearson Correlation 계산 (-1.0 ~ 1.0)
        a_env = audio_env - audio_env.mean()
        v_env = asd_probs - asd_probs.mean()
        
        a_norm = torch.norm(a_env) + 1e-8
        v_norm = torch.norm(v_env) + 1e-8
        
        corr = torch.sum(a_env * v_env) / (a_norm * v_norm)
        return corr.item()

    def match(self, est_sources, asd_scores, chunk_idx):
        K, N = est_sources.shape[0], asd_scores.shape[0]
        T_chunk = est_sources.shape[-1]
        
        S_total = torch.zeros(K, N)
        asd_probs = torch.sigmoid(asd_scores)
        max_vis_scores, _ = torch.max(asd_probs, dim=-1)
        
        for n in range(N):
            vis_prob = max_vis_scores[n].item()
            is_vis_present = vis_prob > 0.01 
            
            for k in range(K):
                wav_k = est_sources[k]
                
                # (A) 시각 유사도 (Correlation 기반)
                score_vis = 0.0
                if is_vis_present:
                    score_vis = self._calc_av_correlation(wav_k, asd_probs[n]) * vis_prob
                
                # (B) 🚀 [수정됨] 단기 오디오 유사도 (제대로 된 벡터 내적/코사인 유사도)
                score_aud_short = 0.0
                if chunk_idx > 0 and n in self.history and self.history[n]['is_active']:
                    prev_tail = self.history[n]['overlap_wav']
                    curr_head = wav_k[:self.overlap_samples]
                    
                    # 수치적 안정성을 위해 F.cosine_similarity 사용 (1D이므로 unsqueeze 필요)
                    # 결과값은 -1.0 ~ 1.0 사이로 직관적으로 나옵니다.
                    score_aud_short = F.cosine_similarity(prev_tail.unsqueeze(0), curr_head.unsqueeze(0)).item()
                
                # (C) 장기 오디오 유사도 (Golden Feature)
                score_aud_long = self.golden_buffer.get_similarity(n, wav_k)
                
                # 🚀 [로깅 추가] 디버깅을 위해 각 점수 출력 (필요 시 주석 해제)
                # print(f"Chunk {chunk_idx} | T:{n}-C:{k} | V:{score_vis:.3f} S:{score_aud_short:.3f} L:{score_aud_long:.3f}")

                # 동적 라우팅
                if is_vis_present and vis_prob > 0.8:
                    S_total[k, n] = score_vis
                elif score_aud_short > 0.4: # 이제 정상적인 코사인 유사도 값이므로 0.4~0.7 수준에서 판단 가능
                    S_total[k, n] = score_aud_short
                else:
                    S_total[k, n] = score_aud_long
                    
        # Hungarian Matching
        row_ind, col_ind = linear_sum_assignment(-S_total.numpy())
        aligned_sources = torch.zeros((N, T_chunk), dtype=est_sources.dtype, device=est_sources.device)
        
        for r, c in zip(row_ind, col_ind):
            aligned_wav = est_sources[r]
            aligned_sources[c] = aligned_wav
            
            # 🚀 [업데이트 조건 완화] Golden Feature를 더 자주 업데이트하도록 변경
            # 비전이 아주 확실하거나(0.85), 비전은 좀 약해도 단기 오디오 매칭이 매우 확실할 때(0.9)
            is_high_conf_vis = max_vis_scores[c] > 0.85
            is_high_conf_aud = chunk_idx > 0 and n in self.history and score_aud_short > 0.9
            
            if (is_high_conf_vis or is_high_conf_aud) and self._calc_rms(aligned_wav) > self.active_rms_thresh:
                self.golden_buffer.update(c, aligned_wav)
                
            tail_wav = aligned_wav[-self.overlap_samples:]
            self.history[c] = {
                "overlap_wav": tail_wav,
                "is_active": self._calc_rms(tail_wav) > self.active_rms_thresh
            }
            
        return aligned_sources


# -------------------------------------------------------------------------
# 2. 메인 추론 루프 (Chunking, Filtering, OLA)
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--asd_dir", required=True)
    parser.add_argument("--output_dir", default="separated_css_hybrid")
    parser.add_argument("--ckpt_path", required=True, help="Path to AUDIO-ONLY TIGER model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # [주의] 이 접근법은 FiLM이 없는 오리지널 Audio-only TIGER 모델을 로드해야 합니다.
    import look2hear.models
    model = look2hear.models.TIGER(out_channels=128, in_channels=256, num_blocks=8, num_sources=2)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict({k.replace('audio_model.', ''): v for k, v in state_dict.items()}, strict=True)
    model.to(device).eval()

    sr, fps = 16000, 25
    chunk_sec, overlap_sec = 2.0, 1.0
    chunk_samples = int(chunk_sec * sr)
    stride_samples = int((chunk_sec - overlap_sec) * sr)
    chunk_frames = int(chunk_sec * fps)
    stride_frames = int((chunk_sec - overlap_sec) * fps)

    # 1. 처리할 오디오 파일 순회
    audio_files = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    
    with torch.no_grad():
        for audio_path in audio_files:
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            
            # 2. 다화자 필터링: 해당 영상의 ASD 파일들을 찾고 길이에 따라 정렬
            asd_files = glob.glob(os.path.join(args.asd_dir, f"{audio_basename}_track*.npy"))
            if not asd_files:
                continue
                
            # (파일 경로, 길이) 튜플 리스트 생성 후 길이 기준 내림차순 정렬
            asd_lengths = [(f, np.load(f).shape[-1]) for f in asd_files]
            asd_lengths.sort(key=lambda x: x[1], reverse=True)
            
            # 🚀 [수정됨] 화자가 1명이면 1개만, 2명 이상이면 가장 긴 2개만 추출
            N_speakers = min(2, len(asd_lengths))
            top_asd_files = [x[0] for x in asd_lengths[:N_speakers]]
            print(f"\nProcessing {audio_basename}: Selected {N_speakers} longest tracks.")

            # 오디오 로드 및 필요시 리샘플링
            wav, orig_sr = torchaudio.load(audio_path)
            if orig_sr != sr:
                wav = T.Resample(orig_sr, sr)(wav)
            wav = wav[0] # (T_audio,)
            
            # ASD 데이터 병합 (N, T_video)
            max_video_len = int((wav.shape[0] / sr) * fps)
            asd_matrix = torch.full((N_speakers, max_video_len), -10.0) # 기본값 -10 (확률 0)
            
            for i, f in enumerate(top_asd_files):
                data = torch.from_numpy(np.load(f)).float()
                L = min(data.shape[-1], max_video_len)
                asd_matrix[i, :L] = data[:L]
            
            # 3. Overlap-Add (OLA) 버퍼 초기화
            total_samples = wav.shape[0]
            out_buffer = torch.zeros(N_speakers, total_samples)
            window_sum = torch.zeros(1, total_samples)
            
            # Hann Window를 사용하여 자연스러운 Crossfade 유도
            window = torch.hann_window(chunk_samples)
            matcher = AdvancedHybridMatcher(sr=sr, overlap_sec=overlap_sec)

            # 4. Chunk 단위 순회 (스트리밍 시뮬레이션)
            num_chunks = (total_samples - chunk_samples) // stride_samples + 1
            if total_samples < chunk_samples: 
                num_chunks = 1
            
            for chunk_idx in range(num_chunks):
                start_samp = chunk_idx * stride_samples
                end_samp = start_samp + chunk_samples
                
                start_frame = chunk_idx * stride_frames
                end_frame = start_frame + chunk_frames
                
                # 끝부분 패딩 방어
                if end_samp > total_samples:
                    break 

                mix_chunk = wav[start_samp:end_samp].unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 32000)
                asd_chunk = asd_matrix[:, start_frame:end_frame] # (N, 50)
                
                # A. Audio-only 모델 추론 (Permutation 섞임)
                est_sources = model(mix_chunk).squeeze(0).cpu() # (3, 32000)
                est_sources = est_sources[:2, :]
                
                # B. Hybrid 매칭 (정렬 수행)
                aligned_chunk = matcher.match(est_sources, asd_chunk, chunk_idx) # (N, 32000)
                
                # C. Overlap-Add
                weighted_chunk = aligned_chunk * window.unsqueeze(0)
                out_buffer[:, start_samp:end_samp] += weighted_chunk
                window_sum[:, start_samp:end_samp] += window

            # 5. 후처리 및 저장
            # Window가 겹치지 않아 0이 된 부분 방어
            window_sum[window_sum < 1e-9] = 1.0 
            final_audio = out_buffer / window_sum
            
            for i, f in enumerate(top_asd_files):
                # 파일명에서 원본 트랙 ID 추출 (예: video_track1.npy -> 1)
                track_id = os.path.splitext(os.path.basename(f))[0].split("_track")[-1]
                
                out_name = f"{audio_basename}_track{track_id}_hybrid_target.wav"
                out_path = os.path.join(args.output_dir, out_name)
                torchaudio.save(out_path, final_audio[i].unsqueeze(0), sr)
                print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
