import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class AdvancedHybridMatcher:
    def __init__(self, sr=16000, overlap_sec=1.5, video_fps=25):
        self.sr = sr
        self.video_fps = video_fps
        self.overlap_samples = int(overlap_sec * sr)
        
        # Phase 2: ECAPA-TDNN 기반 Robust Buffer (기존에 정의한 클래스 사용)
        self.golden_buffer = RobustGoldenFeatureBuffer(sr=sr)
        
        self.history = {}
        self.is_initialized = {}  # 🚀 [Fix #1] 화자별 지연 초기화 상태
        self.init_vis_thresh = 0.85 
        self.active_rms_thresh = 0.005

    def _calc_rms(self, wav):
        return torch.sqrt(torch.mean(wav**2, dim=-1) + 1e-9)

    def _calc_av_correlation(self, wav, asd_probs):
        # 오디오 에너지와 입술 움직임 간의 피어슨 상관계수 (코드 생략 방지 - 원복됨)
        frame_len = self.sr // self.video_fps
        T_video = asd_probs.shape[0]
        req_samples = T_video * frame_len
        wav_padded = F.pad(wav, (0, max(0, req_samples - wav.shape[0])))[:req_samples]
        wav_frames = wav_padded.view(T_video, frame_len)
        audio_env = self._calc_rms(wav_frames)
        a_env = audio_env - audio_env.mean()
        v_env = asd_probs - asd_probs.mean()
        corr = torch.sum(a_env * v_env) / (torch.norm(a_env) * torch.norm(v_env) + 1e-8)
        return corr.item()

    def match(self, est_sources, asd_scores, chunk_idx):
        K, N = est_sources.shape[0], asd_scores.shape[0]
        T_chunk = est_sources.shape[-1]
        
        S_total = torch.zeros(K, N)
        S_short_matrix = torch.zeros(K, N) 
        Log_matrix = [["" for _ in range(N)] for _ in range(K)]
        
        asd_probs = torch.sigmoid(asd_scores)
        max_vis_scores, _ = torch.max(asd_probs, dim=-1)
        
        for n in range(N):
            vis_prob = max_vis_scores[n].item()
            
            # 🚀 [Lazy Init Trigger] 확실한 입술 움직임 포착 시 활성화
            if not self.is_initialized.get(n, False):
                if vis_prob >= self.init_vis_thresh:
                    self.is_initialized[n] = True
                    print(f"🎯 [Track {n}] First Activation! Vision: {vis_prob:.2f}")

            for k in range(K):
                wav_k = est_sources[k]
                
                # 1. 시각 유사도 (AV Correlation)
                score_vis = max(0.0, self._calc_av_correlation(wav_k, asd_probs[n])) * vis_prob
                
                # 2. 단기 오디오 유사도 (Overlap)
                score_aud_short = 0.0
                if chunk_idx > 0 and n in self.history and self.history[n]['is_active']:
                    prev_tail = self.history[n]['overlap_wav']
                    curr_head = wav_k[:self.overlap_samples]
                    # VAD 로직 제거: 순수 코사인 유사도만 사용
                    score_aud_short = max(0.0, F.cosine_similarity(prev_tail.unsqueeze(0), curr_head.unsqueeze(0)).item())
                
                S_short_matrix[k, n] = score_aud_short
                
                # 3. 장기 오디오 유사도 (ECAPA-TDNN)
                score_aud_long = self.golden_buffer.get_similarity(n, wav_k)
                
                # 매칭 우선순위 라우팅
                if vis_prob > 0.8:
                    S_total[k, n] = score_vis
                    Log_matrix[k][n] = f"VIS({score_vis:.2f})"
                elif score_aud_short > 0.4:
                    S_total[k, n] = score_aud_short
                    Log_matrix[k][n] = f"SHORT({score_aud_short:.2f})|L({score_aud_long:.2f})"
                else:
                    S_total[k, n] = score_aud_long
                    Log_matrix[k][n] = f"LONG({score_aud_long:.2f})|S({score_aud_short:.2f})"

            # 🚀 [Isolation] 초기화되지 않은 화자는 매칭에서 사실상 배제 (-1.0점)
            if not self.is_initialized.get(n, False):
                S_total[:, n] = -1.0
        
        # Hungarian Matching (K x N)
        row_ind, col_ind = linear_sum_assignment(-S_total.numpy())
        
        aligned_sources = torch.zeros((N, T_chunk), dtype=est_sources.dtype, device=est_sources.device)
        chunk_decisions = {}
        
        for r, c in zip(row_ind, col_ind):
            # 🚀 [Silent Padding] 활성화 전까지는 무음 유지
            if not self.is_initialized.get(c, False):
                aligned_sources[c] = torch.zeros_like(est_sources[r])
                chunk_decisions[c] = "SILENT (Waiting)"
                continue
                
            aligned_wav = est_sources[r]
            aligned_sources[c] = aligned_wav
            chunk_decisions[c] = Log_matrix[r][c]
            
            matched_short_score = S_short_matrix[r, c].item()
            vis_prob_for_c = max_vis_scores[c].item()
            
            # 🚀 Robust Update (Phase 2 & 5)
            is_strict_vis = vis_prob_for_c > 0.95
            is_high_conf_vis = vis_prob_for_c > 0.85
            is_high_conf_aud = chunk_idx > 0 and c in self.history and matched_short_score > 0.9
            
            if (is_high_conf_vis or is_high_conf_aud) and self._calc_rms(aligned_wav) > self.active_rms_thresh:
                self.golden_buffer.update(c, aligned_wav, is_strict_vis=is_strict_vis)
                
            tail_wav = aligned_wav[-self.overlap_samples:]
            self.history[c] = {
                "overlap_wav": tail_wav,
                "is_active": self._calc_rms(tail_wav) > self.active_rms_thresh
            }
            
        return aligned_sources, chunk_decisions
