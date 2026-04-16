import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class AdvancedHybridMatcher:
    def __init__(self, sr=16000, video_fps=25):
        self.sr = sr
        self.video_fps = video_fps
        # overlap_sec 및 history 변수 완전 제거 (Short Anchor 폐기)
        
        # Phase 2 버퍼 연동 (RobustGoldenFeatureBuffer)
        self.golden_buffer = RobustGoldenFeatureBuffer(sr=sr)
        
        self.is_initialized = {}  # Lazy Init 상태
        self.init_vis_thresh = 0.85 
        self.active_rms_thresh = 0.005
        self.rejection_thresh = 0.15 # 대폭 완화된 Safety Net 임계치

    def _calc_rms(self, wav):
        return torch.sqrt(torch.mean(wav**2, dim=-1) + 1e-9)

    def _calc_av_correlation(self, wav, asd_probs):
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
        Log_matrix = [["" for _ in range(N)] for _ in range(K)]
        
        asd_probs = torch.sigmoid(asd_scores)
        max_vis_scores, _ = torch.max(asd_probs, dim=-1)
        
        for n in range(N):
            vis_prob = max_vis_scores[n].item()
            is_vis_present = vis_prob > 0.01 
            
            # 🚀 1. Lazy Init Trigger
            if not self.is_initialized.get(n, False):
                if vis_prob >= self.init_vis_thresh:
                    self.is_initialized[n] = True
                    print(f"🎯 [Track {n}] Activated! Vision: {vis_prob:.2f}")

            for k in range(K):
                wav_k = est_sources[k]
                
                # 시각 유사도 (Vision)
                score_vis = max(0.0, self._calc_av_correlation(wav_k, asd_probs[n])) * vis_prob if is_vis_present else 0.0
                
                # 장기 오디오 유사도 (Long - ECAPA-TDNN)
                score_aud_long = self.golden_buffer.get_similarity(n, wav_k)
                
                # 🚀 2. Routing (Short가 사라진 깔끔한 2지 선다)
                if vis_prob > 0.8:
                    S_total[k, n] = score_vis
                    Log_matrix[k][n] = f"VIS({score_vis:.2f})"
                else:
                    S_total[k, n] = score_aud_long
                    Log_matrix[k][n] = f"LONG({score_aud_long:.2f})"

            # 🚀 3. Lazy Init Penalty (격리)
            if not self.is_initialized.get(n, False):
                S_total[:, n] = -1.0
        
        # 헝가리안 매칭
        row_ind, col_ind = linear_sum_assignment(-S_total.numpy())
        
        aligned_sources = torch.zeros((N, T_chunk), dtype=est_sources.dtype, device=est_sources.device)
        chunk_decisions = {}
        
        for r, c in zip(row_ind, col_ind):
            matched_score = S_total[r, c].item()
            routing_reason = Log_matrix[r][c]
            is_vision_matched = routing_reason.startswith("VIS")
            
            # 🚀 4. Silent Padding 1순위: 아직 깨어나지 않은 화자
            if not self.is_initialized.get(c, False):
                aligned_sources[c] = torch.zeros_like(est_sources[r])
                chunk_decisions[c] = "SILENT (Waiting)"
                continue
                
            # 🚀 5. Silent Padding 2순위: 오디오(Long) 매칭인데 점수가 15% 미만인 가짜 매칭 거절
            if not is_vision_matched and matched_score < self.rejection_thresh:
                aligned_sources[c] = torch.zeros_like(est_sources[r])
                chunk_decisions[c] = f"REJECTED({matched_score:.2f}) -> SILENT"
                continue

            # 정상 매칭 처리
            aligned_wav = est_sources[r]
            aligned_sources[c] = aligned_wav
            chunk_decisions[c] = routing_reason
            
            vis_prob_for_c = max_vis_scores[c].item()
            
            # 🚀 6. Golden Buffer 업데이트 권한 부여
            is_strict_vis = vis_prob_for_c > 0.95
            is_high_conf_vis = vis_prob_for_c > 0.85
            # Short가 사라졌으므로, 화면 밖으로 나갔을 때를 대비해 ECAPA 점수가 극히 높을 때(0.85 이상)만 오디오 업데이트 허용
            is_high_conf_long = (not is_vision_matched) and (matched_score > 0.85)
            
            if (is_high_conf_vis or is_high_conf_long) and self._calc_rms(aligned_wav) > self.active_rms_thresh:
                self.golden_buffer.update(c, aligned_wav, is_strict_vis=is_strict_vis, is_high_conf_vis=is_high_conf_vis)
            
        return aligned_sources, chunk_decisions
