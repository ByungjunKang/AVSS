import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# (RobustGoldenFeatureBuffer 클래스 내부에 추가)
    def has_profile(self, track_id):
        """해당 화자의 Golden Profile이 버퍼에 존재하는지 확인"""
        return track_id in self.golden_profiles


class AdvancedHybridMatcher:
    def __init__(self, sr=16000, video_fps=25):
        self.sr = sr
        self.video_fps = video_fps
        self.golden_buffer = RobustGoldenFeatureBuffer(sr=sr)
        
        self.is_initialized = {}  
        self.init_vis_thresh = 0.85 
        self.active_rms_thresh = 0.005
        self.rejection_thresh = 0.15 # 최소 안전망

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
            has_profile = self.golden_buffer.has_profile(n) # 🚀 프로필 존재 여부 확인
            
            # Lazy Init Trigger
            if not self.is_initialized.get(n, False):
                if vis_prob >= self.init_vis_thresh:
                    self.is_initialized[n] = True
                    print(f"🎯 [Track {n}] Activated! Vision: {vis_prob:.2f}")

            for k in range(K):
                wav_k = est_sources[k]
                
                score_vis = max(0.0, self._calc_av_correlation(wav_k, asd_probs[n])) * vis_prob if is_vis_present else 0.0
                score_aud_long = self.golden_buffer.get_similarity(n, wav_k)
                
                # 🚀 라우팅 전략: FUSED Anchor 도입
                if vis_prob > 0.8:
                    if has_profile:
                        # 이미 목소리를 안다면 Vision과 Long을 5:5로 융합 (Cross-check)
                        fused_score = (score_vis + score_aud_long) / 2.0
                        S_total[k, n] = fused_score
                        Log_matrix[k][n] = f"FUSED({fused_score:.2f}) [V:{score_vis:.2f}|L:{score_aud_long:.2f}]"
                    else:
                        # 첫 발화라 목소리를 모른다면 Vision만 맹신 (Cold Start)
                        S_total[k, n] = score_vis
                        Log_matrix[k][n] = f"VIS({score_vis:.2f})"
                else:
                    # 입을 닫고 있다면 Long 앵커만 사용
                    S_total[k, n] = score_aud_long
                    Log_matrix[k][n] = f"LONG({score_aud_long:.2f})"

            if not self.is_initialized.get(n, False):
                S_total[:, n] = -1.0
        
        row_ind, col_ind = linear_sum_assignment(-S_total.numpy())
        
        aligned_sources = torch.zeros((N, T_chunk), dtype=est_sources.dtype, device=est_sources.device)
        chunk_decisions = {}
        
        for r, c in zip(row_ind, col_ind):
            matched_score = S_total[r, c].item()
            routing_reason = Log_matrix[r][c]
            
            # Silent Padding 1순위: 대기 중
            if not self.is_initialized.get(c, False):
                aligned_sources[c] = torch.zeros_like(est_sources[r])
                chunk_decisions[c] = "SILENT (Waiting)"
                continue
                
            # Silent Padding 2순위: 0.15 미만은 FUSED든 LONG이든 가짜 매칭으로 간주
            if matched_score < self.rejection_thresh:
                aligned_sources[c] = torch.zeros_like(est_sources[r])
                chunk_decisions[c] = f"REJECTED({matched_score:.2f}) -> SILENT"
                continue

            aligned_wav = est_sources[r]
            aligned_sources[c] = aligned_wav
            chunk_decisions[c] = routing_reason
            
            vis_prob_for_c = max_vis_scores[c].item()
            is_strict_vis = vis_prob_for_c > 0.95
            is_high_conf_vis = vis_prob_for_c > 0.85
            
            # 🚀 업데이트 허용 조건: 극도로 높은 비전(0.85)이거나, 오디오(Long) 점수 자체가 매우 높을 때(0.85)
            # FUSED의 경우에도 비전 점수가 높으므로 is_high_conf_vis를 통해 안정적으로 업데이트됨
            is_high_conf_long = routing_reason.startswith("LONG") and matched_score > 0.85
            
            if (is_high_conf_vis or is_high_conf_long) and self._calc_rms(aligned_wav) > self.active_rms_thresh:
                self.golden_buffer.update(c, aligned_wav, is_strict_vis=is_strict_vis, is_high_conf_vis=is_high_conf_vis)
            
        return aligned_sources, chunk_decisions
