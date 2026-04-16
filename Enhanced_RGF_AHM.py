import os
import torch
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier

class RobustGoldenFeatureBuffer:
    def __init__(self, sr=16000, ema_alpha=0.8, device="cuda"):
        self.sr = sr
        self.ema_alpha = ema_alpha
        self.device = device
        
        # 오프라인 강제 로드 설정
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base_dir, "pretrained_ecapa")
        
        print(f"⏳ Loading ECAPA-TDNN locally from: {local_dir}")
        self.encoder = EncoderClassifier.from_hparams(source=local_dir, savedir=local_dir, run_opts={"device": device})
        self.encoder.eval()
        
        self.golden_profiles = {}
        self.candidate_queues = {}
        self.queue_max_size = 3
        self.consistency_thresh = 0.8

    def _get_vocal_profile(self, wav_tensor):
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(wav_tensor)
            return embeddings.squeeze()

    def update(self, track_id, wav_tensor, is_strict_vis=False, is_high_conf_vis=False):
        new_profile = self._get_vocal_profile(wav_tensor)
        
        # 🚀 [Fix #4] Bootstrap 로직: 버퍼가 비어있을 때는 is_high_conf_vis(0.85)만 넘어도 최초 1회 즉시 허용
        is_empty = track_id not in self.golden_profiles
        if is_strict_vis or (is_empty and is_high_conf_vis):
            self._apply_ema_update(track_id, new_profile)
            self.candidate_queues[track_id] = [] # 큐 초기화
            return

        # Slow Track (Queue 대기)
        if track_id not in self.candidate_queues:
            self.candidate_queues[track_id] = []
            
        self.candidate_queues[track_id].append(new_profile)
        
        if len(self.candidate_queues[track_id]) >= self.queue_max_size:
            q = self.candidate_queues[track_id]
            sim_01 = F.cosine_similarity(q[0].unsqueeze(0), q[1].unsqueeze(0)).item()
            sim_12 = F.cosine_similarity(q[1].unsqueeze(0), q[2].unsqueeze(0)).item()
            sim_02 = F.cosine_similarity(q[0].unsqueeze(0), q[2].unsqueeze(0)).item()
            
            if min(sim_01, sim_12, sim_02) > self.consistency_thresh:
                mean_profile = F.normalize(torch.stack(q).mean(dim=0), dim=0)
                self._apply_ema_update(track_id, mean_profile)
                
            self.candidate_queues[track_id].pop(0)

    def _apply_ema_update(self, track_id, new_profile):
        if track_id not in self.golden_profiles:
            self.golden_profiles[track_id] = new_profile
        else:
            updated = (self.ema_alpha * self.golden_profiles[track_id]) + ((1 - self.ema_alpha) * new_profile)
            self.golden_profiles[track_id] = F.normalize(updated, dim=0)

    def get_similarity(self, track_id, wav_tensor):
        if track_id not in self.golden_profiles:
            return 0.0
        current_profile = self._get_vocal_profile(wav_tensor)
        prof_a = current_profile.unsqueeze(0)
        prof_b = self.golden_profiles[track_id].unsqueeze(0)
        return max(0.0, F.cosine_similarity(prof_a, prof_b).item())


from scipy.optimize import linear_sum_assignment

class AdvancedHybridMatcher:
    def __init__(self, sr=16000, overlap_sec=1.5, video_fps=25):
        self.sr = sr
        self.video_fps = video_fps
        self.overlap_samples = int(overlap_sec * sr)
        
        # Phase 2 버퍼 연동
        self.golden_buffer = RobustGoldenFeatureBuffer(sr=sr)
        
        self.history = {}
        self.is_initialized = {}  # Lazy Init 상태
        self.init_vis_thresh = 0.85 
        self.active_rms_thresh = 0.005
        self.rejection_thresh = 0.45 # 🚀 [추가] 헝가리안 억지 매칭 거절 임계치

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
        S_short_matrix = torch.zeros(K, N) 
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
                
                score_vis = max(0.0, self._calc_av_correlation(wav_k, asd_probs[n])) * vis_prob if is_vis_present else 0.0
                
                score_aud_short = 0.0
                if chunk_idx > 0 and n in self.history and self.history[n]['is_active']:
                    prev_tail = self.history[n]['overlap_wav']
                    curr_head = wav_k[:self.overlap_samples]
                    score_aud_short = max(0.0, F.cosine_similarity(prev_tail.unsqueeze(0), curr_head.unsqueeze(0)).item())
                
                S_short_matrix[k, n] = score_aud_short
                score_aud_long = self.golden_buffer.get_similarity(n, wav_k)
                
                # 🚀 2. Routing (Short Threshold 0.75로 대폭 상향)
                if vis_prob > 0.8:
                    S_total[k, n] = score_vis
                    Log_matrix[k][n] = f"VIS({score_vis:.2f})"
                elif score_aud_short > 0.75:
                    S_total[k, n] = score_aud_short
                    Log_matrix[k][n] = f"SHORT({score_aud_short:.2f})|L({score_aud_long:.2f})"
                else:
                    S_total[k, n] = score_aud_long
                    Log_matrix[k][n] = f"LONG({score_aud_long:.2f})|S({score_aud_short:.2f})"

            # 🚀 3. Lazy Init Penalty (격리)
            if not self.is_initialized.get(n, False):
                S_total[:, n] = -1.0
        
        # 헝가리안 매칭
        row_ind, col_ind = linear_sum_assignment(-S_total.numpy())
        
        aligned_sources = torch.zeros((N, T_chunk), dtype=est_sources.dtype, device=est_sources.device)
        chunk_decisions = {}
        
        for r, c in zip(row_ind, col_ind):
            matched_score = S_total[r, c].item()
            
            # 🚀 4. Silent Padding 1순위: 아직 깨어나지 않은 화자
            if not self.is_initialized.get(c, False):
                aligned_sources[c] = torch.zeros_like(est_sources[r])
                chunk_decisions[c] = "SILENT (Waiting)"
                continue
                
            # 🚀 5. Silent Padding 2순위: 헝가리안 억지 매칭 거절 (Absolute Rejection)
            if matched_score < self.rejection_thresh:
                aligned_sources[c] = torch.zeros_like(est_sources[r])
                chunk_decisions[c] = f"REJECTED({matched_score:.2f}) -> SILENT"
                continue

            # 정상 매칭 처리
            aligned_wav = est_sources[r]
            aligned_sources[c] = aligned_wav
            chunk_decisions[c] = Log_matrix[r][c]
            
            matched_short_score = S_short_matrix[r, c].item()
            vis_prob_for_c = max_vis_scores[c].item()
            
            is_strict_vis = vis_prob_for_c > 0.95
            is_high_conf_vis = vis_prob_for_c > 0.85
            is_high_conf_aud = chunk_idx > 0 and c in self.history and matched_short_score > 0.9
            
            # 🚀 6. Golden Buffer 업데이트 (Bootstrap을 위한 플래그 전달)
            if (is_high_conf_vis or is_high_conf_aud) and self._calc_rms(aligned_wav) > self.active_rms_thresh:
                self.golden_buffer.update(c, aligned_wav, is_strict_vis=is_strict_vis, is_high_conf_vis=is_high_conf_vis)
                
            tail_wav = aligned_wav[-self.overlap_samples:]
            self.history[c] = {
                "overlap_wav": tail_wav,
                "is_active": self._calc_rms(tail_wav) > self.active_rms_thresh
            }
            
        return aligned_sources, chunk_decisions
