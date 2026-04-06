import os
import glob
import argparse
import pickle
import subprocess
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from scipy.optimize import linear_sum_assignment

# -------------------------------------------------------------------------
# 1. Golden Buffer & Hybrid Matcher (로깅 기능 추가)
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

class AdvancedHybridMatcher:
    def __init__(self, sr=16000, overlap_sec=1.5, video_fps=25):
        self.sr = sr
        self.video_fps = video_fps
        self.overlap_samples = int(overlap_sec * sr)
        self.golden_buffer = GoldenFeatureBuffer(sr=sr)
        self.history = {}
        self.golden_vis_thresh = 0.9  
        self.active_rms_thresh = 0.005 # 침묵 기준 약간 완화

    def _calc_rms(self, wav):
        return torch.sqrt(torch.mean(wav**2, dim=-1) + 1e-9)

    def _calc_av_correlation(self, wav, asd_probs):
        frame_len = self.sr // self.video_fps
        T_video = asd_probs.shape[0]
        req_samples = T_video * frame_len
        
        if wav.shape[0] < req_samples:
            wav_padded = F.pad(wav, (0, req_samples - wav.shape[0]))
        else:
            wav_padded = wav[:req_samples]
            
        wav_frames = wav_padded.view(T_video, frame_len)
        audio_env = self._calc_rms(wav_frames)
        
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
        S_short_matrix = torch.zeros(K, N) # 🚀 [버그 픽스] 단기 유사도를 저장할 전용 행렬
        Log_matrix = [["" for _ in range(N)] for _ in range(K)] 
        
        asd_probs = torch.sigmoid(asd_scores)
        max_vis_scores, _ = torch.max(asd_probs, dim=-1)
        
        for n in range(N):
            vis_prob = max_vis_scores[n].item()
            is_vis_present = vis_prob > 0.01 
            
            for k in range(K):
                wav_k = est_sources[k]
                
                # 1. 시각 유사도 (음수 방어 로직 통일)
                score_vis = 0.0
                if is_vis_present:
                    score_vis = max(0.0, self._calc_av_correlation(wav_k, asd_probs[n])) * vis_prob
                
                # 2. 단기 오디오 유사도
                score_aud_short = 0.0
                if chunk_idx > 0 and n in self.history and self.history[n]['is_active']:
                    prev_tail = self.history[n]['overlap_wav']
                    curr_head = wav_k[:self.overlap_samples]
                    # 음수 방어 및 코사인 유사도
                    score_aud_short = max(0.0, F.cosine_similarity(prev_tail.unsqueeze(0), curr_head.unsqueeze(0)).item())
                
                S_short_matrix[k, n] = score_aud_short # 🚀 연산된 점수를 자기 자리에 정확히 저장!
                
                # 3. 장기 오디오 유사도
                score_aud_long = self.golden_buffer.get_similarity(n, wav_k)
                
                # 라우팅
                if is_vis_present and vis_prob > 0.8:
                    S_total[k, n] = score_vis
                    Log_matrix[k][n] = f"VIS ({score_vis:.2f})"
                elif score_aud_short > 0.4:
                    S_total[k, n] = score_aud_short
                    Log_matrix[k][n] = f"SHORT({score_aud_short:.2f}) | L({score_aud_long:.2f})"
                else:
                    S_total[k, n] = score_aud_long
                    Log_matrix[k][n] = f"LONG({score_aud_long:.2f}) | S({score_aud_short:.2f})"
                    
        # Hungarian Matching
        row_ind, col_ind = linear_sum_assignment(-S_total.numpy())
        aligned_sources = torch.zeros((N, T_chunk), dtype=est_sources.dtype, device=est_sources.device)
        chunk_decisions = {} 
        
        for r, c in zip(row_ind, col_ind):
            aligned_wav = est_sources[r]
            aligned_sources[c] = aligned_wav
            chunk_decisions[c] = Log_matrix[r][c] 
            
            # 🚀 [버그 픽스] 정확히 (r, c) 매칭에 해당하는 단기 점수를 꺼내어 검사!
            matched_short_score = S_short_matrix[r, c].item()
            is_high_conf_vis = max_vis_scores[c] > 0.85
            is_high_conf_aud = chunk_idx > 0 and c in self.history and matched_short_score > 0.9
            
            if (is_high_conf_vis or is_high_conf_aud) and self._calc_rms(aligned_wav) > self.active_rms_thresh:
                self.golden_buffer.update(c, aligned_wav)
                
            tail_wav = aligned_wav[-self.overlap_samples:]
            self.history[c] = {
                "overlap_wav": tail_wav,
                "is_active": self._calc_rms(tail_wav) > self.active_rms_thresh
            }
            
        return aligned_sources, chunk_decisions

# -------------------------------------------------------------------------
# 2. 개별 영상 렌더링 함수 (로깅 + Bbox 오버레이)
# -------------------------------------------------------------------------
def render_individual_video(video_path, target_wav_path, asd_npy_path, meta_pkl_path, 
                            track_id, track_logs, output_path, target_fps=25):
    
    # 임시 비디오 파일 (오디오 씌우기 전)
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    
    with open(meta_pkl_path, 'rb') as f:
        tracks_meta = pickle.load(f)
        
    full_scores = np.load(asd_npy_path)
    meta = tracks_meta.get(track_id, None)
    if meta is None:
        print(f"⚠️ No meta for track {track_id}")
        return

    frames_list = meta['frame'].tolist() if isinstance(meta['frame'], np.ndarray) else meta['frame']
    
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (width, height))
    
    frame_idx, target_frame_idx = 0, 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_time_sec = frame_idx / source_fps
        meta_idx = int(round(current_time_sec * target_fps))
        
        # 1. 바운딩 박스 및 ASD 점수 그리기
        if meta_idx in frames_list and meta_idx < len(full_scores):
            score = full_scores[meta_idx]
            try:
                list_pos = frames_list.index(meta_idx)
                bbox = meta['bbox'][list_pos]
                x1, y1, x2, y2 = map(int, bbox)
                
                prob = 1 / (1 + np.exp(-score)) # Sigmoid for display
                color = (0, 255, 0) if score > 0.0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"ID:{track_id} | Prob:{prob:.2f}"
                cv2.putText(frame, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except ValueError:
                pass

        # 2. 🚀 시간에 맞는 매칭 로그 그리기 (좌측 상단)
        # 현재 시간보다 작거나 같은 로그 중 가장 최신 것 찾기
        valid_logs = [log['text'] for log in track_logs if log['start_sec'] <= current_time_sec]
        active_log = valid_logs[-1] if valid_logs else "WAITING..."
        
        # 로그 박스 디자인 (검은 반투명 배경)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        cv2.putText(frame, f"ANCHOR: {active_log}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # 노란색 텍스트

        # 싱크 맞춤 저장
        expected_target_time = target_frame_idx / target_fps
        if current_time_sec >= expected_target_time:
            out.write(frame)
            target_frame_idx += 1
            
        frame_idx += 1
        
    cap.release()
    out.release()
    
    # 3. 🚀 FFmpeg로 비디오와 분리된 타겟 오디오 합성 (Muxing)
    print(f"  └─ Muxing audio & video for Track {track_id}...")
    cmd = [
        "ffmpeg", "-y", "-i", temp_video_path, "-i", target_wav_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path, "-loglevel", "error"
    ]
    subprocess.call(cmd)
    os.remove(temp_video_path) # 임시 파일 삭제
    print(f"  ✅ Finished: {output_path}")

# -------------------------------------------------------------------------
# 3. 메인 추론 루프
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--asd_dir", required=True)
    parser.add_argument("--output_dir", default="diagnostic_results")
    parser.add_argument("--ckpt_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    import look2hear.models
    model = look2hear.models.TIGER(out_channels=128, in_channels=256, num_blocks=8, num_sources=3)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict({k.replace('audio_model.', ''): v for k, v in state_dict.items()}, strict=False)
    model.to(device).eval()

    sr, fps = 16000, 25
    chunk_sec, overlap_sec = 3.0, 1.5
    chunk_samples = int(chunk_sec * sr)
    stride_samples = int((chunk_sec - overlap_sec) * sr)
    chunk_frames = int(chunk_sec * fps)
    stride_frames = int((chunk_sec - overlap_sec) * fps)

    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    
    with torch.no_grad():
        for video_path in video_files:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(args.audio_dir, f"{basename}.wav")
            meta_path = os.path.join(args.asd_dir, f"{basename}_meta.pkl")
            
            if not os.path.exists(audio_path) or not os.path.exists(meta_path):
                continue
                
            top_tracks = [0, 1]
            valid_tracks = []
            for tid in top_tracks:
                if os.path.exists(os.path.join(args.asd_dir, f"{basename}_track{tid}.npy")):
                    valid_tracks.append(tid)
                    
            if not valid_tracks:
                continue

            print(f"\n🎧 Processing {basename} | Tracks: {valid_tracks}")
            
            wav, _ = torchaudio.load(audio_path)
            if wav.dim() > 1: wav = wav[0]
            
            total_samples = wav.shape[0]
            max_video_len = int((total_samples / sr) * fps)
            asd_matrix = torch.full((len(valid_tracks), max_video_len), -10.0)
            
            for i, tid in enumerate(valid_tracks):
                data = torch.from_numpy(np.load(os.path.join(args.asd_dir, f"{basename}_track{tid}.npy"))).float()
                L = min(data.shape[-1], max_video_len)
                asd_matrix[i, :L] = data[:L]

            out_buffer = torch.zeros(len(valid_tracks), total_samples)
            window_sum = torch.zeros(1, total_samples)
            window = torch.hann_window(chunk_samples)
            matcher = AdvancedHybridMatcher(sr=sr, overlap_sec=overlap_sec, video_fps=fps)
            
            # 🚀 화자별 매칭 로깅 저장소 
            # track_logs[tid] = [{"start_sec": 0.0, "text": "VIS(0.9)"}, ...]
            track_logs = {tid: [] for tid in valid_tracks}

            num_chunks = (total_samples - chunk_samples) // stride_samples + 1
            if total_samples < chunk_samples: num_chunks = 1
            
            for chunk_idx in range(num_chunks):
                start_samp = chunk_idx * stride_samples
                end_samp = start_samp + chunk_samples
                start_frame = chunk_idx * stride_frames
                end_frame = start_frame + chunk_frames
                
                # 🚀 [수정됨] break 대신 Padding 수행
                # if end_samp > total_samples: break (이 줄 삭제)

                # 오디오 패딩 준비
                if end_samp > total_samples:
                    actual_wav = wav[start_samp:total_samples]
                    pad_len = end_samp - total_samples
                    mix_chunk = F.pad(actual_wav, (0, pad_len)).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    mix_chunk = wav[start_samp:end_samp].unsqueeze(0).unsqueeze(0).to(device)
                
                # 비디오(ASD) 패딩 준비
                if end_frame > max_video_len:
                    actual_asd = asd_matrix[:, start_frame:max_video_len]
                    pad_len_frames = end_frame - max_video_len
                    # 얼굴이 없는 상태(-10.0)로 패딩
                    pad_asd = torch.full((len(valid_tracks), pad_len_frames), -10.0)
                    asd_chunk = torch.cat([actual_asd, pad_asd], dim=-1)
                else:
                    asd_chunk = asd_matrix[:, start_frame:end_frame]
                
                raw_est_sources = model(mix_chunk).squeeze(0).cpu()
                speech_sources = raw_est_sources[:2, :] 
                
                # 매칭 수행 및 결정 로그 반환
                aligned_chunk, decisions = matcher.match(speech_sources, asd_chunk, chunk_idx)
                
                weighted_chunk = aligned_chunk * window.unsqueeze(0)
                out_buffer[:, start_samp:end_samp] += weighted_chunk
                window_sum[:, start_samp:end_samp] += window
                
                # 🚀 각 화자별 현재 청크의 시작 시간(stride 기반)에 따른 결정 로그 기록
                chunk_start_sec = start_samp / sr
                for i, tid in enumerate(valid_tracks):
                    decision_text = decisions.get(i, "UNKNOWN")
                    track_logs[tid].append({"start_sec": chunk_start_sec, "text": decision_text})

            window_sum[window_sum < 1e-9] = 1.0 
            final_audio = out_buffer / window_sum
            
            # 4. 각 화자별 오디오 저장 및 비디오 렌더링 호출
            for i, tid in enumerate(valid_tracks):
                target_wav_path = os.path.join(args.output_dir, f"{basename}_track{tid}_target.wav")
                torchaudio.save(target_wav_path, final_audio[i].unsqueeze(0), sr)
                
                final_video_path = os.path.join(args.output_dir, f"{basename}_track{tid}_diagnostic.mp4")
                asd_npy_path = os.path.join(args.asd_dir, f"{basename}_track{tid}.npy")
                
                render_individual_video(
                    video_path, target_wav_path, asd_npy_path, meta_path,
                    tid, track_logs[tid], final_video_path, target_fps=fps
                )

if __name__ == "__main__":
    main()
