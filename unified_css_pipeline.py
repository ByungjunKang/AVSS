import os
import sys

# 🚀 [경로 병합] LR-ASD 모델 폴더의 절대 경로를 파이썬 시스템 경로에 강제로 추가합니다.
# 주의: 아래 경로를 실제 PC의 LR-ASD 폴더 절대 경로로 반드시 변경해 주세요!
LR_ASD_DIR = "/home/user/workspace/LR-ASD-folder" 
if not os.path.exists(LR_ASD_DIR):
    raise FileNotFoundError(f"LR-ASD 폴더를 찾을 수 없습니다: {LR_ASD_DIR}")
sys.path.append(LR_ASD_DIR)
import glob
import time
import argparse
import pickle
import subprocess
import warnings
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchaudio
import python_speech_features
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from speechbrain.pretrained import EncoderClassifier

# S3FD, LR-ASD, SceneDetect 등 외부 라이브러리 (기존 환경과 동일하게 유지)
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CSS Matcher & Buffer Classes
# =====================================================================
class RobustGoldenFeatureBuffer:
    def __init__(self, sr=16000, ema_alpha=0.8, device="cuda"):
        self.sr = sr
        self.ema_alpha = ema_alpha
        self.device = device
        
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

    def has_profile(self, track_id):
        return track_id in self.golden_profiles

    def _get_vocal_profile(self, wav_tensor):
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(wav_tensor)
            return embeddings.squeeze()

    def update(self, track_id, wav_tensor, is_strict_vis=False, is_high_conf_vis=False):
        new_profile = self._get_vocal_profile(wav_tensor)
        is_empty = track_id not in self.golden_profiles
        if is_strict_vis or (is_empty and is_high_conf_vis):
            self._apply_ema_update(track_id, new_profile)
            self.candidate_queues[track_id] = [] 
            return

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

class AdvancedHybridMatcher:
    def __init__(self, sr=16000, video_fps=25):
        self.sr = sr
        self.video_fps = video_fps
        self.golden_buffer = RobustGoldenFeatureBuffer(sr=sr)
        self.active_rms_thresh = 0.005

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
            has_profile = self.golden_buffer.has_profile(n)

            for k in range(K):
                wav_k = est_sources[k]
                score_vis = max(0.0, self._calc_av_correlation(wav_k, asd_probs[n])) * vis_prob if is_vis_present else 0.0
                score_aud_long = self.golden_buffer.get_similarity(n, wav_k)
                
                if vis_prob > 0.8:
                    if has_profile:
                        fused_score = (score_vis + score_aud_long) / 2.0
                        S_total[k, n] = fused_score
                        Log_matrix[k][n] = f"FUSED({fused_score:.2f}) [V:{score_vis:.2f}|L:{score_aud_long:.2f}]"
                    else:
                        S_total[k, n] = score_vis
                        Log_matrix[k][n] = f"VIS({score_vis:.2f})"
                else:
                    S_total[k, n] = score_aud_long
                    Log_matrix[k][n] = f"LONG({score_aud_long:.2f})"
        
        row_ind, col_ind = linear_sum_assignment(-S_total.numpy())
        
        aligned_sources = torch.zeros((N, T_chunk), dtype=est_sources.dtype, device=est_sources.device)
        chunk_decisions = {}
        
        for r, c in zip(row_ind, col_ind):
            matched_score = S_total[r, c].item()
            routing_reason = Log_matrix[r][c]
            
            aligned_wav = est_sources[r]
            aligned_sources[c] = aligned_wav
            chunk_decisions[c] = routing_reason
            
            vis_prob_for_c = max_vis_scores[c].item()
            is_strict_vis = vis_prob_for_c > 0.95
            is_high_conf_vis = vis_prob_for_c > 0.85
            is_high_conf_long = routing_reason.startswith("LONG") and matched_score > 0.85
            
            if (is_high_conf_vis or is_high_conf_long) and self._calc_rms(aligned_wav) > self.active_rms_thresh:
                self.golden_buffer.update(c, aligned_wav, is_strict_vis=is_strict_vis, is_high_conf_vis=is_high_conf_vis)
            
        return aligned_sources, chunk_decisions

# =====================================================================
# 2. ASD Extraction Functions
# =====================================================================
def scene_detect(args):
    videoManager = VideoManager([args.videoPath])
    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    return sceneList

def inference_video(args, DET):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
    return dets

def bb_intersection_over_union(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def track_shot(args, sceneFaces):
    iouThres = 0.5     
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frameI = np.arange(frameNum[0], frameNum[-1]+1)
            bboxesI = []
            for ij in range(0,4):
                interpfn = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
                tracks.append({'frame':frameI,'bbox':bboxesI})
    return tracks

def crop_video(args, track, cropFile):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) 
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))
    dets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']: 
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) 
        dets['x'].append((det[0]+det[2])/2) 
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]   
        bsi = int(bs * (1 + 2 * cs))  
        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my, mx = dets['y'][fidx] + bsi, dets['x'][fidx] + bsi  
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd = (track['frame'][-1]+1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
          (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
    subprocess.call(command, shell=True, stdout=None) 
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) 
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track':track, 'proc_track':dets}

def evaluate_network_single(file, ASD_model):
    _, audio = wavfile.read(file.replace('.avi', '.wav'))
    audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
    
    video = cv2.VideoCapture(file)
    videoFeature = []
    while video.isOpened():
        ret, frames = video.read()
        if ret:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            videoFeature.append(face)
        else:
            break
    video.release()
    videoFeature = np.array(videoFeature)
    
    length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
    audioFeature = audioFeature[:int(round(length * 100)),:]
    videoFeature = videoFeature[:int(round(length * 25)),:,:]
    
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} 
    allScore = [] 
    
    for duration in durationSet:
        batchSize = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batchSize):
                inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                embedA = ASD_model.model.forward_audio_frontend(inputA)
                embedV = ASD_model.model.forward_visual_frontend(inputV)    
                out = ASD_model.model.forward_audio_visual_backend(embedA, embedV)
                score = ASD_model.lossAV.forward(out, labels = None)
                scores.extend(score)
        allScore.append(scores)
        
    allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
    return allScore

class ProcessArgs:
    pass

def process_single_video(video_path, output_dir, DET, ASD_model):
    """
    영상 1개에 대해 ASD를 수행하고 NPY와 Meta를 생성. 
    생성된 파일 경로 리스트와 트랙 ID를 반환.
    """
    args = ProcessArgs()
    args.videoPath = video_path
    args.nDataLoaderThread = 4
    args.facedetScale = 0.25
    args.minTrack = 10
    args.numFailedDet = 10
    args.minFaceSize = 1
    args.cropScale = 0.40
    
    vid_name = os.path.basename(video_path).replace('.mp4', '')
    args.savePath = os.path.join(output_dir, 'temp_workspace', vid_name)
    
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav') # 추출된 16kHz 오디오
    
    output_npy_base = os.path.join(output_dir, vid_name)
    
    try:
        os.makedirs(args.pyaviPath, exist_ok = True) 
        os.makedirs(args.pyframesPath, exist_ok = True) 
        os.makedirs(args.pycropPath, exist_ok = True) 
        
        # 1. 미디어 추출
        subprocess.call(f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads {args.nDataLoaderThread} -async 1 -r 25 {args.videoFilePath} -loglevel panic", shell=True)
        subprocess.call(f"ffmpeg -y -i {args.videoFilePath} -qscale:a 0 -ac 1 -vn -threads {args.nDataLoaderThread} -ar 16000 {args.audioFilePath} -loglevel panic", shell=True)
        subprocess.call(f"ffmpeg -y -i {args.videoFilePath} -qscale:v 2 -threads {args.nDataLoaderThread} -f image2 {os.path.join(args.pyframesPath, '%06d.jpg')} -loglevel panic", shell=True)
        
        # 2. 탐지 및 추적
        scene = scene_detect(args)
        faces = inference_video(args, DET)
        
        allTracks = []
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= args.minTrack: 
                allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
                
        if len(allTracks) == 0: return [], None, None
            
        valid_tracks = [t for t in allTracks if len(t['frame']) >= 25]
        valid_tracks.sort(key=lambda x: len(x['frame']), reverse=True)
        top_tracks = valid_tracks[:2]
        
        if len(top_tracks) == 0: return [], None, None

        _, full_audio = wavfile.read(args.audioFilePath)
        total_frames = int((len(full_audio) / 16000.0) * 25)

        valid_tracks_info = {}
        valid_tids = []
        
        # 3. ASD 스코어 추출
        for track_id, track in enumerate(top_tracks):
            crop_file_path = os.path.join(args.pycropPath, f'track_{track_id}')
            crop_video(args, track, crop_file_path)
            scores = evaluate_network_single(crop_file_path + '.avi', ASD_model)
            
            global_scores = np.full(total_frames, -10.0, dtype=np.float32)
            frames_list = track['frame'].tolist() if isinstance(track['frame'], np.ndarray) else track['frame']
            
            valid_len = min(len(scores), len(frames_list))
            for i in range(valid_len):
                global_idx = frames_list[i]
                if global_idx < total_frames:
                    global_scores[global_idx] = scores[i]
            
            track_npy_path = f"{output_npy_base}_track{track_id}.npy"
            np.save(track_npy_path, global_scores)
            
            valid_tracks_info[track_id] = {'frame': track['frame'], 'bbox': track['bbox']}
            valid_tids.append(track_id)
            
        meta_pkl_path = f"{output_npy_base}_meta.pkl"
        with open(meta_pkl_path, 'wb') as f:
            pickle.dump(valid_tracks_info, f)
            
        # 메인 TIGER 루프에서 사용하기 위해 추출된 16kHz 오디오를 아웃풋 폴더에 복사
        final_audio_path = f"{output_npy_base}_16k.wav"
        subprocess.call(f"cp {args.audioFilePath} {final_audio_path}", shell=True)

        return valid_tids, meta_pkl_path, final_audio_path
        
    finally:
        rmtree(args.savePath, ignore_errors=True)

# =====================================================================
# 3. Visualization Rendering
# =====================================================================
def render_individual_video(video_path, target_wav_path, asd_npy_path, meta_pkl_path, 
                            track_id, track_logs, output_path, target_fps=25):
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    
    with open(meta_pkl_path, 'rb') as f:
        tracks_meta = pickle.load(f)
        
    full_scores = np.load(asd_npy_path)
    meta = tracks_meta.get(track_id, None)
    if meta is None: return

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
        
        if meta_idx in frames_list and meta_idx < len(full_scores):
            score = full_scores[meta_idx]
            try:
                list_pos = frames_list.index(meta_idx)
                bbox = meta['bbox'][list_pos]
                x1, y1, x2, y2 = map(int, bbox)
                
                prob = 1 / (1 + np.exp(-score))
                color = (0, 255, 0) if score > 0.0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} | Prob:{prob:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except ValueError:
                pass

        valid_logs = [log['text'] for log in track_logs if log['start_sec'] <= current_time_sec]
        active_log = valid_logs[-1] if valid_logs else "WAITING..."
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, f"ANCHOR: {active_log}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 

        expected_target_time = target_frame_idx / target_fps
        if current_time_sec >= expected_target_time:
            out.write(frame)
            target_frame_idx += 1
            
        frame_idx += 1
        
    cap.release()
    out.release()
    
    cmd = [
        "ffmpeg", "-y", "-i", temp_video_path, "-i", target_wav_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path, "-loglevel", "error"
    ]
    subprocess.call(cmd)
    if os.path.exists(temp_video_path): os.remove(temp_video_path)

# =====================================================================
# 4. Main Pipeline (Unification + Profiling)
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Directory containing source mp4 files")
    parser.add_argument("--output_dir", default="unified_results", help="Directory for all outputs and intermediate files")
    parser.add_argument("--tiger_ckpt", required=True, help="Path to TIGER model checkpoint")
    parser.add_argument("--asd_weight", required=True, help="Path to pretrain_AVA.model for LR-ASD")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("⏳ [Init] Loading Models (S3FD, LR-ASD, TIGER)...")
    # Load ASD Models
    DET = S3FD(device='cuda')
    ASD_model = ASD()
    ASD_model.loadParameters(args.asd_weight)
    ASD_model.eval()

    # Load TIGER
    import look2hear.models
    model = look2hear.models.TIGER(out_channels=132, in_channels=256, num_blocks=8, num_sources=3,
                                  upsampling_depth=5, win=640, stride=160, sample_rate=16000)
    ckpt = torch.load(args.tiger_ckpt, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict({k.replace('audio_model.', ''): v for k, v in state_dict.items()}, strict=False)
    model.to(device).eval()

    sr, fps = 16000, 25
    chunk_sec, overlap_sec = 3.0, 1.0
    chunk_samples = int(chunk_sec * sr)
    stride_samples = int((chunk_sec - overlap_sec) * sr)
    chunk_frames = int(chunk_sec * fps)

    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    
    with torch.no_grad():
        for video_path in video_files:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            print(f"\n=======================================================")
            print(f"🎬 Processing Video: {basename}")
            print(f"=======================================================")

            # ---------------------------------------------------------
            # [Stage 1] ASD Extraction Profiling
            # ---------------------------------------------------------
            print("▶️ [Stage 1] Extracting ASD Scores...")
            t_asd_start = time.perf_counter()
            
            valid_tracks, meta_path, audio_path = process_single_video(video_path, args.output_dir, DET, ASD_model)
            
            t_asd_end = time.perf_counter()
            asd_time_total = t_asd_end - t_asd_start

            if not valid_tracks:
                print(f"⚠️ No valid tracks found for {basename}. Skipping.")
                continue

            # ---------------------------------------------------------
            # Prepare Separation & Matching
            # ---------------------------------------------------------
            wav, _ = torchaudio.load(audio_path)
            if wav.dim() > 1: wav = wav[0]
            
            total_samples = wav.shape[0]
            max_video_len = int((total_samples / sr) * fps)
            asd_matrix = torch.full((len(valid_tracks), max_video_len), -10.0)
            
            for i, tid in enumerate(valid_tracks):
                track_npy_path = os.path.join(args.output_dir, f"{basename}_track{tid}.npy")
                data = torch.from_numpy(np.load(track_npy_path)).float()
                L = min(data.shape[-1], max_video_len)
                asd_matrix[i, :L] = data[:L]

            out_buffer = torch.zeros(len(valid_tracks), total_samples)
            window_sum = torch.zeros(1, total_samples)
            window = torch.hann_window(chunk_samples)
            
            matcher = AdvancedHybridMatcher(sr=sr, video_fps=fps)
            track_logs = {tid: [] for tid in valid_tracks}

            num_chunks = (total_samples - chunk_samples) // stride_samples + 1
            if total_samples < chunk_samples: num_chunks = 1
            
            sep_time_total = 0.0
            match_time_total = 0.0

            print(f"▶️ [Stage 2 & 3] Processing {num_chunks} chunks (TIGER Separation & Matching)...")
            
            for chunk_idx in range(num_chunks):
                start_samp = chunk_idx * stride_samples
                end_samp = start_samp + chunk_samples
                start_frame = int(round((start_samp / sr) * fps))
                end_frame = start_frame + chunk_frames
                
                # Audio Padding
                if end_samp > total_samples:
                    actual_wav = wav[start_samp:total_samples]
                    pad_len = end_samp - total_samples
                    mix_chunk = F.pad(actual_wav, (0, pad_len)).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    mix_chunk = wav[start_samp:end_samp].unsqueeze(0).unsqueeze(0).to(device)
                
                # Video Padding
                if end_frame > max_video_len:
                    actual_asd = asd_matrix[:, start_frame:max_video_len]
                    pad_len_frames = end_frame - max_video_len
                    pad_asd = torch.full((len(valid_tracks), pad_len_frames), -10.0)
                    asd_chunk = torch.cat([actual_asd, pad_asd], dim=-1)
                else:
                    asd_chunk = asd_matrix[:, start_frame:end_frame]
                
                # ---------------------------------------------------------
                # [Stage 2] TIGER Separation Profiling
                # ---------------------------------------------------------
                t_sep_start = time.perf_counter()
                raw_est_sources = model(mix_chunk).squeeze(0).cpu()
                speech_sources = raw_est_sources[:2, :] 
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                sep_time_total += (time.perf_counter() - t_sep_start)
                
                # ---------------------------------------------------------
                # [Stage 3] Matching Profiling
                # ---------------------------------------------------------
                t_match_start = time.perf_counter()
                aligned_chunk, decisions = matcher.match(speech_sources, asd_chunk, chunk_idx)
                match_time_total += (time.perf_counter() - t_match_start)
                
                weighted_chunk = aligned_chunk * window.unsqueeze(0)
                out_buffer[:, start_samp:end_samp] += weighted_chunk
                window_sum[:, start_samp:end_samp] += window
                
                chunk_start_sec = start_samp / sr
                for i, tid in enumerate(valid_tracks):
                    decision_text = decisions.get(i, "UNKNOWN")
                    track_logs[tid].append({"start_sec": chunk_start_sec, "text": decision_text})

            window_sum[window_sum < 1e-9] = 1.0 
            final_audio = out_buffer / window_sum
            
            # ---------------------------------------------------------
            # 📊 Profiling Report Print
            # ---------------------------------------------------------
            print(f"\n📊 [Profiler Report] Performance Breakdown for '{basename}'")
            print(f"  - Total Audio Duration : {total_samples / sr:.2f} seconds")
            print(f"  - Total Chunks         : {num_chunks} chunks")
            print(f"  -------------------------------------------------------")
            print(f"  ⏱️ 1. ASD Extraction : {asd_time_total:.3f} s")
            print(f"  ⏱️ 2. TIGER Inference: {sep_time_total:.3f} s (Avg {(sep_time_total/num_chunks)*1000:.1f} ms / chunk)")
            print(f"  ⏱️ 3. CSS Matching   : {match_time_total:.3f} s (Avg {(match_time_total/num_chunks)*1000:.1f} ms / chunk)")
            print(f"  -------------------------------------------------------")
            print(f"  ⚡ Total Backend Time : {asd_time_total + sep_time_total + match_time_total:.3f} s")

            # ---------------------------------------------------------
            # [Stage 4] Visualization (Not included in profile time)
            # ---------------------------------------------------------
            print("\n▶️ [Stage 4] Rendering Diagnostic Videos...")
            for i, tid in enumerate(valid_tracks):
                target_wav_path = os.path.join(args.output_dir, f"{basename}_track{tid}_target.wav")
                torchaudio.save(target_wav_path, final_audio[i].unsqueeze(0), sr)
                
                final_video_path = os.path.join(args.output_dir, f"{basename}_track{tid}_diagnostic.mp4")
                asd_npy_path = os.path.join(args.output_dir, f"{basename}_track{tid}.npy")
                
                render_individual_video(
                    video_path, target_wav_path, asd_npy_path, meta_path,
                    tid, track_logs[tid], final_video_path, target_fps=fps
                )

if __name__ == "__main__":
    main()
