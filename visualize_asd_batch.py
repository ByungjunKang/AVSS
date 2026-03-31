import cv2
import numpy as np
import os
import pickle
import glob
import argparse

def process_video_batch_resync(video_dir, asd_dir, output_dir, target_fps=25):
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"🔍 Found {len(video_files)} videos. Syncing to {target_fps}fps...")
    
    for video_path in video_files:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        meta_path = os.path.join(asd_dir, f"{video_basename}_meta.pkl")
        
        if not os.path.exists(meta_path):
            continue
            
        with open(meta_path, 'rb') as f:
            tracks_meta = pickle.load(f)
            
        asd_files = glob.glob(os.path.join(asd_dir, f"{video_basename}_track*.npy"))
        if not asd_files: continue
            
        asd_lengths = [(f, np.load(f).shape[-1]) for f in asd_files]
        asd_lengths.sort(key=lambda x: x[1], reverse=True)
        
        N_speakers = min(2, len(asd_lengths))
        top_asd_files = [x[0] for x in asd_lengths[:N_speakers]]
        
        top_tracks = {}
        for f in top_asd_files:
            track_id = int(os.path.splitext(os.path.basename(f))[0].split("_track")[-1])
            top_tracks[track_id] = np.load(f)
            
        # 1. 원본 영상 정보 획득
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS) # 예: 29.97 or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 2. 출력 영상은 25fps로 고정 (모델 결과와 일치)
        output_path = os.path.join(output_dir, f"{video_basename}_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        print(f"🎥 {video_basename}: Source {source_fps:.2f}fps -> Target {target_fps}fps Syncing...")

        frame_idx = 0
        target_frame_idx = 0  # 🚀 [추가] 25fps 기준 몇 번째 프레임을 써야 하는지 추적

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 원본 영상의 시간(초)
            current_time_sec = frame_idx / source_fps
            meta_idx = int(round(current_time_sec * target_fps))
            
            for track_id, scores in top_tracks.items():
                if track_id not in tracks_meta: continue
                
                meta = tracks_meta[track_id]
                # 메타데이터의 frame 리스트는 25fps 기준으로 저장되어 있음
                frames_list = meta['frame'].tolist() if isinstance(meta['frame'], np.ndarray) else meta['frame']
                
                # 계산된 meta_idx가 메타데이터 존재 범위 내에 있는지 확인
                if meta_idx in frames_list:
                    # frames_list 내에서 meta_idx가 몇 번째 위치에 있는지 확인 (bbox 추출용)
                    try:
                        list_pos = frames_list.index(meta_idx)
                        
                        if list_pos < len(scores):
                            bbox = meta['bbox'][list_pos]
                            score = scores[list_pos]
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            color = (0, 255, 0) if score > 0.0 else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            text = f"ID:{track_id} | ASD:{score:.1f}"
                            cv2.putText(frame, text, (x1, max(20, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except ValueError:
                        pass # 해당 meta_idx가 리스트에 없는 경우 스킵
            
            # 3. 25fps 출력 영상에 현재 프레임을 씀
            # 참고: 원본이 30fps인 경우 30개 프레임을 읽어서 25fps 비디오에 다 쓰면 
            # 영상이 미세하게 슬로우 모션이 됩니다. 
            # 완벽한 싱크를 위해선 target_fps 주기에 맞는 프레임만 out.write해야 합니다.
            
            # 🚀 [수정됨] 확실한 타임스탬프 기반 프레임 기록 (Frame Dropping)
            # 타겟 영상(25fps)에서 현재 써야 할 프레임의 시간 위치
            expected_target_time = target_frame_idx / target_fps
            
            # 원본 영상의 시간이 타겟 시간 이상이 될 때만 프레임을 기록
            if current_time_sec >= expected_target_time:
                out.write(frame)
                target_frame_idx += 1  # 다음 타겟 프레임 기록 준비
                
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"✅ Finished: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--asd_dir", required=True)
    parser.add_argument("--output_dir", default="visualized_resync")
    args = parser.parse_args()
    
    process_video_batch_resync(args.video_dir, args.asd_dir, args.output_dir)
