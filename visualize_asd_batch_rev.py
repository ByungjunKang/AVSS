import cv2
import numpy as np
import os
import pickle
import glob
import argparse

def process_video_batch_top2(video_dir, asd_dir, output_dir, target_fps=25):
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"🔍 Found {len(video_files)} videos. Visualizing Top-2 Tracks...")
    
    for video_path in video_files:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        meta_path = os.path.join(asd_dir, f"{video_basename}_meta.pkl")
        
        if not os.path.exists(meta_path):
            print(f"⚠️ Meta file missing for {video_basename}. Skipping...")
            continue
            
        with open(meta_path, 'rb') as f:
            tracks_meta = pickle.load(f)
            
        # 1. 파일명에서 직접 track0, track1 찾기 (이미 전처리에 의해 필터링됨)
        top_tracks = {}
        for tid in [0, 1]:
            npy_path = os.path.join(asd_dir, f"{video_basename}_track{tid}.npy")
            if os.path.exists(npy_path):
                top_tracks[tid] = np.load(npy_path)
        
        if not top_tracks:
            print(f"⚠️ No track .npy files found for {video_basename}. Skipping...")
            continue
            
        # 2. 영상 정보 및 라이터 세팅
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = os.path.join(output_dir, f"{video_basename}_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        print(f"🎥 {video_basename}: Source {source_fps:.2f}fps -> Target {target_fps}fps (IDs: {list(top_tracks.keys())})")

        frame_idx = 0
        target_frame_idx = 0 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 원본 영상의 절대 시간(초)
            current_time_sec = frame_idx / source_fps
            # 25fps로 구성된 '절대 시간 캔버스'(.npy)의 인덱스 역산
            meta_idx = int(round(current_time_sec * target_fps))
            
            # 3. 저장된 Top-2 트랙(0 또는 1)에 대해서만 박스 그리기
            for track_id, full_scores in top_tracks.items():
                # 전처리 시 저장한 meta 정보가 있는지 확인
                if track_id not in tracks_meta:
                    continue
                
                meta = tracks_meta[track_id]
                # meta['frame']은 이 화자가 실제로 화면에 나타났던 25fps 기준 프레임 번호들
                frames_list = meta['frame'].tolist() if isinstance(meta['frame'], np.ndarray) else meta['frame']
                
                # 현재 시간이 이 화자가 화면에 존재하던 시간대라면
                if meta_idx in frames_list:
                    # npy 파일(full_scores)은 전체 길이를 가지고 있으므로 meta_idx로 직접 접근
                    if meta_idx < len(full_scores):
                        score = full_scores[meta_idx]
                        
                        # bbox 정보를 가져오기 위해 frames_list 내의 상대 위치 찾기
                        try:
                            list_pos = frames_list.index(meta_idx)
                            bbox = meta['bbox'][list_pos]
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            # 발화 중이면 초록, 아니면 빨강 (로짓 0.0 기준)
                            color = (0, 255, 0) if score > 0.0 else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            text = f"ID:{track_id} | ASD:{score:.1f}"
                            cv2.putText(frame, text, (x1, max(20, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        except ValueError:
                            pass

            # 4. 타임스탬프 기반 프레임 기록 (0초 영상 버그 해결 로직)
            expected_target_time = target_frame_idx / target_fps
            if current_time_sec >= expected_target_time:
                out.write(frame)
                target_frame_idx += 1
                
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--asd_dir", required=True)
    parser.add_argument("--output_dir", default="visualized_top2")
    args = parser.parse_args()
    
    process_video_batch_top2(args.video_dir, args.asd_dir, args.output_dir)
