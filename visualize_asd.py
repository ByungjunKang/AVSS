import cv2
import numpy as np
import os
import pickle
import glob
import argparse

def visualize_video(video_path, meta_path, npy_dir, output_path, fps=25):
    # 1. 메타데이터 및 스코어 로드
    with open(meta_path, 'rb') as f:
        tracks_meta = pickle.load(f)
        
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    scores_dict = {}
    
    for track_id in tracks_meta.keys():
        npy_path = os.path.join(npy_dir, f"{video_basename}_track{track_id}.npy")
        if os.path.exists(npy_path):
            scores_dict[track_id] = np.load(npy_path)
            
    # 2. 비디오 라이터 세팅
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. 현재 프레임에 존재하는 모든 얼굴 박스 그리기
        for track_id, meta in tracks_meta.items():
            if track_id not in scores_dict:
                continue
                
            frames_list = meta['frame'].tolist() if isinstance(meta['frame'], np.ndarray) else meta['frame']
            
            if frame_idx in frames_list:
                # 리스트 내 인덱스를 찾아 바운딩 박스와 스코어 매칭
                idx = frames_list.index(frame_idx)
                
                # 영상 길이와 스코어 길이가 미세하게 다를 수 있는 에지 케이스 방어
                if idx >= len(scores_dict[track_id]):
                    idx = len(scores_dict[track_id]) - 1
                    
                bbox = meta['bbox'][idx]
                score = scores_dict[track_id][idx]
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # 시각적 피드백: 로짓 기준 0 이상이면 초록색(Active), 미만이면 빨간색(Inactive)
                color = (0, 255, 0) if score > 0.0 else (0, 0, 255)
                
                # 박스 및 텍스트 오버레이
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"ID:{track_id} | ASD:{score:.1f}"
                cv2.putText(frame, text, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"✅ Visualization complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to original .mp4")
    parser.add_argument("--meta_path", required=True, help="Path to extracted _meta.pkl")
    parser.add_argument("--npy_dir", required=True, help="Directory containing _trackX.npy files")
    parser.add_argument("--output_path", default="output_viz.mp4", help="Path to save visualization")
    args = parser.parse_args()
    
    visualize_video(args.video_path, args.meta_path, args.npy_dir, args.output_path)
