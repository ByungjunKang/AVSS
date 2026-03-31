import cv2
import numpy as np
import os
import pickle
import glob
import argparse

def process_video_batch(video_dir, asd_dir, output_dir, fps=25):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 처리할 비디오 파일 목록 순회
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"🔍 Found {len(video_files)} videos in {video_dir}")
    
    for video_path in video_files:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        # 2. 메타데이터(.pkl) 파일 확인
        meta_path = os.path.join(asd_dir, f"{video_basename}_meta.pkl")
        if not os.path.exists(meta_path):
            print(f"⚠️ Warning: Meta file missing for {video_basename}. Skipping...")
            continue
            
        with open(meta_path, 'rb') as f:
            tracks_meta = pickle.load(f)
            
        # 3. ASD 결과 파일(*_trackX.npy) 찾기 및 Top-2 필터링
        asd_files = glob.glob(os.path.join(asd_dir, f"{video_basename}_track*.npy"))
        if not asd_files:
            print(f"⚠️ Warning: No ASD files found for {video_basename}. Skipping...")
            continue
            
        # 길이 기준 내림차순 정렬
        asd_lengths = [(f, np.load(f).shape[-1]) for f in asd_files]
        asd_lengths.sort(key=lambda x: x[1], reverse=True)
        
        # 🚀 [핵심] 가장 긴 얼굴 최대 2개만 선택
        N_speakers = min(2, len(asd_lengths))
        top_asd_files = [x[0] for x in asd_lengths[:N_speakers]]
        
        # 선택된 파일에서 track_id와 score 로드
        top_tracks = {}
        for f in top_asd_files:
            track_id = int(os.path.splitext(os.path.basename(f))[0].split("_track")[-1])
            top_tracks[track_id] = np.load(f)
            
        print(f"\n🎥 Processing {video_basename}: Selected {N_speakers} longest tracks -> IDs: {list(top_tracks.keys())}")
        
        # 4. 비디오 라이터 세팅
        output_path = os.path.join(output_dir, f"{video_basename}_viz.mp4")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 5. 프레임 순회하며 바운딩 박스 오버레이
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 🚀 [핵심] 필터링된 Top-2 Track ID에 대해서만 그리기 수행
            for track_id, scores in top_tracks.items():
                if track_id not in tracks_meta:
                    continue
                    
                meta = tracks_meta[track_id]
                frames_list = meta['frame'].tolist() if isinstance(meta['frame'], np.ndarray) else meta['frame']
                
                # 현재 프레임에 이 화자가 존재하는 경우
                if frame_idx in frames_list:
                    idx = frames_list.index(frame_idx)
                    
                    # 영상 길이와 스코어 길이가 미세하게 다를 수 있는 에지 케이스 방어
                    if idx >= len(scores):
                        idx = len(scores) - 1
                        
                    bbox = meta['bbox'][idx]
                    score = scores[idx]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # 시각적 피드백: 로짓 기준 0 이상이면 초록색(Active), 미만이면 빨간색(Inactive)
                    color = (0, 255, 0) if score > 0.0 else (0, 0, 255)
                    
                    # 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 텍스트 배경 (가독성 향상)
                    text = f"ID:{track_id} | ASD:{score:.1f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw, y1), color, -1)
                    
                    # 텍스트 오버레이 (배경이 초록/빨강이므로 글씨는 검정색)
                    cv2.putText(frame, text, (x1, max(20, y1 - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"✅ Saved visualization: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch visualize Top-2 ASD tracks on videos.")
    parser.add_argument("--video_dir", required=True, help="Directory containing original .mp4 videos")
    parser.add_argument("--asd_dir", required=True, help="Directory containing extracted .npy and .pkl files")
    parser.add_argument("--output_dir", default="visualized_videos", help="Directory to save output .mp4 videos")
    args = parser.parse_args()
    
    process_video_batch(args.video_dir, args.asd_dir, args.output_dir)
