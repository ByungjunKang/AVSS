import cv2
import numpy as np
import os

def visualize_asd_tracks(video_path, all_tracks, asd_scores_dict, output_path, fps=25):
    """
    video_path: 원본 영상 (.mp4)
    all_tracks: 트래커가 반환한 전체 궤적 리스트 (원본 프레임 인덱스와 bbox 포함)
    asd_scores_dict: {track_id: np.array(asd_scores)} 형태의 딕셔너리
    output_path: 결과물이 저장될 영상 경로
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # MP4 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 현재 프레임에 해당하는 모든 트랙 검사
        for track_id, track_info in enumerate(all_tracks):
            if track_id not in asd_scores_dict:
                continue # 필터링되어 점수가 없는 트랙은 패스
                
            frames_list = track_info["track"]["frame"]
            
            # 이 화자가 현재 프레임에 존재하는 경우
            if frame_idx in frames_list:
                # 리스트 내 인덱스 찾기
                idx = frames_list.index(frame_idx)
                bbox = track_info["track"]["bbox"][idx]
                score = asd_scores_dict[track_id][idx]
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # 시각적 피드백: Score가 0 이상(Active)이면 초록색, 미만(Inactive)이면 빨간색
                # (만약 이미 Sigmoid를 통과한 확률값이라면 기준을 0.5로 설정하세요)
                color = (0, 255, 0) if score > 0.0 else (0, 0, 255)
                
                # Bounding Box 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # ID 및 Score 텍스트 오버레이
                text = f"ID:{track_id} Score:{score:.2f}"
                cv2.putText(frame, text, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"✅ Visualization saved to {output_path}")

# --- 사용 예시 ---
# asd_scores_dict = {
#     0: np.load("mix_track0.npy"),
#     1: np.load("mix_track1.npy")
# }
# visualize_asd_tracks("test_video.mp4", all_tracks, asd_scores_dict, "test_video_asd_viz.mp4")
