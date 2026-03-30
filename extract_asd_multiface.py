import numpy as np
import os

# 가설: face_tracker가 반환한 전체 얼굴 궤적 리스트가 'all_tracks' 라고 가정
# all_tracks = [{"track": {"frame": [...], "bbox": [...]}}, ...]

min_frames_threshold = 25  # 1초(25fps 기준) 미만으로 등장한 얼굴은 노이즈로 간주하고 무시

for track_id, track_info in enumerate(all_tracks):
    frames = track_info["track"]["frame"]
    
    # 너무 짧게 스쳐 지나간 얼굴은 추출 생략
    if len(frames) < min_frames_threshold:
        continue
        
    # LR-ASD 모델 추론 (기존 코드의 모델 포워드 부분)
    # asd_score shape: (1, T_video)
    asd_score = lr_asd_model(track_info) 
    
    # 텐서를 numpy로 변환 후 1D 배열로 평탄화
    asd_score_np = asd_score.squeeze().cpu().numpy()
    
    # [수정된 부분] 파일명에 track_id를 부여하여 모두 저장
    # 예: mix_track0.npy, mix_track1.npy ...
    save_path = os.path.join(output_dir, f"{video_name_no_ext}_track{track_id}.npy")
    np.save(save_path, asd_score_np)
    print(f"Saved ASD score for Track {track_id} -> {save_path}")
