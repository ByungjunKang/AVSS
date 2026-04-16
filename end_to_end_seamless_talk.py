import sys
import os
import argparse
import cv2
import torch
import numpy as np
import mediapipe as mp
import librosa
import soundfile as sf
from TTS.api import TTS

# ----------------------------------------------------
# Auto-AVSR 공식 모듈 임포트 (auto_avsr 최상단 폴더에서 실행 필수)
# ----------------------------------------------------
try:
    from lightning import ModelModule
    from datamodule.transforms import VideoTransform
except ImportError:
    print("[오류] auto_avsr 폴더 최상단에서 스크립트를 실행해주세요.")
    sys.exit(1)

class SeamlessSilentTalkSystem:
    def __init__(self, vsr_ckpt_path="vsr_trlrs2lrs3vox2avsp_base.pth"):
        print("1. 모델 초기화 중...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ==========================================
        # 1. VSR 모델 로드 (Auto-AVSR Lightning Module)
        # ==========================================
        print(f"   - VSR 체크포인트 로드: {vsr_ckpt_path}")
        parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args(args=[])
        setattr(args, 'modality', 'video') # 비디오 모드 강제 설정
        
        # 모델 및 가중치(State Dict) 로딩
        ckpt = torch.load(vsr_ckpt_path, map_location=lambda storage, loc: storage)
        self.vsr_model = ModelModule(args)
        self.vsr_model.model.load_state_dict(ckpt)
        self.vsr_model.eval()
        self.vsr_model.to(self.device)
        
        # 전처리 모듈 (Auto-AVSR 내부 변환기)
        self.video_transform = VideoTransform(subset="test")
        
        # ==========================================
        # 2. MediaPipe Face Mesh 로드 (전처리용)
        # ==========================================
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        
        # ==========================================
        # 3. XTTS 모델 로드 (개인화 음성 합성용)
        # ==========================================
        print("   - XTTS-v2 모델 로드 (시간이 약간 소요될 수 있습니다)")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print("초기화 완료.\n")

    def extract_lips_from_video(self, video_path):
        """MediaPipe를 이용해 영상에서 입술만 96x96 Grayscale로 크롭"""
        print(f"2. 비디오 전처리 중: {video_path}")
        cap = cv2.VideoCapture(video_path)
        cropped_lips = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                
                # 입술 랜드마크 추출 (61, 291: 입꼬리 / 0, 17: 위아래 입술)
                x_coords = [int(l.x * w) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
                y_coords = [int(l.y * h) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
                
                margin = 15
                x_min, x_max = max(0, min(x_coords)-margin), min(w, max(x_coords)+margin)
                y_min, y_max = max(0, min(y_coords)-margin), min(h, max(y_coords)+margin)
                
                lip_crop = frame[y_min:y_max, x_min:x_max]
                lip_crop_resized = cv2.resize(lip_crop, (96, 96))
                
                # Auto-AVSR 학습 데이터 기준에 맞추기 위해 흑백(Grayscale) 변환
                lip_gray = cv2.cvtColor(lip_crop_resized, cv2.COLOR_BGR2GRAY)
                lip_gray = np.expand_dims(lip_gray, axis=-1) # (96, 96, 1)로 맞춤
                cropped_lips.append(lip_gray)
            else:
                # 얼굴을 놓쳤을 경우 검은색 빈 프레임으로 채움 (시간 동기화 유지)
                cropped_lips.append(np.zeros((96, 96, 1), dtype=np.uint8))
                
        cap.release()
        print(f"   - 총 {len(cropped_lips)} 프레임 추출 완료.")
        return np.array(cropped_lips) # (T, 96, 96, 1)

    def extract_text_from_lips(self, lip_frames):
        """Auto-AVSR 모델을 통과시켜 텍스트 추론"""
        print("3. VSR 텍스트 디코딩 수행 중...")
        
        # 모델 입력 형태로 텐서 변환: (T, H, W, C) -> (T, C, H, W)
        video_tensor = torch.tensor(lip_frames).float().permute(0, 3, 1, 2)
        video_tensor = self.video_transform(video_tensor).unsqueeze(0).to(self.device) # 배치 차원 추가
        
        with torch.no_grad():
            transcript = self.vsr_model(video_tensor)
            
        # 리스트로 반환될 경우 첫 번째 요소 추출
        if isinstance(transcript, list):
            transcript = transcript[0]
            
        print(f"   - VSR 결과: '{transcript}'")
        return transcript

    def generate_personalized_ssi(self, text, reference_audio_path):
        """XTTS를 이용해 Y_audio의 음색을 복제한 합성음(Y_ssi) 생성"""
        print(f"4. 개인화 TTS 합성 중 (원본 음색 복제)...")
        output_path = "temp_y_ssi.wav"
        
        # XTTS: 분리된 오디오(reference)의 화자 특성을 뽑아내어 텍스트를 합성
        self.tts.tts_to_file(text=text, 
                             speaker_wav=reference_audio_path, 
                             language="en", 
                             file_path=output_path)
        
        y_ssi, sr = librosa.load(output_path, sr=16000)
        return y_ssi

    def run_pipeline(self, raw_video_path, separated_audio_path, output_mixed_path):
        """전체 통합 파이프라인 실행"""
        print("========================================")
        print("교차 모달 상관관계 기반 음질 복원 시스템 (PoC)")
        print("========================================")
        
        # 1. 1차 분리된 오디오(Y_audio) 로드
        y_audio, sr = librosa.load(separated_audio_path, sr=16000)
        
        # 2. 입술 크롭 (MediaPipe)
        lip_frames = self.extract_lips_from_video(raw_video_path)
        
        # 3. 무성/소음 구간 텍스트 추출 (Auto-AVSR)
        text = self.extract_text_from_lips(lip_frames)
        
        # 4. 개인화 음성 합성 (Y_ssi 생성)
        y_ssi = self.generate_personalized_ssi(text, separated_audio_path)
        
        # 5. 상관관계 기반 Soft-Mixing (가상 시나리오)
        # 본 PoC에서는 극한 소음을 가정하여 합성음 비중(Alpha)을 80%로 높게 설정합니다.
        alpha = 0.8 
        print(f"5. 동적 믹싱(Soft-Mixing) 적용: Alpha = {alpha}")
        
        min_len = min(len(y_audio), len(y_ssi))
        y_audio_matched = y_audio[:min_len]
        y_ssi_matched = y_ssi[:min_len]
        
        y_out = (1.0 - alpha) * y_audio_matched + (alpha * y_ssi_matched)
        
        # 6. 저장
        sf.write(output_mixed_path, y_out, 16000)
        print(f"\n✅ 파이프라인 완료! 최종 파일: {output_mixed_path}")

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    # 다운받은 "VSR" 모델의 정확한 경로를 지정해주세요.
    ckpt = "vsr_trlrs2lrs3vox2avsp_base.pth" 
    
    system = SeamlessSilentTalkSystem(vsr_ckpt_path=ckpt)
    
    # TC(화상회의) 영상이나 내 얼굴이 나오는 MP4 파일
    raw_video = "sample_tc_video.mp4" 
    
    # 위 영상과 대응되며, 분리 모델을 한 번 거친 노이즈 낀 타겟 음성(WAV)
    separated_audio = "separated_target_audio.wav"
    
    output_audio = "final_seamless_mixed.wav"
    
    system.run_pipeline(raw_video, separated_audio, output_audio)
