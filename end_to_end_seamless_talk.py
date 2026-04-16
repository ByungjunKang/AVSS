import cv2
import torch
import numpy as np
import mediapipe as mp
import librosa
import soundfile as sf
from TTS.api import TTS

# [가정] Auto-AVSR 저장소의 추론 파이프라인 임포트
# from auto_avsr.models import AVSRModel

class EndToEndSeamlessTalk:
    def __init__(self):
        print("1. 모델 초기화 중...")
        # 1. MediaPipe Face Mesh 로드 (입술 크롭용)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        
        # 2. VSR 모델 로드 (Auto-AVSR)
        # self.vsr_model = AVSRModel.from_pretrained("auto_avsr_lrs3.pth")
        
        # 3. XTTS 모델 로드 (개인화 TTS, 화자 음색 복제용)
        print("   - XTTS-v2 모델 로딩 (시간이 약간 소요될 수 있습니다)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print("초기화 완료.\n")

    def extract_lips_from_video(self, video_path):
        """원본 MP4에서 입술 영역만 96x96으로 자동 크롭 (Auto Preprocessing)"""
        print(f"2. 비디오 전처리 중: {video_path}")
        cap = cv2.VideoCapture(video_path)
        cropped_lips = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # MediaPipe 처리를 위해 RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                
                # 입술 좌표 추출 (상하좌우 끝점)
                x_coords = [int(l.x * w) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
                y_coords = [int(l.y * h) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
                
                # Bounding Box 여유 공간(Margin) 추가
                margin = 15
                x_min, x_max = max(0, min(x_coords)-margin), min(w, max(x_coords)+margin)
                y_min, y_max = max(0, min(y_coords)-margin), min(h, max(y_coords)+margin)
                
                # 크롭 및 96x96 리사이즈 (Auto-AVSR 입력 규격)
                lip_crop = frame[y_min:y_max, x_min:x_max]
                lip_crop_resized = cv2.resize(lip_crop, (96, 96))
                
                # 흑백(Grayscale) 변환이 필요할 경우: cv2.cvtColor(lip_crop_resized, cv2.COLOR_BGR2GRAY)
                cropped_lips.append(lip_crop_resized)
                
        cap.release()
        print(f"   - 총 {len(cropped_lips)} 프레임 입술 추출 완료.")
        # 모델에 넣기 위해 numpy 배열로 변환 (Time, Channel, Height, Width)
        return np.array(cropped_lips)

    def extract_text_from_lips(self, lip_frames):
        """VSR 모델을 통한 텍스트 디코딩"""
        print("3. VSR 텍스트 디코딩 수행 중...")
        # 실제 구현 시: predicted_text = self.vsr_model.transcribe(lip_frames)
        predicted_text = "hello how are you doing today" # PoC용
        print(f"   - 추출된 텍스트: '{predicted_text}'")
        return predicted_text

    def generate_personalized_ssi(self, text, reference_audio_path):
        """XTTS를 이용해 Y_audio의 음색을 복제한 Y_ssi 생성"""
        print(f"4. 개인화 TTS 합성 중 (원본 음색 복제)...")
        output_path = "temp_y_ssi.wav"
        
        # XTTS: 입력된 텍스트를, reference_audio_path의 목소리와 똑같이 합성
        self.tts.tts_to_file(text=text, 
                             speaker_wav=reference_audio_path, 
                             language="en", 
                             file_path=output_path)
        
        # 합성된 오디오 로드 (Librosa)
        y_ssi, sr = librosa.load(output_path, sr=16000)
        return y_ssi

    def calculate_alpha(self, y_audio, lip_frames):
        """교차 모달 상관관계 분석 (가상)"""
        print("5. 오디오-비전 상관관계 분석 기반 Alpha 산출 중...")
        # 특허 로직: y_audio의 에너지와 lip_frames의 변화량간 Correlation 분석
        # 본 PoC에서는 극한 소음/분리 실패 상황을 가정하여 높은 Alpha 값 반환
        alpha = 0.85 
        print(f"   - 산출된 Alpha 값: {alpha:.2f}")
        return alpha

    def run_pipeline(self, raw_video_path, separated_audio_path, output_mixed_path):
        """메인 실행 파이프라인"""
        print("========================================")
        print("Seamless Silent Talk Pipeline 시작")
        print("========================================")
        
        # 1. 오디오 로드 (Y_audio, 보통 분리 모델 TIGER를 거친 결과물)
        y_audio, sr = librosa.load(separated_audio_path, sr=16000)
        
        # 2. 비디오 전처리 (MP4 -> 96x96 Lip Frames)
        lip_frames = self.extract_lips_from_video(raw_video_path)
        
        # 3. 입술 읽기 (VSR)
        text = self.extract_text_from_lips(lip_frames)
        
        # 4. 개인화 음성 합성 (Y_ssi 생성 - Y_audio를 레퍼런스로 사용!)
        y_ssi = self.generate_personalized_ssi(text, separated_audio_path)
        
        # 5. 가중치 산출 및 Soft-Mixing
        alpha = self.calculate_alpha(y_audio, lip_frames)
        
        # 길이 맞추기
        min_len = min(len(y_audio), len(y_ssi))
        y_audio_matched = y_audio[:min_len]
        y_ssi_matched = y_ssi[:min_len]
        
        # 특허 수식 적용: Y_out = (1-α) * Y_audio + α * Y_ssi
        y_out = (1.0 - alpha) * y_audio_matched + (alpha * y_ssi_matched)
        
        # 6. 최종 파일 저장
        sf.write(output_mixed_path, y_out, 16000)
        print(f"\n✅ 최종 믹싱 완료! 결과물 저장: {output_mixed_path}")
        return y_out

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    system = EndToEndSeamlessTalk()
    
    # [입력] 
    # 1. 얼굴이 나오는 원본 MP4 영상
    # 2. TIGER 모델 등으로 1차 분리된 타겟 화자의 음성 (노이즈가 껴있거나 끊김이 있을 수 있음)
    raw_video = "sample_tc_video.mp4" 
    separated_audio = "separated_target_audio.wav"
    output_audio = "final_seamless_mixed.wav"
    
    # 파이프라인 실행
    # (주의: 실제 파일 경로가 있어야 동작합니다. 없으면 에러가 발생합니다.)
    try:
        final_audio = system.run_pipeline(raw_video, separated_audio, output_audio)
    except Exception as e:
        print(f"[안내] 파일이 없거나 오류 발생: {e}\n위의 raw_video와 separated_audio 경로를 실제 파일에 맞게 수정하세요.")
