import sys
import os
import argparse
import cv2
import torch
import numpy as np
import mediapipe as mp
import librosa
import soundfile as sf
# 기존 TTS import 주석 처리 또는 삭제
# from TTS.api import TTS 

# 로컬 로딩을 위한 XTTS Core API 임포트
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

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
        # 3. XTTS 모델 로드 (완전 오프라인 로컬 방식)
        # ==========================================
        print("   - XTTS-v2 로컬 모델 로드 중 (사내망 모드)...")
        
        # 아까 만든 로컬 폴더 경로 지정
        local_model_path = "./xtts_local_model" 
        
        # 설정 파일 로드
        self.xtts_config = XttsConfig()
        self.xtts_config.load_json(os.path.join(local_model_path, "config.json"))
        
        # 빈 모델 껍데기 생성 후 가중치 덮어씌우기
        self.tts = Xtts.init_from_config(self.xtts_config)
        self.tts.load_checkpoint(self.xtts_config, checkpoint_dir=local_model_path, eval=True)
        self.tts.to(self.device)
        print("초기화 완료.\n")

    def extract_lips_from_video(self, video_path):
        """면적이 가장 큰 타겟 화자의 입술만 96x96으로 크롭 (차원 수정)"""
        print(f"2. 비디오 전처리 중: {video_path} (가장 큰 얼굴 추적)")
        cap = cv2.VideoCapture(video_path)
        cropped_lips = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                target_landmarks = None
                max_area = 0
                
                for face_landmarks in results.multi_face_landmarks:
                    x_coords_all = [l.x for l in face_landmarks.landmark]
                    y_coords_all = [l.y for l in face_landmarks.landmark]
                    face_area = (max(x_coords_all) - min(x_coords_all)) * (max(y_coords_all) - min(y_coords_all))
                    
                    if face_area > max_area:
                        max_area = face_area
                        target_landmarks = face_landmarks.landmark
                
                landmarks = target_landmarks
                
                x_coords = [int(l.x * w) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
                y_coords = [int(l.y * h) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
                
                margin = 15
                x_min, x_max = max(0, min(x_coords)-margin), min(w, max(x_coords)+margin)
                y_min, y_max = max(0, min(y_coords)-margin), min(h, max(y_coords)+margin)
                
                lip_crop = frame[y_min:y_max, x_min:x_max]
                
                if lip_crop.size == 0:
                    cropped_lips.append(np.zeros((96, 96), dtype=np.uint8))
                    continue
                    
                lip_crop_resized = cv2.resize(lip_crop, (96, 96))
                lip_gray = cv2.cvtColor(lip_crop_resized, cv2.COLOR_BGR2GRAY)
                
                # [수정포인트] 차원 확장(expand_dims)을 제거하고 순수 (96, 96) 2D 이미지로 저장
                cropped_lips.append(lip_gray)
            else:
                cropped_lips.append(np.zeros((96, 96), dtype=np.uint8))
                
        cap.release()
        print(f"   - 총 {len(cropped_lips)} 프레임 입술 추출 완료.")
        return np.array(cropped_lips) # 최종 형태: (T, 96, 96)

    def extract_text_from_lips(self, lip_frames):
        """torchvision 전처리기를 배제하고 수동으로 차원을 제어하여 에러 완벽 회피"""
        print("3. VSR 텍스트 디코딩 수행 중...")
        
        # 1. 현재 lip_frames는 (T, 96, 96) 형태의 Numpy 배열입니다.
        # Auto-AVSR 모델은 88x88 크기를 요구하므로 수동으로 CenterCrop(가장자리 4픽셀씩 제거)
        lip_crop = lip_frames[:, 4:92, 4:92] # Shape: (T, 88, 88)
        
        # 2. PyTorch Tensor로 변환 및 [0, 1] 스케일링
        video_tensor = torch.tensor(lip_crop).float() / 255.0
        
        # 3. Grayscale 정규화 (Auto-AVSR 논문의 공식 평균/표준편차 적용)
        video_tensor = (video_tensor - 0.421) / 0.165
        
        # 4. 모델이 요구하는 정확한 5차원(B, C, T, H, W) 생성
        # 현재 (T, 88, 88) 차원의 맨 앞에 배치(B=1)와 채널(C=1) 차원을 2개 추가합니다.
        video_tensor = video_tensor.unsqueeze(0).unsqueeze(0) 
        # 최종 Shape 보장: (1, 1, T, 88, 88)
        
        video_tensor = video_tensor.to(self.device)
        
        # 5. 시퀀스 길이 정보 생성
        video_lengths = torch.tensor([video_tensor.size(2)]).to(self.device)
        
        # 6. 모델 추론
        with torch.no_grad():
            try:
                # 완벽한 5차원 텐서가 들어가므로 xs_pad.size() 언패킹 에러가 사라집니다.
                transcript = self.vsr_model(video_tensor, video_lengths)
            except Exception as e:
                # 만약 PyTorch Lightning 래퍼 내부에서 충돌이 날 경우를 대비한 직접 디코딩 호출
                if hasattr(self.vsr_model, "model") and hasattr(self.vsr_model.model, "recognize"):
                    transcript = self.vsr_model.model.recognize(video_tensor[0, 0])
                else:
                    raise e
                    
        # 결과물이 리스트나 튜플로 나올 경우 텍스트만 깔끔하게 추출
        if isinstance(transcript, list) or isinstance(transcript, tuple):
            transcript = transcript[0]
            
        print(f"   - VSR 결과: '{transcript}'")
        return transcript

    def generate_personalized_ssi(self, text, reference_audio_path):
        """XTTS를 이용해 Y_audio의 음색을 복제한 합성음(Y_ssi) 생성"""
        print(f"4. 개인화 TTS 합성 중 (원본 음색 복제)...")
        
        # XTTS Core API를 사용하여 메모리 내에서 직접 오디오 생성
        outputs = self.tts.synthesize(
            text,
            self.xtts_config,
            speaker_wav=reference_audio_path,
            gpt_cond_len=3,
            language="en" # 한국어 테스트 시 "ko"로 변경
        )
        
        # XTTS의 기본 출력 샘플링 레이트는 24000Hz 입니다.
        y_ssi_24k = np.array(outputs["wav"])
        
        # 전체 시스템 파이프라인 동기화(16000Hz)를 위해 리샘플링 수행
        y_ssi_16k = librosa.resample(y_ssi_24k, orig_sr=24000, target_sr=16000)
        
        return y_ssi_16k

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
