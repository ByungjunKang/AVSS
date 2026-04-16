import sys
import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import librosa
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# [주의] auto_avsr 최상단 폴더에서 실행해야 합니다.
try:
    from lightning import ModelModule
except ImportError:
    print("[오류] auto_avsr 폴더 최상단에서 스크립트를 실행해주세요.")
    sys.exit(1)

class DualSeamlessTalkPoC:
    def __init__(self, vsr_ckpt_path="vsr_trlrs2lrs3vox2avsp_base.pth"):
        print("1. 모델 초기화 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ==========================================
        # 1. VSR 모델 로드 (실제 추론 모드 활성화)
        # ==========================================
        print(f"   - VSR 체크포인트 로드: {vsr_ckpt_path}")
        import argparse
        parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args(args=[])
        setattr(args, 'modality', 'video') 
        
        ckpt = torch.load(vsr_ckpt_path, map_location=lambda storage, loc: storage)
        self.vsr_model = ModelModule(args)
        self.vsr_model.model.load_state_dict(ckpt)
        self.vsr_model.eval()
        self.vsr_model.to(self.device)
        
        # ==========================================
        # 2. MediaPipe Face Mesh 로드 (최대 5명까지 감지 후 필터링)
        # ==========================================
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)
        
        # ==========================================
        # 3. XTTS 모델 로드 (사내망 로컬 방식)
        # ==========================================
        print("   - XTTS-v2 로컬 모델 로드 중...")
        local_model_path = "./xtts_local_model" 
        self.xtts_config = XttsConfig()
        self.xtts_config.load_json(os.path.join(local_model_path, "config.json"))
        self.tts = Xtts.init_from_config(self.xtts_config)
        self.tts.load_checkpoint(self.xtts_config, checkpoint_dir=local_model_path, eval=True)
        self.tts.to(self.device)
        print("초기화 완료.\n")

    def _get_lip_crop(self, landmarks, frame):
        """내부 헬퍼 함수: 랜드마크에서 입술만 잘라내기"""
        h, w, _ = frame.shape
        x_coords = [int(l.x * w) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
        y_coords = [int(l.y * h) for i, l in enumerate(landmarks) if i in [61, 291, 0, 17]]
        margin = 15
        x_min, x_max = max(0, min(x_coords)-margin), min(w, max(x_coords)+margin)
        y_min, y_max = max(0, min(y_coords)-margin), min(h, max(y_coords)+margin)
        
        lip_crop = frame[y_min:y_max, x_min:x_max]
        if lip_crop.size == 0:
            return np.zeros((96, 96), dtype=np.uint8)
        
        lip_crop_resized = cv2.resize(lip_crop, (96, 96))
        return cv2.cvtColor(lip_crop_resized, cv2.COLOR_BGR2GRAY)

    def extract_dual_lips_from_video(self, video_path):
        """2명의 화자를 L/R로 나누어 각각 입술 프레임 추출"""
        print(f"2. 비디오 전처리 중 (L/R 듀얼 화자 추적): {video_path}")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        lips_L = []
        lips_R = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)
            
            frame_lip_L = np.zeros((96, 96), dtype=np.uint8)
            frame_lip_R = np.zeros((96, 96), dtype=np.uint8)
            
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                faces_info = []
                
                # 얼굴 넓이와 중앙 X 좌표 계산
                for lm in results.multi_face_landmarks:
                    x_coords = [l.x * w for l in lm.landmark]
                    y_coords = [l.y * h for l in lm.landmark]
                    area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                    center_x = sum(x_coords) / len(x_coords)
                    faces_info.append({'lm': lm.landmark, 'area': area, 'cx': center_x})
                
                # 면적 기준 상위 2명 추출 후, X좌표(왼쪽/오른쪽) 기준으로 정렬
                faces_info.sort(key=lambda x: x['area'], reverse=True)
                top_faces = faces_info[:2]
                top_faces.sort(key=lambda x: x['cx']) 
                
                if len(top_faces) == 2:
                    frame_lip_L = self._get_lip_crop(top_faces[0]['lm'], frame) # cx가 작은 쪽이 Left
                    frame_lip_R = self._get_lip_crop(top_faces[1]['lm'], frame) # cx가 큰 쪽이 Right
                elif len(top_faces) == 1:
                    # 얼굴이 1개만 잡혔을 경우, 화면 중앙을 기준으로 L/R 할당
                    if top_faces[0]['cx'] < width / 2:
                        frame_lip_L = self._get_lip_crop(top_faces[0]['lm'], frame)
                    else:
                        frame_lip_R = self._get_lip_crop(top_faces[0]['lm'], frame)
                        
            lips_L.append(frame_lip_L)
            lips_R.append(frame_lip_R)
                
        cap.release()
        print(f"   - 총 {len(lips_L)} 프레임 (L/R 각각) 추출 완료.")
        return np.array(lips_L), np.array(lips_R)

    def extract_text_from_lips(self, lip_frames, speaker_label=""):
        """Auto-AVSR 실제 추론 로직 (에러 수정된 5차원 텐서 버전)"""
        # 프레임이 모두 비어있으면(얼굴이 아예 안 나온 경우) 빈 문자열 반환
        if np.sum(lip_frames) == 0: return ""
            
        T = lip_frames.shape[0]
        lip_crop = lip_frames[:, 4:92, 4:92]
        video_tensor = torch.tensor(lip_crop).float() / 255.0
        video_tensor = (video_tensor - 0.421) / 0.165
        
        # [핵심] ESPnet 규칙: (B, T, C, H, W) -> 내부에서 알아서 조립됨
        video_tensor = video_tensor.contiguous().view(1, T, 1, 88, 88).to(self.device)
        
        with torch.no_grad():
            transcript = self.vsr_model(video_tensor)
            
        if isinstance(transcript, list) or isinstance(transcript, tuple): transcript = transcript[0]
        if isinstance(transcript, dict) and "text" in transcript: transcript = transcript["text"]
            
        print(f"   - VSR 결과 [{speaker_label}]: '{transcript}'")
        return transcript

    def generate_personalized_ssi(self, text, reference_audio_path):
        if not text: return np.zeros(16000) # 텍스트가 없으면 1초짜리 무음 반환
        outputs = self.tts.synthesize(
            text, self.xtts_config, speaker_wav=reference_audio_path, gpt_cond_len=3, language="en"
        )
        return librosa.resample(np.array(outputs["wav"]), orig_sr=24000, target_sr=16000)

    def visualize_tracking_dual(self, raw_video_path, text_L, text_R, output_viz_path):
        """원본 비디오에 L/R 두 명의 얼굴 박스와 텍스트를 각각 오버레이"""
        print(f"6. 듀얼 시각화 비디오 생성 중: {output_viz_path}")
        cap = cv2.VideoCapture(raw_video_path)
        fps, width, height = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_viz_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                faces_info = []
                for lm in results.multi_face_landmarks:
                    x_coords = [l.x * width for l in lm.landmark]
                    y_coords = [l.y * height for l in lm.landmark]
                    area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                    cx = sum(x_coords) / len(x_coords)
                    faces_info.append({'lm': lm.landmark, 'area': area, 'cx': cx, 'x_coords': x_coords, 'y_coords': y_coords})
                
                faces_info.sort(key=lambda x: x['area'], reverse=True)
                top_faces = faces_info[:2]
                top_faces.sort(key=lambda x: x['cx'])
                
                # 박스 그리기
                for i, face in enumerate(top_faces):
                    margin = 15
                    x_min, x_max = max(0, min(face['x_coords'])-margin), min(width, max(face['x_coords'])+margin)
                    y_min, y_max = max(0, min(face['y_coords'])-margin), min(height, max(face['y_coords'])+margin)
                    
                    # 왼쪽 사람은 초록색, 오른쪽 사람은 파란색 박스
                    color = (0, 255, 0) if i == 0 and len(top_faces)==2 or (len(top_faces)==1 and face['cx'] < width/2) else (255, 0, 0)
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

            # 텍스트 시각화 (왼쪽 화자는 화면 좌상단, 오른쪽 화자는 화면 우상단 중앙쯤)
            cv2.putText(frame, f"L: {text_L[:30]}...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"R: {text_R[:30]}...", (width // 2 + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            out.write(frame)
            
        cap.release()
        out.release()
        print(f"   - 🎬 시각화 비디오 저장 완료")

    def mix_and_save(self, y_audio, y_ssi, alpha, output_path):
        """Soft-Mixing 및 파일 저장 헬퍼 함수"""
        min_len = min(len(y_audio), len(y_ssi))
        y_out = (1.0 - alpha) * y_audio[:min_len] + (alpha * y_ssi[:min_len])
        sf.write(output_path, y_out, 16000)
        return y_out

    def run_pipeline(self, raw_video_path, separated_audio_path):
        """L, R 듀얼 파이프라인 병렬 실행"""
        print("========================================")
        print("Dual-Core 음질 복원 시스템 시작 (L/R 분리 처리)")
        print("========================================")
        
        y_audio, _ = librosa.load(separated_audio_path, sr=16000)
        
        # 1. 비디오에서 L, R 입술 동시 분리
        lips_L, lips_R = self.extract_dual_lips_from_video(raw_video_path)
        
        # 2. VSR (입술 -> 텍스트)
        text_L = self.extract_text_from_lips(lips_L, "Left")
        text_R = self.extract_text_from_lips(lips_R, "Right")
        
        with open("final_extracted_text_L.txt", "w", encoding="utf-8") as f: f.write(text_L)
        with open("final_extracted_text_R.txt", "w", encoding="utf-8") as f: f.write(text_R)
        
        # 3. TTS (텍스트 -> 음성)
        y_ssi_L = self.generate_personalized_ssi(text_L, separated_audio_path)
        y_ssi_R = self.generate_personalized_ssi(text_R, separated_audio_path)
        
        # 4. Mixing (알파 0.8) 및 오디오 저장
        self.mix_and_save(y_audio, y_ssi_L, 0.8, "final_seamless_mixed_L.wav")
        self.mix_and_save(y_audio, y_ssi_R, 0.8, "final_seamless_mixed_R.wav")
        
        # 5. L/R 동시 시각화 비디오 생성
        self.visualize_tracking_dual(raw_video_path, text_L, text_R, "final_tracking_viz_Dual.mp4")
        
        print("\n✅ 듀얼 파이프라인 처리 완료! (L, R 분리 파일 생성)")

# ==========================================
if __name__ == "__main__":
    ckpt = "vsr_trlrs2lrs3vox2avsp_base.pth" 
    system = DualSeamlessTalkPoC(vsr_ckpt_path=ckpt)
    
    # [입력 파일들] - 영어 인터뷰 영상으로 테스트하세요!
    raw_video = "sample_english_interview.mp4" 
    separated_audio = "separated_target_audio.wav" 
    
    system.run_pipeline(raw_video, separated_audio)
