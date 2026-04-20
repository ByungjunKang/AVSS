import os
import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
import librosa
import soundfile as sf

# ==========================================
# [환경 세팅 주의] 
# 1. Auto-AVSR 코드 최상단 경로에서 실행
# 2. CosyVoice가 설치된 경우 sys.path.append() 로 경로 추가 필요
# ==========================================
try:
    from lightning import ModelModule # Auto-AVSR
except ImportError:
    print("[오류] auto_avsr 폴더 최상단에서 실행해주세요.")

try:
    # CosyVoice 로컬 라이브러리 임포트 (설치된 환경 가정)
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    print("[안내] CosyVoice 패키지가 없습니다. 더미(Dummy) TTS 모드로 실행합니다.")
    CosyVoice = None


class VpuSyncedLrs3PoC:
    def __init__(self, vsr_ckpt_path, tts_model_dir="./pretrained_models/CosyVoice-300M-SFT"):
        print("1. 오프라인 모델 초기화 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Auto-AVSR 로드 (Offline)
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
        
        # 2. 텍스트 추출용 얼굴 메쉬 로드
        self.mp_face_mesh_single = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        
        # 3. TTS 모델 로드 (완전 오프라인)
        print(f"   - Zero-shot TTS (CosyVoice) 로컬 모델 로드: {tts_model_dir}")
        if CosyVoice is not None:
            # 외부 통신 없이 로컬 폴더에서 가중치 직행 로드
            self.tts_model = CosyVoice(tts_model_dir)
        else:
            self.tts_model = None
        print("초기화 완료.\n")

    def extract_lips_from_face_crop(self, video_path):
        """LRS3 전용: 1인 크롭 영상에서 입술만 빠르게 추출"""
        print(f"2. 비디오 전처리 중: {video_path}")
        cap = cv2.VideoCapture(video_path)
        cropped_lips = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh_single.process(rgb_frame)
            
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = results.multi_face_landmarks[0].landmark
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
                cropped_lips.append(cv2.cvtColor(lip_crop_resized, cv2.COLOR_BGR2GRAY))
            else:
                cropped_lips.append(np.zeros((96, 96), dtype=np.uint8))
                
        cap.release()
        return np.array(cropped_lips)

    def extract_text_from_lips(self, lip_frames):
        """VSR 디코딩 (Auto-AVSR)"""
        if np.sum(lip_frames) == 0: return ""
        T = lip_frames.shape[0]
        lip_crop = lip_frames[:, 4:92, 4:92]
        video_tensor = torch.tensor(lip_crop).float() / 255.0
        video_tensor = (video_tensor - 0.421) / 0.165
        video_tensor = video_tensor.contiguous().view(1, T, 1, 88, 88).to(self.device)
        
        with torch.no_grad():
            transcript = self.vsr_model(video_tensor)
            
        if isinstance(transcript, list) or isinstance(transcript, tuple): transcript = transcript[0]
        if isinstance(transcript, dict) and "text" in transcript: transcript = transcript["text"]
        print(f"   - VSR 결과: '{transcript}'")
        return transcript

    def get_vpu_proxy_anchor(self, clean_audio_path):
        """[특허 검증 1] LRS3 정답 오디오에서 VPU 앵커 (T_vpu) 추출"""
        y, sr = librosa.load(clean_audio_path, sr=16000)
        # VPU 가속도계가 감지하는 성대 진동과 유사하게 에너지 기반 발화 구간 추출
        intervals = librosa.effects.split(y, top_db=25) 
        if len(intervals) > 0:
            t_start = intervals[0][0] / sr
            t_end = intervals[-1][1] / sr
            t_vpu = t_end - t_start
        else:
            t_start, t_end = 0, len(y) / sr
            t_vpu = t_end
            
        print(f"3. VPU 프록시 앵커 획득: T_start={t_start:.2f}s, T_end={t_end:.2f}s, T_vpu(발화길이)={t_vpu:.2f}s")
        return t_start, t_end, t_vpu

    def generate_synced_ssi(self, text, ref_audio_path, t_vpu_target):
        """[특허 검증 2] Duration Predictor 기반 감마(Gamma) 스케일링 합성"""
        print(f"4. Duration Control 합성 진행 (Zero-shot)")
        
        if self.tts_model is None:
            # TTS 미설치 시 1초짜리 빈 오디오 반환 (디버깅용)
            return np.zeros(16000)

        # 1. 모델이 예측한 기본 발화 길이(T_pred)를 알아내기 위한 베이스라인 추론
        # (원래는 모델 내부 텐서 길이를 직접 가져와야 하지만, PoC 레벨에서는 기본 합성본의 길이로 T_pred를 정의합니다.)
        prompt_speech_16k = load_wav(ref_audio_path, 16000)
        baseline_output = self.tts_model.inference_zero_shot(text, "영어", prompt_speech_16k, prompt_text="")
        
        # 기본 합성된 파형의 길이를 T_pred로 삼음
        y_baseline = baseline_output['tts_speech'].numpy().flatten()
        t_pred = len(y_baseline) / 22050.0 # CosyVoice 기본 sample rate
        
        # ==========================================
        # ★ [특허 수식 적용] 감마(Gamma) 산출 및 Duration 강제 조절
        # ==========================================
        gamma = t_vpu_target / t_pred
        
        # 모델의 speed 파라미터는 보통 감마의 역수로 작용 (속도가 빠르면 길이가 짧아짐)
        speed_factor = 1.0 / gamma 
        
        print(f"   - T_pred(모델기본): {t_pred:.2f}s | T_vpu(목표): {t_vpu_target:.2f}s")
        print(f"   - γ(Gamma): {gamma:.2f} | 적용 Speed Factor: {speed_factor:.2f}")

        # 스케일링(Speed)이 적용된 최종 합성
        synced_output = self.tts_model.inference_zero_shot(
            text, "영어", prompt_speech_16k, prompt_text="", speed=speed_factor
        )
        y_ssi_22k = synced_output['tts_speech'].numpy().flatten()
        
        # 파이프라인 동기화를 위해 16kHz로 리샘플링
        y_ssi_16k = librosa.resample(y_ssi_22k, orig_sr=22050, target_sr=16000)
        return y_ssi_16k

    def mix_and_save_lrs3(self, y_audio, y_ssi, t_start, output_path, sr=16000):
        """최종 동적 믹싱 (VPU T_start 시점에 정확히 오디오를 위치시킴)"""
        print(f"5. 동적 믹싱(Soft-Mixing) 및 저장")
        target_length = len(y_audio)
        
        # 1. 빈 캔버스(배경음) 준비
        y_out = y_audio.copy() * 0.2 # 분리된 원본 소음/음성은 볼륨을 줄임 (Alpha 조절 효과)
        
        # 2. VPU가 찍어준 T_start 시점부터 스케일링된 y_ssi 덮어쓰기
        start_idx = int(t_start * sr)
        end_idx = start_idx + len(y_ssi)
        
        if end_idx <= target_length:
            y_out[start_idx:end_idx] += y_ssi * 0.8
        else:
            # y_ssi가 약간 넘치면 잘라냄
            available_len = target_length - start_idx
            y_out[start_idx:] += y_ssi[:available_len] * 0.8
            
        sf.write(output_path, y_out, sr)
        print(f"✅ 결과 저장 완료: {output_path}")

    def run_pipeline(self, lrs3_video_path, lrs3_clean_audio_path, separated_audio_path):
        print("========================================")
        print("LRS3 벤치마크 기반 VPU-TTS 동기화 검증 PoC")
        print("========================================")
        
        # 1. 텍스트 추출
        lip_frames = self.extract_lips_from_face_crop(lrs3_video_path)
        text = self.extract_text_from_lips(lip_frames)
        
        # 2. VPU Proxy 앵커 획득
        t_start, t_end, t_vpu = self.get_vpu_proxy_anchor(lrs3_clean_audio_path)
        
        # 3. 음색 복제 및 Duration 제어
        y_ssi = self.generate_synced_ssi(text, separated_audio_path, t_vpu)
        
        # 4. 믹싱
        y_audio, _ = librosa.load(separated_audio_path, sr=16000)
        self.mix_and_save_lrs3(y_audio, y_ssi, t_start, "final_lrs3_vpu_synced.wav")

# ==========================================
if __name__ == "__main__":
    ckpt = "vsr_trlrs2lrs3vox2avsp_base.pth" 
    # CosyVoice 체크포인트가 있는 로컬 폴더 경로를 지정하세요
    tts_ckpt = "./pretrained_models/CosyVoice-300M-SFT" 
    
    system = VpuSyncedLrs3PoC(vsr_ckpt_path=ckpt, tts_model_dir=tts_ckpt)
    
    # [입력 파일들] (LRS3-2Mix 데이터셋 기준)
    video = "sample_lrs3_face.mp4" 
    clean_audio = "sample_lrs3_clean_target.wav" # 정답 오디오 (VPU 앵커용)
    sep_audio = "sample_lrs3_separated_noisy.wav" # 분리된 노이즈 섞인 타겟 오디오 (음색 및 믹싱용)
    
    system.run_pipeline(video, clean_audio, sep_audio)
