import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer

class WERCalculator:
    def __init__(self, model_id="openai/whisper-small"):
        """Whisper STT 모델 로딩"""
        print(f"STT 모델({model_id}) 로딩 중...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("모델 로딩 완료.")

    def transcribe_audio(self, audio_path):
        """WAV 파일을 읽어 텍스트로 변환"""
        # Whisper는 16kHz 샘플링 레이트를 요구함
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # 입력 특징 추출
        input_features = self.processor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        # 텍스트 디코딩
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()
        
        return transcription

    def calculate_wer(self, audio_path, ground_truth_text):
        """음성 파일과 정답 텍스트를 비교하여 WER 산출"""
        # 1. 모델을 통한 텍스트 추론
        hypothesis_text = self.transcribe_audio(audio_path)
        
        # 2. 대소문자 통일 및 특수문자 제거 (정확한 WER 비교를 위해)
        gt_clean = ground_truth_text.lower().replace(".", "").replace(",", "")
        hyp_clean = hypothesis_text.lower().replace(".", "").replace(",", "")
        
        # 3. Jiwer를 이용한 WER 계산
        error_rate = wer(gt_clean, hyp_clean)
        
        print("\n--- WER 계산 결과 ---")
        print(f"File         : {audio_path}")
        print(f"Ground Truth : {gt_clean}")
        print(f"Hypothesis   : {hyp_clean}")
        print(f"WER          : {error_rate * 100:.2f} %")
        print("---------------------\n")
        
        return error_rate

# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    wer_calc = WERCalculator()
    
    # 평가할 오디오 파일 경로 (예: 원본 분리음, 믹싱된 결과음 등)
    audio_file = "y_out_mixed_audio.wav" 
    
    # 정답 스크립트 (실제 영상/오디오의 정답 대본)
    gt_text = "hello how are you doing today"
    
    # 만약 파일이 없다면 에러가 나므로, 실제 파일 경로로 변경 후 실행하세요.
    try:
        wer_score = wer_calc.calculate_wer(audio_file, gt_text)
    except FileNotFoundError:
        print(f"[안내] {audio_file} 파일을 찾을 수 없습니다. 실제 파일 경로를 지정해주세요.")
