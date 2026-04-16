import torch
import librosa
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

class KoreanSTTEvaluator:
    def __init__(self, model_id="openai/whisper-small"):
        """Whisper STT 모델 로딩"""
        print(f"STT 모델({model_id}) 로딩 중...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("모델 로딩 완료.")

    def clean_korean_text(self, text):
        """한글 평가를 위한 텍스트 정제 (특수문자 제거 및 다중 공백 압축)"""
        # 한글, 영문, 숫자, 공백만 남기고 모두 제거
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
        # 연속된 공백을 하나의 공백으로 압축
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def transcribe_audio(self, audio_path):
        """WAV 파일을 읽어 한글 텍스트로 변환"""
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        input_features = self.processor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        # 💡 [핵심] 한국어(Korean)로 강제 변환 및 Transcribe Task 지정
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ko", task="transcribe")
        
        predicted_ids = self.model.generate(
            input_features, 
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=255 # 생성할 최대 토큰 수
        )
        
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()
        
        return transcription

    def evaluate(self, audio_path, ground_truth_text):
        """음성 파일과 정답 텍스트를 비교하여 WER 및 CER 산출"""
        # 1. 오디오 추론
        hypothesis_text = self.transcribe_audio(audio_path)
        
        # 2. 텍스트 정제 (특수문자 제거 등)
        gt_clean = self.clean_korean_text(ground_truth_text)
        hyp_clean = self.clean_korean_text(hypothesis_text)
        
        # 3. Jiwer를 이용한 WER(단어 오류율) 및 CER(글자 오류율) 계산
        error_wer = wer(gt_clean, hyp_clean)
        error_cer = cer(gt_clean, hyp_clean)
        
        print("\n=== STT 평가 결과 ===")
        print(f"File         : {audio_path}")
        print(f"Ground Truth : {gt_clean}")
        print(f"Hypothesis   : {hyp_clean}")
        print("-" * 21)
        print(f"WER (단어 오류율) : {error_wer * 100:.2f} %")
        print(f"CER (글자 오류율) : {error_cer * 100:.2f} %  <-- 한글 평가 핵심 지표")
        print("=====================\n")
        
        return error_wer, error_cer

# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    evaluator = KoreanSTTEvaluator()
    
    audio_file = "y_out_mixed_audio.wav" 
    
    # 정답 스크립트 (한글)
    gt_text = "안녕하세요 오늘 날씨가 참 좋네요. 회의를 시작하겠습니다."
    
    try:
        wer_score, cer_score = evaluator.evaluate(audio_file, gt_text)
    except FileNotFoundError:
        print(f"[안내] {audio_file} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
