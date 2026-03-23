import os
import glob
import argparse
import yaml
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

# 모델 정의가 포함된 룩투히어(look2hear) 임포트
import look2hear.models

def main():
    parser = argparse.ArgumentParser(description="Batch inference for TIGER TSE model with ASD scores.")
    # --- 폴더 단위 입출력 인자 ---
    parser.add_argument("--audio_dir", required=True, help="Directory containing input mixture audio files (.wav).")
    parser.add_argument("--asd_dir", required=True, help="Directory containing corresponding ASD score files (.npy).")
    parser.add_argument("--output_dir", default="separated_audio", help="Directory to save separated audio files.")
    parser.add_argument("--ckpt_path", required=True, help="Path to your trained model checkpoint (.ckpt or .pth).")
    
    # 모델 설정 (학습하실 때 사용했던 config와 동일하게 맞춰주세요)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--upsampling_depth", type=int, default=5)
    parser.add_argument("--num_sources", type=int, default=2, help="2 for Target & Interference")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- 1. Load Custom Model ---
    print("Loading custom TIGER TSE model...")
    # 우리가 개조한 TIGER 모델 클래스 초기화
    model = look2hear.models.TIGER(
        out_channels=args.out_channels,
        in_channels=args.in_channels,
        num_blocks=args.num_blocks,
        upsampling_depth=args.upsampling_depth,
        num_sources=args.num_sources,
        sample_rate=16000
    )
    
    # 체크포인트 로드 (PyTorch Lightning .ckpt 파일 지원)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    # PyTorch Lightning으로 학습한 경우 'state_dict' 키 안에 가중치가 있음
    # 모델명 앞에 'audio_model.' 이 붙어있을 수 있으므로 제거하는 로직 추가
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('audio_model.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Directory & Parameter Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    target_sr = 16000
    
    audio_files = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    print(f"Found {len(audio_files)} audio files to process.")

    # --- 3. Batch Inference Loop ---
    with torch.no_grad():
        for audio_path in audio_files:
            basename = os.path.basename(audio_path)
            name_no_ext = os.path.splitext(basename)[0]
            
            # 오디오 파일명과 동일한 이름의 .npy 파일 경로 추론
            asd_path = os.path.join(args.asd_dir, f"{name_no_ext}.npy")
            
            if not os.path.exists(asd_path):
                print(f"⚠️ Warning: ASD score missing for {basename}. Skipping...")
                continue
                
            print(f"Processing: {basename}")
            
            # A. Audio Loading & Resampling
            waveform, original_sr = torchaudio.load(audio_path)
            if original_sr != target_sr:
                resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
                waveform = resampler(waveform)
            
            # 모델 입력 규격에 맞게 형태 변환: (B, C, T) -> (1, 1, T)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            audio_input = waveform.unsqueeze(0).to(device)
            
            # B. ASD Score Loading
            try:
                asd_data = np.load(asd_path)
                # 모델 입력 규격에 맞게 텐서 변환: (B, 1, T_video) -> (1, 1, T_video)
                asd_tensor = torch.from_numpy(asd_data).float()
                asd_input = asd_tensor.unsqueeze(0).unsqueeze(0).to(device)
            except Exception as e:
                print(f"❌ Error loading ASD {asd_path}: {e}")
                continue

            # C. Inference (Target Extraction)
            try:
                # 입력 인자 2개 (Audio, Visual Cues) 전달
                ests_speech = model(audio_input, asd_input) 
            except RuntimeError as e:
                print(f"❌ Inference failed for {basename}. Size mismatch? Error: {e}")
                continue
            
            # D. Output Processing & Saving
            # ests_speech shape: [1, num_sources, T_audio] -> [num_sources, T_audio]
            ests_speech = ests_speech.squeeze(0).cpu()
            
            # 채널 0은 Target(ASD가 지목한 화자), 채널 1은 Interference
            target_wav = ests_speech[0].unsqueeze(0)
            interf_wav = ests_speech[1].unsqueeze(0)
            
            # 결과 저장 (이름 뒤에 _target, _interf를 붙여서 저장)
            target_out_path = os.path.join(args.output_dir, f"{name_no_ext}_target.wav")
            interf_out_path = os.path.join(args.output_dir, f"{name_no_ext}_interf.wav")
            
            torchaudio.save(target_out_path, target_wav, target_sr)
            torchaudio.save(interf_out_path, interf_wav, target_sr)

    print(f"🎉 All processing complete! Results are saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
