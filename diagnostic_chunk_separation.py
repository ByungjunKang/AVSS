import os
import glob
import argparse
import torch
import torchaudio
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description="Save raw separated chunks for diagnostic listening.")
    parser.add_argument("--audio_dir", required=True, help="Directory containing input mixture audio files.")
    parser.add_argument("--output_dir", default="diagnostic_chunks", help="Directory to save raw chunk wav files.")
    parser.add_argument("--ckpt_path", required=True, help="Path to trained Audio-only model checkpoint.")
    parser.add_argument("--chunk_sec", type=float, default=3.0, help="Chunk length in seconds (matching training config).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- 1. Load Audio-only Backbone Model ---
    # (주의: 학습 시 사용했던 정확한 클래스명과 파라미터로 세팅해야 합니다)
    import look2hear.models
    model = look2hear.models.TIGER(out_channels=128, in_channels=256, num_blocks=8, num_sources=2) 
    
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    new_state_dict = {k.replace('audio_model.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")

    target_sr = 16000
    chunk_samples = int(args.chunk_sec * target_sr)

    audio_files = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    
    with torch.no_grad():
        for audio_path in audio_files:
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            
            # 각 비디오/오디오 파일별로 폴더 생성
            save_dir = os.path.join(args.output_dir, basename)
            os.makedirs(save_dir, exist_ok=True)
            
            wav, orig_sr = torchaudio.load(audio_path)
            if orig_sr != target_sr:
                wav = torchaudio.transforms.Resample(orig_sr, target_sr)(wav)
            wav = wav[0] # (T,)
            
            total_samples = wav.shape[0]
            num_chunks = total_samples // chunk_samples
            if total_samples % chunk_samples != 0:
                num_chunks += 1
                
            print(f"\n🎧 Processing {basename}: Slicing into {num_chunks} chunks...")

            for i in range(num_chunks):
                start = i * chunk_samples
                end = start + chunk_samples
                
                mix_chunk = wav[start:end]
                
                # 마지막 청크가 3초보다 짧을 경우 0으로 패딩 (모델 연산 오류 방지)
                if mix_chunk.shape[0] < chunk_samples:
                    pad_len = chunk_samples - mix_chunk.shape[0]
                    mix_chunk = F.pad(mix_chunk, (0, pad_len))
                
                # 1. 인풋 믹스처 저장 (비교용)
                mix_out_path = os.path.join(save_dir, f"chunk_{i:03d}_mix.wav")
                torchaudio.save(mix_out_path, mix_chunk.unsqueeze(0), target_sr)
                
                # 2. 모델 추론
                input_tensor = mix_chunk.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 48000)
                ests = model(input_tensor).squeeze(0).cpu() # (2, 48000)
                
                # 3. 분리된 각 채널(ch0, ch1) 저장
                for ch in range(ests.shape[0]):
                    ch_out_path = os.path.join(save_dir, f"chunk_{i:03d}_ch{ch}.wav")
                    torchaudio.save(ch_out_path, ests[ch].unsqueeze(0), target_sr)
                
            print(f"  └─ Saved {num_chunks} chunks to {save_dir}/")

if __name__ == "__main__":
    main()
