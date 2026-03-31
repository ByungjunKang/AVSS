import os
import glob
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch.nn.functional as F

# 개조된 TIGER_ASD_MultiFiLM 모델 클래스가 있는 모듈 임포트
import look2hear.models

def main():
    parser = argparse.ArgumentParser(description="Top-2 Multi-face Batch inference for TIGER TSE model.")
    parser.add_argument("--audio_dir", required=True, help="Directory containing input mixture audio files (.wav).")
    parser.add_argument("--asd_dir", required=True, help="Directory containing multi-face ASD files (*_track*.npy).")
    parser.add_argument("--output_dir", default="separated_tse_top2", help="Directory to save separated audio files.")
    parser.add_argument("--ckpt_path", required=True, help="Path to trained Multi-scale FiLM model checkpoint (.ckpt).")
    
    # 모델 파라미터
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_sources", type=int, default=2)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- 1. Load Custom Multi-scale Model ---
    print("Loading TIGER_ASD_MultiFiLM model...")
    model = look2hear.models.TIGER_ASD_MultiFiLM(
        out_channels=args.out_channels,
        in_channels=args.in_channels,
        num_blocks=args.num_blocks,
        num_sources=args.num_sources,
        sample_rate=16000
    )
    
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('audio_model.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")

    os.makedirs(args.output_dir, exist_ok=True)
    target_sr = 16000
    video_fps = 25

    # --- 2. Iterate over Audio files instead of ASD files ---
    audio_files = glob.glob(os.path.join(args.audio_dir, "*.wav"))
    print(f"🔍 Found {len(audio_files)} audio files to process.")

    with torch.no_grad():
        for audio_path in audio_files:
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            
            # 해당 오디오의 ASD 파일들 검색
            asd_files = glob.glob(os.path.join(args.asd_dir, f"{audio_basename}_track*.npy"))
            if not asd_files:
                print(f"⚠️ Warning: No ASD files found for {audio_basename}. Skipping...")
                continue
                
            # 🚀 [핵심] 길이 기준 내림차순 정렬 후 Top-2 추출
            asd_lengths = [(f, np.load(f).shape[-1]) for f in asd_files]
            asd_lengths.sort(key=lambda x: x[1], reverse=True)
            
            N_speakers = min(2, len(asd_lengths))
            top_asd_files = [x[0] for x in asd_lengths[:N_speakers]]
            
            # 선택된 트랙 ID 확인용
            track_ids = [os.path.splitext(os.path.basename(f))[0].split("_track")[-1] for f in top_asd_files]
            print(f"\nProcessing {audio_basename}: Selected Top-{N_speakers} tracks -> IDs: {track_ids}")
            
            # A. Audio Loading & Resampling (오디오는 한 번만 로드)
            waveform, original_sr = torchaudio.load(audio_path)
            if original_sr != target_sr:
                resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
                waveform = resampler(waveform)
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            mixture_input = waveform.unsqueeze(0).to(device) # (1, 1, T)
            mixture_samples = mixture_input.shape[-1]
            expected_frames = int((mixture_samples / target_sr) * video_fps)

            # B. 선택된 Top-2 화자에 대해 각각 추론 수행
            for asd_path, track_id in zip(top_asd_files, track_ids):
                # ASD Score Loading
                asd_data = np.load(asd_path)
                asd_tensor = torch.from_numpy(asd_data).float()
                current_frames = asd_tensor.shape[-1]
                
                # Audio-Visual 오디오 싱크 미세 교정 로직 (replicate padding)
                if current_frames < expected_frames:
                    pad_len = expected_frames - current_frames
                    asd_temp = asd_tensor.view(1, 1, -1)
                    asd_padded = F.pad(asd_temp, (0, pad_len), mode='replicate')
                    asd_input = asd_padded.to(device)
                else:
                    asd_input = asd_tensor[..., :expected_frames].view(1, 1, expected_frames).to(device)

                # C. 모델 추론 (FiLM 레이어로 시각 정보 주입)
                ests_speech = model(mixture_input, asd_input) 
                
                # D. Output Processing & Saving
                ests_speech = ests_speech.squeeze(0).cpu()
                
                # 채널 0은 Target(시각 정보에 대응)
                target_wav = ests_speech[0].unsqueeze(0)
                
                target_out_name = f"{audio_basename}_track{track_id}_target.wav"
                target_out_path = os.path.join(args.output_dir, target_out_name)
                
                try:
                    torchaudio.save(target_out_path, target_wav, target_sr)
                    print(f"  └─ Saved: {target_out_name}")
                except Exception as e:
                    print(f"  └─ ❌ Error saving {target_out_path}: {e}")

    print(f"\n🎉 Processing complete! Results are in: {args.output_dir}")

if __name__ == "__main__":
    main()
