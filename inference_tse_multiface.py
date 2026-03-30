import os
import glob
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch.nn.functional as F

# 우리가 개조한 TIGER_ASD_MultiFiLM 모델 정의가 포함된 룩투히어 임포트
import look2hear.models

def main():
    parser = argparse.ArgumentParser(description="Multi-face Batch inference for TIGER TSE model.")
    # --- 폴더 단위 입출력 인자 ---
    parser.add_argument("--audio_dir", required=True, help="Directory containing input mixture audio files (.wav).")
    parser.add_argument("--asd_dir", required=True, help="Directory containing corresponding multi-face ASD files (*_track*.npy).")
    parser.add_argument("--output_dir", default="separated_audio_multiface", help="Directory to save separated audio files.")
    parser.add_argument("--ckpt_path", required=True, help="Path to your trained Multi-scale FiLM model checkpoint (.ckpt).")
    
    # 모델 설정 (학습 config와 동일)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_sources", type=int, default=2, help="2 for Target & Interference")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- 1. Load Custom Multi-scale Model ---
    print("Loading TIGER_ASD_MultiFiLM model...")
    # 개조된 TIGER_ASD_MultiFiLM 모델 클래스 초기화
    model = look2hear.models.TIGER_ASD_MultiFiLM(
        out_channels=args.out_channels,
        in_channels=args.in_channels,
        num_blocks=args.num_blocks,
        num_sources=args.num_sources,
        sample_rate=16000
    )
    
    # 체크포인트 로드
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('audio_model.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Directory & Multi-face Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    target_sr = 16000
    video_fps = 25
    
    # [핵심 변경] ASD 결과 파일들을 기준으로 순회 (다중 화자 대응)
    asd_files = glob.glob(os.path.join(args.asd_dir, "*_track*.npy"))
    print(f"Found {len(asd_files)} ASD tracks to process.")

    # --- 3. Inference Loop ---
    with torch.no_grad():
        for asd_path in asd_files:
            asd_basename = os.path.basename(asd_path) # e.g., 'video1_track0.npy'
            name_part = os.path.splitext(asd_basename)[0] # e.g., 'video1_track0'
            
            # 파일명 파싱: '_track'을 기준으로 원본 이름과 Track ID 분리
            parts = name_part.split("_track")
            if len(parts) < 2:
                print(f"⚠️ Warning: Filename format invalid for {asd_basename}. Skipping...")
                continue
            
            audio_basename = parts[0] # e.g., 'video1'
            track_id = parts[1] # e.g., '0'
            
            audio_path = os.path.join(args.audio_dir, f"{audio_basename}.wav")
            
            # 매칭되는 원본 오디오가 없는 경우 스킵
            if not os.path.exists(audio_path):
                print(f"⚠️ Warning: Matching audio missing for {asd_basename}. Expected at {audio_path}. Skipping...")
                continue
                
            print(f"Processing: Audio '{audio_basename}' <-> Track '{track_id}'")
            
            # A. Audio Loading & Resampling
            waveform, original_sr = torchaudio.load(audio_path)
            if original_sr != target_sr:
                resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
                waveform = resampler(waveform)
            
            # (1, 1, T) 형태 변환
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            mixture_input = waveform.unsqueeze(0).to(device)
            
            # B. ASD Score Loading
            asd_data = np.load(asd_path)
            asd_tensor = torch.from_numpy(asd_data).float()
            
            # 🚀 [이전 대화 복습] Audio-Visual 오디오 싱크 미세 교정 로직 (replicate padding)
            mixture_samples = mixture_input.shape[-1]
            # 오디오 길이에 따른 이상적인 비디오 프레임 수 계산
            expected_frames = int((mixture_samples / target_sr) * video_fps)
            current_frames = asd_tensor.shape[-1]
            
            if current_frames < expected_frames:
                # 모자란 경우 마지막 프레임 복사 패딩
                pad_len = expected_frames - current_frames
                # F.pad replicate 모드를 위한 3D 변환 (B, C, T) -> (1, 1, current_frames)
                asd_temp = asd_tensor.view(1, 1, -1)
                asd_padded = F.pad(asd_temp, (0, pad_len), mode='replicate')
                asd_input = asd_padded.to(device)
            else:
                # 같거나 긴 경우 잘라냄
                asd_input = asd_tensor[..., :expected_frames].view(1, 1, expected_frames).to(device)

            # C. Inference (Target Extraction)
            # 입력 인자 2개 (Audio, Visual Cues) 전달
            ests_speech = model(mixture_input, asd_input) 
            
            # D. Output Processing & Saving
            ests_speech = ests_speech.squeeze(0).cpu()
            
            # 채널 0은 Target(시각 정보에 대응), 채널 1은 Interference
            target_wav = ests_speech[0].unsqueeze(0)
            # interf_wav = ests_speech[1].unsqueeze(0) # 필요시 저장
            
            # [핵심 변경] 결과 파일명에 track_id를 명시하여 저장
            target_out_name = f"{audio_basename}_track{track_id}_target.wav"
            target_out_path = os.path.join(args.output_dir, target_out_name)
            
            try:
                torchaudio.save(target_out_path, target_wav, target_sr)
                # print(f"Saved: {target_out_name}")
            except Exception as e:
                print(f"❌ Error saving {target_out_path}: {e}")

    print(f"🎉 Processing complete! Multi-face results are in: {args.output_dir}")

if __name__ == "__main__":
    main()
