import torchaudio
import torchaudio.transforms as T

# --- TIGER separation ---
parser.add_argument('--run_tiger', action='store_true', help='Run TIGER speech separation inside Columbia_test.py')
parser.add_argument('--tiger_model_path', type=str, default="", help='Local path to TIGER model folder (downloaded HF snapshot)')
parser.add_argument('--tiger_num_sources', type=int, default=2, help='Max number of separated sources to use (top-N)')
parser.add_argument('--tiger_outdir', type=str, default="", help='Where to save sep wavs (default: {savePath}/separated_audio)')
parser.add_argument('--tiger_device', type=str, default="cuda", help='cuda|cpu')


def run_tiger_separation(audio_path: str, model_path: str, out_dir: str, device: str = "cuda", max_sources: int = 2):
    """
    Run TIGER speech separation on `audio_path` and save sep wavs into `out_dir`.
    Returns: list of wav paths [sep0.wav, sep1.wav, ...]
    """
    if not model_path:
        raise ValueError("--tiger_model_path is required when --run_tiger is set")

    os.makedirs(out_dir, exist_ok=True)

    # device
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # Load model (local folder path)
    # TIGER inference script uses look2hear.models.TIGER.from_pretrained(...) :contentReference[oaicite:4]{index=4}
    import look2hear.models
    model = look2hear.models.TIGER.from_pretrained(model_path)  # local path
    model.to(dev)
    model.eval()

    # Load audio
    waveform, sr = torchaudio.load(audio_path)  # [C, T]
    target_sr = 16000  # TIGER script targets 16kHz :contentReference[oaicite:5]{index=5}
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    # Prepare input: TIGER script builds [B, C, T] :contentReference[oaicite:6]{index=6}
    audio_input = waveform.unsqueeze(0).to(dev)  # [1, 1, T]

    with torch.no_grad():
        est = model(audio_input)

    # Robust shape handling (expected: [B, N, T] per script :contentReference[oaicite:7]{index=7})
    if est.dim() == 3:
        est = est.squeeze(0)  # [N, T] or [C, T]? assume [N, T]
    else:
        raise RuntimeError(f"Unexpected TIGER output shape: {tuple(est.shape)}")

    num_src = est.shape[0]
    use_n = min(num_src, max_sources)

    sep_paths = []
    for i in range(use_n):
        out_wav = os.path.join(out_dir, f"sep{i}.wav")
        track = est[i].detach().cpu().float().unsqueeze(0)  # torchaudio.save expects [C, T]
        torchaudio.save(out_wav, track, sr)
        sep_paths.append(out_wav)

    return sep_paths


def run_tiger_separation(audio_path: str, model_path: str, out_dir: str, device: str = "cuda", max_sources: int = 2):
    """
    Run TIGER speech separation on `audio_path` and save sep wavs into `out_dir`.
    Returns: list of wav paths [sep0.wav, sep1.wav, ...]
    """
    if not model_path:
        raise ValueError("--tiger_model_path is required when --run_tiger is set")

    os.makedirs(out_dir, exist_ok=True)

    # device
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # Load model (local folder path)
    # TIGER inference script uses look2hear.models.TIGER.from_pretrained(...) :contentReference[oaicite:4]{index=4}
    import look2hear.models
    model = look2hear.models.TIGER.from_pretrained(model_path)  # local path
    model.to(dev)
    model.eval()

    # Load audio
    waveform, sr = torchaudio.load(audio_path)  # [C, T]
    target_sr = 16000  # TIGER script targets 16kHz :contentReference[oaicite:5]{index=5}
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    # Prepare input: TIGER script builds [B, C, T] :contentReference[oaicite:6]{index=6}
    audio_input = waveform.unsqueeze(0).to(dev)  # [1, 1, T]

    with torch.no_grad():
        est = model(audio_input)

    # Robust shape handling (expected: [B, N, T] per script :contentReference[oaicite:7]{index=7})
    if est.dim() == 3:
        est = est.squeeze(0)  # [N, T] or [C, T]? assume [N, T]
    else:
        raise RuntimeError(f"Unexpected TIGER output shape: {tuple(est.shape)}")

    num_src = est.shape[0]
    use_n = min(num_src, max_sources)

    sep_paths = []
    for i in range(use_n):
        out_wav = os.path.join(out_dir, f"sep{i}.wav")
        track = est[i].detach().cpu().float().unsqueeze(0)  # torchaudio.save expects [C, T]
        torchaudio.save(out_wav, track, sr)
        sep_paths.append(out_wav)

    return sep_paths
