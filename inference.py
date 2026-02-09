import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment

def _sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def _rms_energy_frames(wav, sr, fps, n_frames):
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    frame_len = int(round(sr / fps))
    need_len = n_frames * frame_len
    if len(wav) < need_len:
        wav = np.pad(wav, (0, need_len - len(wav)))
    else:
        wav = wav[:need_len]
    frames = wav.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return rms

def _safe_corr(a, b):
    if a.size < 5 or b.size < 5:
        return -1.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return -1.0
    return float(np.dot(a, b) / denom)

def build_track_prob_and_bbox(vidTracks, scores, n_frames, use_sigmoid=True):
    """
    Returns:
      P: (T,F) speaking prob (NaN where not visible)
      B: (T,F,4) bbox (x1,y1,x2,y2), undefined where not visible
      V: (T,F) visibility mask
    """
    T = len(vidTracks)
    P = np.full((T, n_frames), np.nan, dtype=np.float32)
    V = np.zeros((T, n_frames), dtype=bool)
    B = np.zeros((T, n_frames, 4), dtype=np.float32)

    for tid in range(T):
        frames = np.asarray(vidTracks[tid]['track']['frame'], dtype=int)
        bboxes  = np.asarray(vidTracks[tid]['track']['bbox'], dtype=np.float32)
        s = np.asarray(scores[tid], dtype=np.float32)

        L = min(len(frames), len(bboxes), len(s))
        frames = frames[:L]
        bboxes = bboxes[:L]
        s = s[:L]
        p = _sigmoid(s) if use_sigmoid else s

        P[tid, frames] = p
        V[tid, frames] = True
        B[tid, frames, :] = bboxes

    return P, B, V

def match_tracks_to_streams(P, V, sep_wavs, fps, n_frames, topK=None):
    """
    Compute corr(track_prob, stream_energy) and Hungarian assignment.
    Returns:
      mapping: dict {track_id -> stream_id}
      used_tracks: list of track_id used (topK by activity if topK given)
    """
    K = len(sep_wavs)
    T = P.shape[0]
    if topK is None:
        topK = min(T, K)

    # pick topK tracks by total speaking prob
    activity = np.nan_to_num(P, nan=0.0) * V.astype(np.float32)
    activity = activity.sum(axis=1)
    used_tracks = list(np.argsort(-activity)[:topK])

    # energies
    E = []
    sr0 = None
    for wp in sep_wavs:
        sr, wav = wavfile.read(wp)
        if sr0 is None: sr0 = sr
        if sr != sr0:
            raise ValueError(f"Sample rate mismatch: {sr0} vs {sr} for {wp}")
        E.append(_rms_energy_frames(wav, sr, fps, n_frames))
    E = np.stack(E, axis=0)  # (K,F)

    # similarity (topK x K)
    S = np.full((topK, K), -1.0, dtype=np.float32)
    for i, tid in enumerate(used_tracks):
        m = V[tid]
        if m.sum() < int(0.5 * fps):  # <0.5s visible
            continue
        pt = P[tid, m]
        pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-9)
        for k in range(K):
            ek = E[k, m]
            ek = (ek - ek.min()) / (ek.max() - ek.min() + 1e-9)
            S[i, k] = _safe_corr(pt, ek)

    # Hungarian maximize
    r, c = linear_sum_assignment(-S)
    mapping = {}
    for rr, cc in zip(r, c):
        tid = used_tracks[rr]
        mapping[int(tid)] = int(cc)

    return mapping, used_tracks




def export_per_speaker_videos(args, vidTracks, scores, sep_wavs, thr=0.5, fps=25):
    """
    Create per-track visualization video and mux matched separated wav.
    Output:
      {savePath}/pyavi_speakers/trackXXXXX_video_only.avi
      {savePath}/pyavi_speakers/trackXXXXX_out.mp4
    """
    # count frames from extracted frames dir
    frame_files = sorted(glob.glob(os.path.join(args.pyframesPath, "*.jpg")))
    n_frames = len(frame_files)
    if n_frames == 0:
        raise RuntimeError(f"No frames found in {args.pyframesPath}")

    P, B, V = build_track_prob_and_bbox(vidTracks, scores, n_frames, use_sigmoid=True)

    # 1) match tracks to separated streams
    mapping, used_tracks = match_tracks_to_streams(P, V, sep_wavs, fps=fps, n_frames=n_frames, topK=len(sep_wavs))

    out_dir = os.path.join(args.savePath, "pyavi_speakers")
    os.makedirs(out_dir, exist_ok=True)

    # get video size from first frame
    img0 = cv2.imread(frame_files[0])
    H, W = img0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    for tid, sid in mapping.items():
        video_only = os.path.join(out_dir, f"track{tid:05d}_video_only.avi")
        video_out  = os.path.join(out_dir, f"track{tid:05d}_out.mp4")
        wav_path   = sep_wavs[sid]

        vw = cv2.VideoWriter(video_only, fourcc, fps, (W, H))
        if not vw.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {video_only}")

        for fidx, fp in enumerate(frame_files):
            img = cv2.imread(fp)

            if V[tid, fidx]:
                p = float(P[tid, fidx])
                x1, y1, x2, y2 = B[tid, fidx].astype(int).tolist()

                # color by active / inactive
                color = (0, 255, 0) if p >= thr else (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"T{tid} S{sid} p={p:.2f}", (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            vw.write(img)

        vw.release()

        # 2) mux separated wav into video (overwrite audio)
        # -shortest: 맞는 길이까지만
        # -map: 비디오/오디오 정확히 매핑
        cmd = (
            f'ffmpeg -y -i "{video_only}" -i "{wav_path}" '
            f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "{video_out}" -loglevel error'
        )
        subprocess.call(cmd, shell=True, stdout=None)

    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Per-speaker videos saved in {out_dir}\r\n")




else:
    if args.per_speaker_video and args.separated_wavs.strip():
        sep_wavs = [x.strip() for x in args.separated_wavs.split(",") if x.strip()]
        export_per_speaker_videos(args, vidTracks, scores, sep_wavs, thr=args.as_thr, fps=25)
    else:
        visualization(vidTracks, scores, args)
