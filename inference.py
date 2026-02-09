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

def match_tracks_to_streams(P, V, sep_wavs, fps, n_frames, topK=None, min_sim=0.0):
    """
    Robust matching:
      - If T > K: select topK=min(K, T) tracks by activity and match to K streams.
      - If K > T: match all tracks to best unique K subset; extra streams are left unmatched.
      - Apply min_sim threshold to avoid forced bad matches.

    Returns:
      mapping: dict {track_id -> stream_id}  (only confident matches)
      used_tracks: list of track_id that were considered (size <= min(T,K))
      unmatched_streams: list of stream_id not assigned to any track (off-screen candidates)
    """
    K = len(sep_wavs)
    T = P.shape[0]
    if T == 0 or K == 0:
        return {}, [], list(range(K))

    if topK is None:
        topK = min(T, K)
    topK = min(topK, T, K)   # <<< 핵심: 절대 K나 T보다 커지지 않게

    # 1) pick topK tracks by total speaking prob (visibility-weighted)
    activity = (np.nan_to_num(P, nan=0.0) * V.astype(np.float32)).sum(axis=1)
    used_tracks = list(np.argsort(-activity)[:topK])

    # 2) per-stream energy
    E = []
    sr0 = None
    for wp in sep_wavs:
        sr, wav = wavfile.read(wp)
        if sr0 is None: sr0 = sr
        if sr != sr0:
            raise ValueError(f"Sample rate mismatch: {sr0} vs {sr} for {wp}")
        E.append(_rms_energy_frames(wav, sr, fps, n_frames))
    E = np.stack(E, axis=0)  # (K,F)

    # 3) similarity S: (topK, K)
    S = np.full((topK, K), -1.0, dtype=np.float32)
    for i, tid in enumerate(used_tracks):
        m = V[tid]
        if m.sum() < int(0.5 * fps):  # visible < 0.5s → unreliable
            continue
        pt = P[tid, m]
        # normalize
        pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-9)
        for k in range(K):
            ek = E[k, m]
            ek = (ek - ek.min()) / (ek.max() - ek.min() + 1e-9)
            S[i, k] = _safe_corr(pt, ek)

    # 4) Hungarian maximize (works for rectangular)
    r, c = linear_sum_assignment(-S)

    mapping = {}
    assigned_streams = set()
    for rr, cc in zip(r, c):
        sim = float(S[rr, cc])
        if sim >= min_sim:
            tid = used_tracks[rr]
            mapping[int(tid)] = int(cc)
            assigned_streams.add(int(cc))

    unmatched_streams = [i for i in range(K) if i not in assigned_streams]
    return mapping, used_tracks, unmatched_streams





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
    # mapping, used_tracks = match_tracks_to_streams(P, V, sep_wavs, fps=fps, n_frames=n_frames, topK=len(sep_wavs))
    mapping, used_tracks, unmatched = match_tracks_to_streams(
        P, V, sep_wavs, fps=fps, n_frames=n_frames,
        topK=None,         # <<< min(T,K) 자동
        min_sim=0.05       # <<< 억지 매칭 방지(환경 따라 0.0~0.2 조절)
    )
    print("unmatched streams(off-screen candidates):", unmatched)


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
