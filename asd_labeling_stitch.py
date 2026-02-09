import argparse
import itertools
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Utils
# -----------------------------
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns -1 if degenerate or no overlap."""
    if a.size == 0 or b.size == 0:
        return -1.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return -1.0
    aa = a - a.mean()
    bb = b - b.mean()
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-12:
        return -1.0
    return float(np.dot(aa, bb) / denom)


def rms_energy_frames(wav: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """
    Convert waveform -> per-video-frame RMS energy.
    Uses non-overlapping windows aligned to video frames: frame_len = sr / fps.
    """
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    frame_len = int(round(sr / fps))
    if frame_len <= 0:
        raise ValueError("Invalid frame_len; check sr/fps")

    # pad to cover n_frames
    need_len = n_frames * frame_len
    if len(wav) < need_len:
        pad = np.zeros(need_len - len(wav), dtype=np.float32)
        wav = np.concatenate([wav, pad], axis=0)
    else:
        wav = wav[:need_len]

    frames = wav.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return rms


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


# -----------------------------
# LR-ASD outputs -> unified arrays
# -----------------------------
@dataclass
class ASDTracks:
    dets: List[List[dict]]          # faces.pckl
    vid_tracks: List[dict]          # tracks.pckl (list of {'track':..., 'proc_track':...})
    scores: List[np.ndarray]        # scores.pckl (list, per track: score seq)
    fps: float


def load_lrasd_pywork(pywork_dir: str, fps: float = 25.0) -> ASDTracks:
    dets = load_pkl(os.path.join(pywork_dir, "faces.pckl"))
    vid_tracks = load_pkl(os.path.join(pywork_dir, "tracks.pckl"))
    scores = load_pkl(os.path.join(pywork_dir, "scores.pckl"))
    # scores is list; each element could be list -> convert to np.array
    scores = [np.asarray(s, dtype=np.float32) for s in scores]
    return ASDTracks(dets=dets, vid_tracks=vid_tracks, scores=scores, fps=fps)


def build_track_prob_matrix(
    asd: ASDTracks,
    use_sigmoid: bool = True,
    smooth_radius: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      P: (T, F) track speaking probability (or score) aligned to global frame index.
         Non-visible frames are NaN.
      V: (T, F) visibility mask (True where track exists in that frame).
    """
    n_frames = len(asd.dets)  # number of video frames
    n_tracks = len(asd.vid_tracks)

    P = np.full((n_tracks, n_frames), np.nan, dtype=np.float32)
    V = np.zeros((n_tracks, n_frames), dtype=bool)

    for tidx, tr in enumerate(asd.vid_tracks):
        frames = tr["track"]["frame"]  # numpy array of global frame ids
        frames = np.asarray(frames, dtype=int)

        s = asd.scores[tidx]
        # 길이 mismatch 방어: 짧으면 자르고, 길면 트림
        L = min(len(frames), len(s))
        frames = frames[:L]
        s = s[:L]

        if use_sigmoid:
            p = sigmoid(s)
        else:
            p = s.copy()

        # optional temporal smoothing (like visualization uses a local mean window)
        if smooth_radius > 0:
            w = 2 * smooth_radius + 1
            kernel = np.ones(w, dtype=np.float32) / float(w)
            # pad edge
            pad = np.pad(p, (smooth_radius, smooth_radius), mode="edge")
            p = np.convolve(pad, kernel, mode="valid").astype(np.float32)

        P[tidx, frames] = p
        V[tidx, frames] = True

    return P, V


def num_faces_per_frame(asd: ASDTracks) -> np.ndarray:
    return np.asarray([len(x) for x in asd.dets], dtype=int)


def active_speaker_count_per_frame(P: np.ndarray, V: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    Count active speakers per frame using (prob >= thr) among visible tracks.
    """
    # treat NaN as not visible
    active = (P >= thr) & V
    return active.sum(axis=0).astype(int)


# -----------------------------
# Step 3/4: similarity + Hungarian
# -----------------------------
def similarity_matrix_corr(
    P: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    min_overlap_frames: int = 25
) -> np.ndarray:
    """
    P: (T,F) track prob (NaN for not visible)
    V: (T,F) visibility
    E: (K,F) energy per stream
    returns S: (T,K) correlation similarity
    """
    T, F = P.shape
    K, F2 = E.shape
    assert F == F2

    S = np.full((T, K), -1.0, dtype=np.float32)
    for t in range(T):
        mask = V[t]
        if mask.sum() < min_overlap_frames:
            continue
        pt = P[t, mask].astype(np.float32)
        # normalize for stability
        pt = normalize01(pt)
        for k in range(K):
            ek = E[k, mask].astype(np.float32)
            ek = normalize01(ek)
            S[t, k] = safe_corr(pt, ek)
    return S


def hungarian_maximize(sim: np.ndarray, min_sim: float = -0.2) -> Dict[int, int]:
    """
    sim: (T,K). Returns mapping track_id -> stream_id.
    Pads to square and maximizes similarity.
    min_sim: discard assignments below this similarity.
    """
    T, K = sim.shape
    N = max(T, K)
    pad = np.full((N, N), -1e9, dtype=np.float32)
    pad[:T, :K] = sim

    r, c = linear_sum_assignment(-pad)  # maximize
    mapping: Dict[int, int] = {}
    for rr, cc in zip(r, c):
        if rr < T and cc < K:
            if pad[rr, cc] >= min_sim:
                mapping[int(rr)] = int(cc)
    return mapping


# -----------------------------
# Step 5: windowed matching + stitching (swap suppression)
# -----------------------------
def top_tracks_by_activity(P: np.ndarray, V: np.ndarray, top_n: int) -> List[int]:
    # sum of probs on visible frames
    score = np.nan_to_num(P, nan=0.0) * V.astype(np.float32)
    activity = score.sum(axis=1)
    idx = np.argsort(-activity)[:top_n]
    return [int(i) for i in idx]


def permutation_viterbi(
    window_sims: List[np.ndarray],
    track_ids: List[int],
    n_streams: int,
    switch_penalty: float = 0.2
) -> List[Dict[int, int]]:
    """
    window_sims: list of sim matrices for each window, shape (T,K) over ALL tracks.
    track_ids: chosen track indices to consider (size N)
    n_streams: number of streams (K); we assume K == N for permutation Viterbi.
    Returns list of mappings for each window (track->stream).
    """
    N = len(track_ids)
    if N != n_streams:
        raise ValueError("Permutation Viterbi assumes N tracks == K streams. Use fallback otherwise.")

    perms = list(itertools.permutations(range(n_streams)))  # track i -> stream perms[i]
    # score at each window for each permutation
    W = len(window_sims)
    perm_scores = np.full((W, len(perms)), -1e9, dtype=np.float32)

    for w in range(W):
        sim = window_sims[w]
        for pi, perm in enumerate(perms):
            s = 0.0
            ok = True
            for ti, stream_id in enumerate(perm):
                t_global = track_ids[ti]
                val = float(sim[t_global, stream_id])
                if val <= -0.99:  # invalid/no overlap
                    ok = False
                    break
                s += val
            perm_scores[w, pi] = s if ok else -1e9

    # DP
    dp = np.full_like(perm_scores, -1e9)
    back = np.full((W, len(perms)), -1, dtype=int)
    dp[0] = perm_scores[0]

    for w in range(1, W):
        for pi in range(len(perms)):
            best_val = -1e9
            best_prev = -1
            for pj in range(len(perms)):
                trans = 0.0 if perms[pj] == perms[pi] else -switch_penalty
                val = dp[w-1, pj] + trans + perm_scores[w, pi]
                if val > best_val:
                    best_val = val
                    best_prev = pj
            dp[w, pi] = best_val
            back[w, pi] = best_prev

    # backtrack
    last = int(np.argmax(dp[-1]))
    path = [last]
    for w in range(W-1, 0, -1):
        last = back[w, last]
        path.append(int(last))
    path = path[::-1]

    # mapping per window
    mappings: List[Dict[int, int]] = []
    for w, pi in enumerate(path):
        perm = perms[pi]
        m: Dict[int, int] = {}
        for ti, stream_id in enumerate(perm):
            m[track_ids[ti]] = int(stream_id)
        mappings.append(m)
    return mappings


def windowed_match_and_stitch(
    P: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    fps: float,
    win_sec: float = 4.0,
    hop_sec: float = 1.0,
    min_overlap_frames: int = 25,
    min_sim: float = -0.2,
    switch_penalty: float = 0.2
) -> Tuple[List[Dict[int, int]], np.ndarray]:
    """
    Returns:
      window_mappings: list of dict track->stream per window
      frame_track_to_stream: (F,) per frame -> stream id for the "best active track",
                            or -1 if no active track.
    Notes:
      - For general multi-track, multi-stream, we window-match with Hungarian.
      - For small N where (selected topN tracks == n_streams), we do permutation-Viterbi
        to reduce swaps across windows.
    """
    T, F = P.shape
    K, F2 = E.shape
    assert F == F2

    win = int(round(win_sec * fps))
    hop = int(round(hop_sec * fps))
    if win <= 0 or hop <= 0:
        raise ValueError("Invalid win/hop")

    starts = list(range(0, max(1, F - win + 1), hop))
    if starts[-1] != F - win:
        starts.append(max(0, F - win))

    window_sims: List[np.ndarray] = []
    for st in starts:
        ed = min(F, st + win)
        Pw = P[:, st:ed]
        Vw = V[:, st:ed]
        Ew = E[:, st:ed]
        sim = similarity_matrix_corr(Pw, Vw, Ew, min_overlap_frames=max(5, min_overlap_frames // 2))
        window_sims.append(sim)

    # choose top tracks if possible for permutation viterbi (2-speaker case etc.)
    topN = min(T, K)
    track_ids = top_tracks_by_activity(P, V, top_n=topN)

    if topN == K and K <= 4:
        # strong swap suppression by permutation-level Viterbi
        window_mappings = permutation_viterbi(
            window_sims=window_sims,
            track_ids=track_ids,
            n_streams=K,
            switch_penalty=switch_penalty
        )
    else:
        # fallback: per-window Hungarian
        window_mappings = []
        for sim in window_sims:
            window_mappings.append(hungarian_maximize(sim, min_sim=min_sim))

    # Convert window mappings -> per-frame stream label for "active track"
    # 여기서는 간단히: 각 프레임에서 (P가 가장 큰 track) -> 해당 window mapping으로 stream 부여
    frame_track_to_stream = np.full((F,), -1, dtype=int)

    # precompute active track per frame (argmax over visible probs)
    Pn = np.nan_to_num(P, nan=-1e9)
    Pn[~V] = -1e9
    best_track = np.argmax(Pn, axis=0).astype(int)
    best_val = np.max(Pn, axis=0)

    # assign by nearest window (center-based)
    centers = [st + win // 2 for st in starts]
    for f in range(F):
        if best_val[f] < 0:  # nothing visible
            continue
        t = int(best_track[f])
        # nearest window
        w = int(np.argmin([abs(f - c) for c in centers]))
        m = window_mappings[w]
        if t in m:
            frame_track_to_stream[f] = int(m[t])

    return window_mappings, frame_track_to_stream


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pywork", required=True, help=".../pywork (contains faces.pckl tracks.pckl scores.pckl)")
    ap.add_argument("--separated_wavs", nargs="+", required=True, help="Paths to separated wavs (one per stream)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--win_sec", type=float, default=4.0)
    ap.add_argument("--hop_sec", type=float, default=1.0)
    ap.add_argument("--thr_active", type=float, default=0.5, help="threshold for active speaker counting")
    ap.add_argument("--no_sigmoid", action="store_true", help="use raw score instead of sigmoid(score)")
    args = ap.parse_args()

    asd = load_lrasd_pywork(args.pywork, fps=args.fps)

    # Step 1: track probs (and visibility)
    P, V = build_track_prob_matrix(asd, use_sigmoid=(not args.no_sigmoid), smooth_radius=2)

    # (optional) frame-level face count, active speaker count
    faces_cnt = num_faces_per_frame(asd)
    act_cnt = active_speaker_count_per_frame(P, V, thr=args.thr_active)

    print(f"[INFO] frames={len(faces_cnt)}, tracks={P.shape[0]}, streams={len(args.separated_wavs)}")
    print(f"[INFO] avg faces/frame={faces_cnt.mean():.2f}, avg active speakers/frame={act_cnt.mean():.2f}")

    # Step 2: per-stream energy aligned to video frames
    n_frames = P.shape[1]
    energies = []
    sr0 = None
    for wp in args.separated_wavs:
        sr, wav = wavfile.read(wp)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            raise ValueError(f"Sample rate mismatch: {sr0} vs {sr} in {wp}")
        e = rms_energy_frames(wav, sr, fps=args.fps, n_frames=n_frames)
        energies.append(e)
    E = np.stack(energies, axis=0)  # (K,F)

    # Step 3/4/5: windowed matching + stitching
    window_mappings, frame_track_to_stream = windowed_match_and_stitch(
        P=P, V=V, E=E, fps=args.fps, win_sec=args.win_sec, hop_sec=args.hop_sec
    )

    # Print a compact summary (window mappings)
    for i, m in enumerate(window_mappings[:5]):
        print(f"[WIN {i:02d}] track->stream: {m}")

    # You can now label separated streams by track id:
    # Example: choose the dominant mapping per track by majority vote over frames
    T = P.shape[0]
    track_major_stream = {}
    for t in range(T):
        # frames where this track is best and mapped
        idx = np.where((np.argmax(np.nan_to_num(P, nan=-1e9), axis=0) == t) & (frame_track_to_stream >= 0))[0]
        if len(idx) == 0:
            continue
        streams = frame_track_to_stream[idx]
        # majority
        s = int(np.bincount(streams).argmax())
        track_major_stream[t] = s

    print("[RESULT] track_major_stream (majority over frames):", track_major_stream)
    print("[RESULT] frame_track_to_stream saved in memory; integrate this with your separator stitching.")

if __name__ == "__main__":
    main()
