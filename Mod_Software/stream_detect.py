"""
Stream-based call detection with subsequence DTW.

Instead of cutting audio into fixed windows and scoring each window, we compute
the spectrogram of a whole hour at once and slide each template along it.
For every spectrogram frame j we get:
    cost[j]  = how well the template matches a stretch of audio that ENDS at frame j
    start[j] = the frame where that best-matching stretch BEGAN
So a dip in cost = a call, and we know where it starts and ends.
"""
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.signal import find_peaks

from background_functions import read_audio_chunk
from testing import match_detections, get_exclusion_intervals

TYPES = ("st", "mt", "bt")
MAX_COST = 3.0          # costs above this are never considered candidates


# ----------------------------------------------------------------------------
# 1. DTW core
# ----------------------------------------------------------------------------
def pairwise_dist(A, B, metric="cosine"):
    """A: (n, n_feat), B: (m, n_feat) -> (n, m). Cosine via matrix multiply (fast)."""
    if metric == "cosine":
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return 1.0 - A @ B.T
    from scipy.spatial import distance as dist
    return dist.cdist(A, B, metric)


@njit(cache=True)
def subseq_dtw_stream(D):
    """
    Subsequence DTW. D: (N template frames, M stream frames) distance matrix.

    The whole template must be matched, but it may START and END at any stream
    frame. Allowed steps (limits warping to between 0.5x and 2x speed):
        (1,1)  diagonal
        (1,2)  template row i matched to two stream columns
        (2,1)  two template rows matched to one stream column
    The two non-diagonal steps also add the cost of the cell they "skip", so
    every template row is always paid for (no cheating by skipping rows).

    Returns:
        cost[j]  : best path cost ending at stream frame j, divided by N
        start[j] : stream frame at which that best path started
    """
    N, M = D.shape
    C = np.full((N, M), np.inf)
    S = np.zeros((N, M), dtype=np.int64)
    for j in range(M):
        C[0, j] = D[0, j]
        S[0, j] = j
    for i in range(1, N):
        for j in range(1, M):
            best = C[i - 1, j - 1] + D[i, j]
            s = S[i - 1, j - 1]
            if j >= 2:
                c = C[i - 1, j - 2] + D[i, j] + D[i, j - 1]
                if c < best:
                    best = c
                    s = S[i - 1, j - 2]
            if i >= 2:
                c = C[i - 2, j - 1] + D[i, j] + D[i - 1, j]
                if c < best:
                    best = c
                    s = S[i - 2, j - 1]
            C[i, j] = best
            S[i, j] = s
    return C[N - 1] / N, S[N - 1]


def stream_costs_for_template(template_feat, stream_feat, metric="cosine", chunk=20000):
    """
    template_feat: (n_feat, N)   stream_feat: (n_feat, M)
    Returns cost (M,) and start (M,) for one template. Works in chunks so the
    distance matrix never gets huge; chunks overlap by 2N so no path is cut off.
    """
    T = np.asarray(template_feat).T            # (N, n_feat)
    S = np.asarray(stream_feat).T              # (M, n_feat)
    N, M = T.shape[0], S.shape[0]
    overlap = 2 * N                            # a path can span at most 2N columns

    cost = np.empty(M, dtype=np.float32)
    start = np.empty(M, dtype=np.int64)
    for a in range(0, M, chunk):
        lo = max(0, a - overlap)
        hi = min(M, a + chunk)
        D = pairwise_dist(T, S[lo:hi], metric)
        c, s = subseq_dtw_stream(D)
        cost[a:hi] = c[a - lo:]
        start[a:hi] = s[a - lo:] + lo
    return cost, start


def stream_scores(templates, stream_feat, metric="cosine"):
    """
    templates: {"st": [feat, ...], "mt": [...], "bt": [...]}
    Returns {type: (costs (n_temps, M), starts (n_temps, M))}
    """
    out = {}
    for t, feats in templates.items():
        res = [stream_costs_for_template(f, stream_feat, metric) for f in feats]
        out[t] = (np.stack([r[0] for r in res]), np.stack([r[1] for r in res]))
    return out


def kth_curve(costs, starts, vote_frac):
    """
    Voting, per frame: the k-th smallest cost across templates, with
    k = ceil(vote_frac * n_templates). Frame is a "hit" if this is below the threshold.
    Also returns the start frame from the best (lowest-cost) template at each frame.
    """
    n_t, M = costs.shape
    k = int(np.ceil(vote_frac * n_t))
    kth = np.partition(costs, k - 1, axis=0)[k - 1]
    best_tmpl = costs.argmin(axis=0)
    start = starts[best_tmpl, np.arange(M)]
    return kth, start


# ----------------------------------------------------------------------------
# 2. Helpers: timing, loading, candidates
# ----------------------------------------------------------------------------
def frame_geometry(extractor, fs_new):
    """(hop in seconds, frame length in seconds) for the STFT extractor."""
    p = extractor.params
    nperseg = int(fs_new * p["frame_dur"])
    hop = nperseg - int(nperseg * p["overlap"])
    return hop / fs_new, nperseg / fs_new


def load_hour_features(audio_file, hour, extractor, fs_new):
    """STFT of one whole hour -> (n_freq, n_frames). None if past the end of the file."""
    try:
        _, audio = read_audio_chunk(audio_file, hour, 1.0, fs_new)
    except Exception:
        return None
    if len(audio) < 4 * int(fs_new * extractor.params["frame_dur"]):
        return None
    return extractor(audio, fs_new)     # extractor works on audio of any length


def find_candidates(score, start_frame, hop_s, frame_s, t0, min_sep_s, max_score=MAX_COST):
    """
    Local minima of a score curve (lower = better match) = candidate calls.
    STFT frames are centred, so a frame at index j covers t0 + j*hop +- frame/2.
    Returns (frame indices, scores, start times, end times).
    """
    score = np.where(np.isfinite(score), score, 10.0)
    idx, _ = find_peaks(-score, height=-max_score,
                        distance=max(1, int(round(min_sep_s / hop_s))))
    start_t = t0 + start_frame[idx] * hop_s - frame_s / 2
    end_t = t0 + idx * hop_s + frame_s / 2
    return idx, score[idx], start_t, end_t


def overlaps_any(starts, ends, intervals):
    """Boolean array: does each [start, end] overlap any (a, b) in intervals?"""
    if len(intervals) == 0:
        return np.zeros(len(starts), dtype=bool)
    iv = np.array(sorted(intervals), dtype=float)
    max_end = np.maximum.accumulate(iv[:, 1])
    idx = np.searchsorted(iv[:, 0], ends, side="left") - 1   # last interval starting before end
    ok = idx >= 0
    out = np.zeros(len(starts), dtype=bool)
    out[ok] = max_end[idx[ok]] > starts[ok]
    return out


# ----------------------------------------------------------------------------
# 3. Calibration from VERIFIED clips (no need for a fully labelled recording)
# ----------------------------------------------------------------------------
def clip_scores(template_feats, clip_feats, vote_fracs, metric="cosine"):
    """
    Score each clip: run the stream detector over the clip (the clip is a short
    'stream') and keep the best (lowest) voted cost anywhere in it.
    Returns {vote_frac: array of one score per clip}.
    """
    out = {vf: [] for vf in vote_fracs}
    for clip in clip_feats:
        res = [stream_costs_for_template(f, clip, metric) for f in template_feats]
        costs = np.stack([r[0] for r in res])
        starts = np.stack([r[1] for r in res])
        for vf in vote_fracs:
            kth, _ = kth_curve(costs, starts, vf)
            kth = np.where(np.isfinite(kth), kth, 10.0)
            out[vf].append(float(kth.min()))
    return {vf: np.array(v) for vf, v in out.items()}


def best_threshold(pos, neg):
    """
    Best-F1 threshold between call-clip scores (pos) and background-clip scores (neg).
    Candidates are midpoints between neighbouring scores; if several give the same
    best F1 (common with ~10 clips) we take the one in the middle of that plateau.
    """
    allv = np.unique(np.concatenate([pos, neg]))
    cands = np.concatenate([[allv[0] - 1e-6], (allv[:-1] + allv[1:]) / 2, [allv[-1] + 1e-6]])
    f1s = []
    for th in cands:
        tp = int((pos <= th).sum())
        fn = len(pos) - tp
        fp = int((neg <= th).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    f1s = np.array(f1s)
    ties = cands[f1s >= f1s.max() - 1e-9]
    th = float(ties[np.argmin(np.abs(ties - np.median(ties)))])
    tp = int((pos <= th).sum())
    fp = int((neg <= th).sum())
    return dict(f1=float(f1s.max()), threshold=th,
                p=tp / (tp + fp) if tp + fp else 0.0,
                r=tp / len(pos),
                pos=pos.tolist(), neg=neg.tolist())


def pair_costs(template_feats, clip_feats, metric="cosine"):
    """
    (n_templates, n_clips) matrix: for every single template and every single clip, the
    best cost of that template anywhere inside the clip (min over end frames).
    This is the stream version of your old 'cost between every template and every clip'.
    """
    out = np.empty((len(template_feats), len(clip_feats)))
    for j, clip in enumerate(clip_feats):
        for i, f in enumerate(template_feats):
            c, _ = stream_costs_for_template(f, clip, metric)
            c = c[np.isfinite(c)]
            out[i, j] = c.min() if len(c) else 10.0
    return out


def calibrate_from_clips(templates, call_clips, bg_clips, vote_fracs, metric="cosine"):
    """
    Same two stages as your original calibration:

    1. THRESHOLD per call type: best-F1 threshold on pair costs
       (every template of that type vs every verified call clip of that type  ->  positives,
        every template of that type vs every verified background clip         ->  negatives).
       With 5 templates and 10 clips that is 50 vs 50 costs.
    2. VOTE FRACTION: with those thresholds fixed, a clip counts as detected if at least
       k = ceil(vote_frac * n_templates) templates match it below the threshold (evaluated
       frame-by-frame, exactly as in deployment). Pick the vote_frac with the best
       worst-case F1 across call types.

    templates / call_clips: {"st": [feat,...], "mt": [...], "bt": [...]}, bg_clips: [feat,...]
    Clips must all be the same length and longer than the longest template.
    """
    # ---- stage 1: thresholds from pair costs
    pair = {}
    for t in TYPES:
        pos = pair_costs(templates[t], call_clips[t], metric).ravel()
        neg = pair_costs(templates[t], bg_clips, metric).ravel()
        pair[t] = best_threshold(pos, neg)

    # ---- stage 2: vote fraction with thresholds fixed
    results = {}
    for t in TYPES:
        pos_s = clip_scores(templates[t], call_clips[t], vote_fracs, metric)
        neg_s = clip_scores(templates[t], bg_clips, vote_fracs, metric)
        th = pair[t]["threshold"]
        for vf in vote_fracs:
            tp = int((pos_s[vf] <= th).sum())
            fn = len(pos_s[vf]) - tp
            fp = int((neg_s[vf] <= th).sum())
            denom = 2 * tp + fp + fn
            results[(vf, t)] = dict(
                f1=2 * tp / denom if denom else 0.0,
                p=tp / (tp + fp) if tp + fp else 0.0,
                r=tp / (tp + fn) if tp + fn else 0.0,
            )

    vote_score = {vf: min(results[(vf, t)]["f1"] for t in TYPES) for vf in vote_fracs}
    best_vf = max(vote_fracs, key=lambda v: (vote_score[v], v))
    per_type = {
        t: dict(threshold=pair[t]["threshold"],
                **results[(best_vf, t)],                      # f1, p, r at the chosen vote_frac (per clip)
                pair_f1=pair[t]["f1"], pair_p=pair[t]["p"], pair_r=pair[t]["r"],
                pos=pair[t]["pos"], neg=pair[t]["neg"])       # the pair costs, for plotting
        for t in TYPES
    }
    return dict(
        vote_frac=best_vf,
        thresholds={t: pair[t]["threshold"] for t in TYPES},
        per_type=per_type,
        score=vote_score[best_vf],
    )


# ----------------------------------------------------------------------------
# 4. Deployment: run over a long recording
# ----------------------------------------------------------------------------
def detect_stream(audio_file, n_hours, templates, thresholds, vote_frac, extractor, fs_new,
                  min_sep_s=0.5, exclude_intervals=(), alpha_max=3.0, metric="cosine"):
    """
    Returns a dict of candidate calls (arrays): start, end, ratio, label.
    ratio = kth_cost / threshold, so ratio <= 1 means "detected at the calibrated
    threshold". Candidates are kept up to ratio alpha_max so you can sweep alpha later.
    """
    hop_s, frame_s = frame_geometry(extractor, fs_new)
    out = {"start": [], "end": [], "ratio": [], "label": []}

    for hour in range(n_hours):
        feat = load_hour_features(audio_file, hour, extractor, fs_new)
        if feat is None:
            print(f"Stopped at hour {hour}: end of file")
            break
        scores = stream_scores(templates, feat, metric)

        ratios, starts = [], []
        for t in TYPES:
            costs, st = scores[t]
            kth, s = kth_curve(costs, st, vote_frac)
            ratios.append(kth / thresholds[t])
            starts.append(s)
        ratios, starts = np.stack(ratios), np.stack(starts)          # (3, M)

        M = ratios.shape[1]
        lab = ratios.argmin(axis=0)                                  # best call type per frame
        best_ratio = ratios[lab, np.arange(M)]
        best_start = starts[lab, np.arange(M)]

        idx, r, s, e = find_candidates(best_ratio, best_start, hop_s, frame_s,
                                       hour * 3600.0, min_sep_s, max_score=alpha_max)
        out["start"].append(s)
        out["end"].append(e)
        out["ratio"].append(r)
        out["label"].append(lab[idx])
        print(f"Hour {hour + 1}/{n_hours}: {int((r <= 1.0).sum())} detections at alpha=1")

    out = {k: np.concatenate(v) for k, v in out.items()}
    keep = ~overlaps_any(out["start"], out["end"], list(exclude_intervals))
    return {k: v[keep] for k, v in out.items()}


def cands_to_calls(cands, alpha=1.0):
    """Candidate dict -> list of (start, end, label) at a given threshold scale alpha."""
    m = cands["ratio"] <= alpha
    return [(float(s), float(e), TYPES[int(lab)])
            for s, e, lab in zip(cands["start"][m], cands["end"][m], cands["label"][m])]


def pr_sweep(cands, ground_truth, alphas):
    """Precision/recall for each alpha (threshold scale factor)."""
    P, R, used = [], [], []
    for a in alphas:
        calls = cands_to_calls(cands, a)
        if not calls:
            continue
        tp, fp, fn, _, _ = match_detections(calls, ground_truth)
        P.append(tp / (tp + fp))
        R.append(tp / (tp + fn) if tp + fn else 0.0)
        used.append(a)
    return np.array(P), np.array(R), np.array(used)


def subset_cands(cands, t_lo, t_hi):
    """Keep candidate calls whose start lies in [t_lo, t_hi) seconds."""
    m = (cands["start"] >= t_lo) & (cands["start"] < t_hi)
    return {k: v[m] for k, v in cands.items()}


def subset_gt(ground_truth, t_lo, t_hi):
    """Keep ground-truth calls whose start lies in [t_lo, t_hi) seconds (order preserved)."""
    return [g for g in ground_truth if t_lo <= g["start"] < t_hi]


def choose_alpha(cands, ground_truth, alphas):
    """
    Best global scale factor on all thresholds (alpha < 1 = stricter), by F1.
    If many alphas are within 2% of the best F1, take the middle one (in log scale)
    so we don't sit on a sharp edge.
    """
    P, R, used = pr_sweep(cands, ground_truth, alphas)
    f1 = 2 * P * R / np.maximum(P + R, 1e-12)
    good = used[f1 >= 0.98 * f1.max()]
    mid = np.median(np.log(good))
    alpha = float(good[np.argmin(np.abs(np.log(good) - mid))])
    return alpha, float(f1.max()), (P, R, used)


# ----------------------------------------------------------------------------
# 5. Config save / load
# ----------------------------------------------------------------------------
def save_stream_config(path, *, feat_method, frame_len, f_min, fs_new, thresholds, vote_frac,
                       n_temps, n_calib, search_len_s, min_sep_s, score, per_type, template_files):
    cfg = dict(feat_method=feat_method, frame_len=float(frame_len), f_min=float(f_min), fs_new=int(fs_new),
               thresholds={k: float(v) for k, v in thresholds.items()},
               vote_frac=float(vote_frac), n_temps=int(n_temps), min_sep_s=float(min_sep_s),
               calibration_worst_type_f1=float(score), per_type=per_type,
               n_calib=int(n_calib), calib_search_len_s=float(search_len_s), template_files=template_files)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved config to {path}")
    return cfg


def load_stream_config(path):
    with open(path) as f:
        return json.load(f)