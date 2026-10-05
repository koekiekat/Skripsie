import json
import soundfile as sf
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from pathlib import Path

from feature import resample_audio
from dtw import(
    dtw_calc_new,
    dtw_calc,
    dtw_calc_times
)

def load_entries(path, n=None):
    with open(path) as f:
        return json.load(f)[:n]

def load_template_features_fixed_length(template, extractor, window_len, fs_new):
    """
    Like load_segment_features, but centers/pads the segment to exactly
    window_len seconds instead of using the call's own start/end duration.
    """
    with sf.SoundFile(template["wav_path"]) as f:
        f_s = f.samplerate
        call_start, call_end = template["start_time"], template["end_time"]
        call_center = (call_start + call_end) / 2

        half_len = window_len / 2
        seg_start = max(0.0, call_center - half_len)
        seg_end = seg_start + window_len

        start_sample = int(seg_start * f_s)
        end_sample = int(seg_end * f_s)
        f.seek(start_sample)
        segment = f.read(end_sample - start_sample, dtype="int16")

    audio = resample_audio(segment, f_s, fs_new)

    target_len = int(fs_new * window_len)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    elif len(audio) > target_len:
        audio = audio[:target_len]

    return extractor(audio, fs_new)

def dtw_cross_costs(feat_temps, feat_bg_calib, metric="cosine"):
    """
    Calculate costs of every item in feat_temps vs every item in feat_bg_calib.

    Returns:
        costs
    """

    #Build all feature pairs (feat_temps, feat_bg_calib)
    pairs = list(product(range(len(feat_temps)), range(len(feat_bg_calib)))) 

    #Calculate costs for each pair
    costs = [dtw_calc(feat_temps[i], feat_bg_calib[j], metric) for i, j in pairs]

    return costs

def dtw_cross_costs_new(feat_temps, feat_bg_calib, metric="cosine"):
    """
    Calculate costs of every item in feat_temps vs every item in feat_bg_calib.

    Returns:
        costs
    """

    #Build all feature pairs (feat_temps, feat_bg_calib)
    pairs = list(product(range(len(feat_temps)), range(len(feat_bg_calib)))) 

    #Calculate costs for each pair
    costs = [dtw_calc_new(feat_temps[i], feat_bg_calib[j], metric) for i, j in pairs]

    return costs

def compute_pr_curve_from_costs(call_costs, background_costs, n_thresholds=200):
    """
    Compute precision/recall at each threshold, sweeping over the cost range.
    Mirrors compute_roc_curve's structure but returns precision/recall instead of fpr/tpr.
    """
    call_costs = np.array(call_costs)
    background_costs = np.array(background_costs)

    lo = min(call_costs.min(), background_costs.min())
    hi = max(call_costs.max(), background_costs.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    precision, recall = [], []
    for t in thresholds:
        tp = np.sum(call_costs <= t)
        fn = np.sum(call_costs > t)
        fp = np.sum(background_costs <= t)

        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    return np.array(precision), np.array(recall), thresholds

def find_threshold_best_f1_from_pr(precision, recall, thresholds, beta=1.0):
    """Given precision/recall/thresholds (from compute_pr_curve_from_costs), find the best F-beta point."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    f_scores = np.nan_to_num(f_scores, nan=0.0)


    ties = np.flatnonzero(f_scores >= f_scores.max() - 1e-9)
    best_idx = ties[np.argmin(np.abs(thresholds[ties] - np.median(thresholds[ties])))]
    #best_idx = np.argmax(f_scores)
    return thresholds[best_idx], f_scores[best_idx]

def plot_pr_curve_from_costs(call_costs, background_costs, n_thresholds=200, best_threshold=None, ax=None):
    precision, recall, thresholds = compute_pr_curve_from_costs(call_costs, background_costs, n_thresholds)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(recall, precision, color="tab:blue", linewidth=2, label="PR curve")

    if best_threshold is not None:
        idx = np.argmin(np.abs(thresholds - best_threshold))
        ax.scatter(recall[idx], precision[idx], color="red", zorder=5,
                   label=f"Chosen threshold = {best_threshold:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve: DTW-based Call Detection")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    if standalone:
        plt.show()

    return precision, recall, thresholds

def plot_dtw_scatter(call_costs, background_costs, threshold=None, seed=0, ax=None):
    rng = np.random.default_rng(seed)
    call_costs = np.array(call_costs)
    background_costs = np.array(background_costs)

    call_y = 1 + rng.uniform(-0.15, 0.15, size=len(call_costs))
    background_y = 0 + rng.uniform(-0.15, 0.15, size=len(background_costs))

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(call_costs, call_y, color="tab:blue", label="Call vs Call", alpha=0.8)
    ax.scatter(background_costs, background_y, color="tab:orange", label="Call vs Background", alpha=0.8)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.3f}")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Background", "Call"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("DTW Cost")
    ax.set_title("Individual DTW Costs: Call vs Call vs Background")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)

    if standalone:
        plt.show()

def save_model_config(path, *, feat_method, extractor, fs_new, best_frame_len, thresholds, best_vote_frac, n_temps,
                      n_calib, best_f1=None, vote_f1_scores=None, template_files=None, search_len_s):
    config = {
        "feature": {
            "method": feat_method,
            "frame_len": float(best_frame_len),
            "params": extractor.params,          # frame_dur, overlap, window
            "metric": extractor.metric,
            "fs_new": int(fs_new),
        },
        "search_len_s": float(search_len_s),
        "thresholds": {k: float(v) for k, v in thresholds.items()},   # {"st":..,"mt":..,"bt":..}
        "best_vote_frac": float(best_vote_frac),
        "n_temps": int(n_temps),
        "n_calib": int(n_calib),
        "best_f1_calibration": None if best_f1 is None else float(best_f1),
        # tuple keys aren't valid JSON, so flatten them to strings
        "vote_f1_scores": None if vote_f1_scores is None else
            {"|".join(map(str, k)): float(v) for k, v in vote_f1_scores.items()},
        "template_files": template_files,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config to {path}")
    return config

def frame_hop_len(extractor, fs_new):
    #frame_dur, overlap, window
    feature_parameters = extractor.params
    samples_per_frame =int(fs_new *feature_parameters["frame_dur"])
    hop_in_samples= samples_per_frame - int(samples_per_frame * feature_parameters["overlap"])
    hop_in_seconds = hop_in_samples / fs_new
    return hop_in_seconds

def dtw_cross_costs_time(feat_temps, feat_comps, hop_sec, frame_sec, metric):
    pair_res = list(product(range(len(feat_temps)), range(len(feat_comps))))
    results = [dtw_calc_times(feat_temps[i], feat_comps[j], hop_sec, frame_sec, metric) for i, j in pair_res]
    costs, start_s, end_s = zip(*results)
    return list(costs), list(start_s), list(end_s)
