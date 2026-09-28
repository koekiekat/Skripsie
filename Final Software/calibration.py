import json
import soundfile as sf
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from feature import resample_audio
from dtw import(
    dtw_calc_new
)

def load_entries(path, n=None):
    with open(path) as f:
        return json.load(f)[:n]

def load_template_features_fixed_length(entry, extractor, window_len, fs_new=1000):
    """
    Like load_segment_features, but centers/pads the segment to exactly
    window_len seconds instead of using the call's own start/end duration.
    """
    with sf.SoundFile(entry["wav_path"]) as f:
        f_s = f.samplerate
        call_start, call_end = entry["start_time"], entry["end_time"]
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

    best_idx = np.argmax(f_scores)
    return thresholds[best_idx], f_scores[best_idx]

def plot_pr_curve_from_costs(call_costs, background_costs, n_thresholds=200,
                              best_threshold=None, ax=None):
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

