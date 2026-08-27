from itertools import combinations, product
import json
import matplotlib.pyplot as plt
import numpy as np

from background_functions import read_audio_file
from stft import resample_audio, stft_calculation
from dtw import dtw_calc  # adjust this import to match wherever dtw_calc actually lives

MIN_STFT_DURATION = 0.128  # seconds -- must match the framelength used in short_time_calc

def _load_template_segment(template, fs_new=1000):
    """
    Given one saved template/background entry ({start_time, end_time,
    wav_path, ...}), reload its audio from disk and compute its STFT.

    Pads short segments up to the minimum length required by the STFT
    window before transforming, so every spectrogram has the same number
    of frequency bins regardless of the original segment's duration.
    """
    f_s, x = read_audio_file(template["wav_path"])
    start_sample = int(template["start_time"] * f_s)
    end_sample = int(template["end_time"] * f_s)
    segment = x[start_sample:end_sample]

    resampled = resample_audio(segment, f_s, fs_new)

    min_len = int(fs_new * MIN_STFT_DURATION)
    if len(resampled) < min_len:
        resampled = np.pad(resampled, (0, min_len - len(resampled)), mode="constant")

    _, _, Zxx = stft_calculation(resampled, f_s, fs_new)
    return Zxx

def compute_template_dtw_costs(templates_path, n_templates, fs_new=1000):
    """
    Load a saved templates JSON file, take the first `n_templates` entries,
    and compute the DTW cost of every template against every other template
    (all unique pairs, no self-comparisons, no duplicate reverse pairs since
    dtw_calc is symmetric).

    Returns:
        costs: 1D list of DTW costs, one per pair
        pairs: list of (i, j) index pairs (positions within the first
               n_templates entries) matching each entry in `costs`
    """
    with open(templates_path) as f:
        templates = json.load(f)

    if len(templates) < n_templates:
        print(f"Warning: file only has {len(templates)} template(s), "
              f"using all of them instead of {n_templates}.")
    templates = templates[:n_templates]

    spectrograms = [_load_template_segment(t, fs_new) for t in templates]

    costs = []
    pairs = []
    for i, j in combinations(range(len(templates)), 2):
        cost = dtw_calc(spectrograms[i], spectrograms[j])
        costs.append(cost)
        pairs.append((i, j))

    return costs, pairs

def compute_template_background_dtw_costs(templates_path, background_path,
                                            n_templates, n_background, fs_new=1000):
    """
    Load a saved templates JSON file and a saved background JSON file, take
    the first `n_templates` and first `n_background` entries respectively,
    and compute the DTW cost of every template against every background
    segment (the full cross product -- there's no symmetry to exploit here
    since templates and background are two different sets).

    Returns:
        costs: 1D list of DTW costs, one per (template, background) pair
        pairs: list of (i, j) index pairs (i = position in the first
               n_templates entries, j = position in the first n_background
               entries) matching each entry in `costs`
    """
    with open(templates_path) as f:
        templates = json.load(f)
    with open(background_path) as f:
        background = json.load(f)

    if len(templates) < n_templates:
        print(f"Warning: templates file only has {len(templates)} entr(ies), "
              f"using all of them instead of {n_templates}.")
    if len(background) < n_background:
        print(f"Warning: background file only has {len(background)} entr(ies), "
              f"using all of them instead of {n_background}.")

    templates = templates[:n_templates]
    background = background[:n_background]

    template_specs = [_load_template_segment(t, fs_new) for t in templates]
    background_specs = [_load_template_segment(b, fs_new) for b in background]

    costs = []
    pairs = []
    for i, j in product(range(len(templates)), range(len(background))):
        cost = dtw_calc(template_specs[i], background_specs[j])
        costs.append(cost)
        pairs.append((i, j))

    return costs, pairs

def compute_roc_curve(call_costs, background_costs, n_thresholds=200):
    """
    Compute ROC curve points by sweeping thresholds over the full range of
    both cost distributions.

    call_costs: DTW costs from call-vs-call comparisons (positive class)
    background_costs: DTW costs from call-vs-background comparisons (negative class)

    A sample is classified as "call" when its DTW cost <= threshold.

    Returns:
        fpr: array of false positive rates, one per threshold
        tpr: array of true positive rates, one per threshold
        thresholds: the threshold values used, same order as fpr/tpr
    """
    call_costs = np.array(call_costs)
    background_costs = np.array(background_costs)

    lo = min(call_costs.min(), background_costs.min())
    hi = max(call_costs.max(), background_costs.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    tpr = []
    fpr = []
    for t in thresholds:
        tp = np.sum(call_costs <= t)
        fn = np.sum(call_costs > t)
        fp = np.sum(background_costs <= t)
        tn = np.sum(background_costs > t)

        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    return np.array(fpr), np.array(tpr), thresholds

def compute_auc(fpr, tpr):
    """Area under the ROC curve via the trapezoidal rule."""
    # Sort by fpr ascending so np.trapz integrates correctly
    order = np.argsort(fpr)
    return np.trapz(tpr[order], fpr[order])

def plot_roc_curve(call_costs, background_costs, n_thresholds=200,
                    best_threshold=None):
    """
    Plot the ROC curve for your DTW-based call/background classifier,
    optionally marking the point corresponding to a chosen threshold
    (e.g. from find_threshold_youden).
    """
    fpr, tpr, thresholds = compute_roc_curve(call_costs, background_costs, n_thresholds)
    auc = compute_auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance level")

    if best_threshold is not None:
        idx = np.argmin(np.abs(thresholds - best_threshold))
        plt.scatter(fpr[idx], tpr[idx], color="red", zorder=5,
                    label=f"Chosen threshold = {best_threshold:.3f}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: DTW-based Call Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return fpr, tpr, thresholds, auc

def find_threshold_youden_from_roc(fpr, tpr, thresholds):
    """
    Given ROC curve arrays (from compute_roc_curve), find the threshold
    that maximizes Youden's J statistic (J = TPR - FPR) -- i.e. the point
    on the curve furthest above the diagonal chance line.
    """
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]

def plot_dtw_scatter(call_costs, background_costs, threshold=None, seed=0):
    """
    Plot every individual DTW cost as a point, split into two rows
    (call-vs-call and call-vs-background), with optional jitter so
    overlapping points are still visible. Draws a vertical line at
    `threshold` if given, so you can see exactly where it falls
    relative to the actual data points.
    """
    rng = np.random.default_rng(seed)
    call_costs = np.array(call_costs)
    background_costs = np.array(background_costs)

    # Small vertical jitter purely for visual separation of overlapping points
    call_y = 1 + rng.uniform(-0.15, 0.15, size=len(call_costs))
    background_y = 0 + rng.uniform(-0.15, 0.15, size=len(background_costs))

    plt.figure(figsize=(10, 4))
    plt.scatter(call_costs, call_y, color="tab:blue", label="Call vs Call", alpha=0.8)
    plt.scatter(background_costs, background_y, color="tab:orange",
                label="Call vs Background", alpha=0.8)

    if threshold is not None:
        plt.axvline(threshold, color="black", linestyle="--", linewidth=2,
                    label=f"Threshold = {threshold:.3f}")

    plt.yticks([0, 1], ["Background", "Call"])
    plt.ylim(-0.5, 1.5)
    plt.xlabel("DTW Cost")
    plt.title("Individual DTW Costs: Call vs Call vs Background")
    plt.legend(loc="upper right")
    plt.grid(axis="x", alpha=0.3)
    plt.show()

def _threshold_path(file_label, call_type, results_dir):
    return results_dir / f"{file_label}_{call_type}_threshold.json"

def save_threshold(threshold, j_stat, auc, call_type, file_label, results_dir,
                    n_templates=None, n_background=None):
    """
    Save a computed DTW threshold for a given call type, along with the
    metrics used to select it, so it can be reloaded later for detection
    without needing to recompute the DTW distributions.
    """
    data = {
        "call_type": call_type,
        "file_label": file_label,
        "threshold": float(threshold),
        "youden_j": float(j_stat),
        "auc": float(auc),
        "n_templates": n_templates,
        "n_background": n_background,
    }
    path = _threshold_path(file_label, call_type, results_dir)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved threshold for '{call_type}' ({file_label}) to {path.name}: "
          f"threshold={threshold:.4f}, J={j_stat:.4f}, AUC={auc:.4f}")
    return data

def load_threshold(file_label, call_type, results_dir):
    """Load a previously saved threshold for this call type, or None if missing."""
    path = _threshold_path(file_label, call_type, results_dir)
    if not path.exists():
        print(f"No saved threshold found for '{call_type}' ({file_label}).")
        return None
    with open(path) as f:
        return json.load(f)