import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from dtw import (
    dtw_calc_new,
)


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

def compute_roc_curve(call_costs, background_costs, n_thresholds=200):#automatically evaluates 200 thresholds
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
    #saves all costs in arrays as it was previously lists(this is NB for vector comparison)
    call_costs = np.array(call_costs)
    background_costs = np.array(background_costs)

    #finds min and max DTW cost in each array to determine the range of evaluation of thresholds
    lo = min(call_costs.min(), background_costs.min())
    hi = max(call_costs.max(), background_costs.max())

    #Generates n_thresholds evenly spaced decision threshold values between lo and hi.
    thresholds = np.linspace(lo, hi, n_thresholds)

    tpr = [] #true positive rate
    fpr = [] #false positive rate

    #works through each threshold value generated
    for t in thresholds:
        tp = np.sum(call_costs <= t) #counts how many calls are correctly classified
        fn = np.sum(call_costs > t) #counts incorrect call classifications
        fp = np.sum(background_costs <= t) #counts incorrect background classifications
        tn = np.sum(background_costs > t) #counts correct background classifications

        #Calculates rates for the current threshold t
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0) #sensitivity / recall [TP/(TP + FN)]
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0) #[FP/(FP + TN)]

    return np.array(fpr), np.array(tpr), thresholds

def compute_auc(fpr, tpr):
    """Area under the ROC curve via the trapezoidal rule."""
    # Sort by fpr ascending so np.trapz integrates correctly(left to right integration)
    order = np.argsort(fpr)

    #calculate area under ROC curve
    return np.trapezoid(tpr[order], fpr[order])

def plot_roc_curve(call_costs, background_costs, n_thresholds=200, best_threshold=None, ax=None):
    fpr, tpr, thresholds = compute_roc_curve(call_costs, background_costs, n_thresholds)
    auc = compute_auc(fpr, tpr)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance level")

    if best_threshold is not None:
        idx = np.argmin(np.abs(thresholds - best_threshold))
        ax.scatter(fpr[idx], tpr[idx], color="red", zorder=5, label=f"Chosen threshold = {best_threshold:.3f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: DTW-based Call Detection")
    ax.legend()
    ax.grid(alpha=0.3)

    if standalone:
        plt.show()

    return fpr, tpr, thresholds, auc

def find_threshold_youden_from_roc(fpr, tpr, thresholds):
    """
    Given ROC curve arrays (from compute_roc_curve), find the threshold
    that maximizes Youden's J statistic (J = TPR - FPR) -- i.e. the point
    on the curve furthest above the diagonal chance line.
    """
    #calculate Youden's J statistic(used to find best threshold)
    j_scores = tpr - fpr

    #saves maximum j_score
    best_idx = np.argmax(j_scores)
    
    return thresholds[best_idx], j_scores[best_idx]

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

def threshold_path_new(file_label, call_type, results_dir, feature_name="stft"):
    return results_dir / f"{file_label}_{feature_name}_{call_type}_threshold.json"

def save_threshold_new(threshold, j_stat, auc, call_type, file_label, results_dir,
                   n_templates=None, n_background=None,
                   feature_name="stft", feature_params=None):
    data = {
        "call_type": call_type, "file_label": file_label,
        "feature": feature_name, "feature_params": feature_params,
        "threshold": float(threshold), "youden_j": float(j_stat), "auc": float(auc),
        "n_templates": n_templates, "n_background": n_background,
    }
    path = threshold_path_new(file_label, call_type, results_dir, feature_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved threshold for '{call_type}' [{feature_name}] to {path.name}: "
          f"threshold={threshold:.4f}, J={j_stat:.4f}, AUC={auc:.4f}")
    return data

def load_threshold_new(file_label, call_type, results_dir, feature_name="stft"):
    path = threshold_path_new(file_label, call_type, results_dir, feature_name)
    if not path.exists():
        print(f"No saved threshold found for '{call_type}' [{feature_name}] ({file_label}).")
        return None
    with open(path) as f:
        return json.load(f)
