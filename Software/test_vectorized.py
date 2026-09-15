import soundfile as sf
import numpy as np
from stft import resample_audio, stft_calculation
from dtw import batched_dtw_costs_for_template, dtw_calc, batched_dtw_costs_for_template
from background_functions import read_audio_file
import json
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

MIN_STFT_DURATION = 0.128  # seconds -- matches framelength in stft.py's short_time_calc


def read_audio_chunk(audio_file, start_hour, duration_hour = 1.0, fs_new = 1000):
    start_sec = start_hour * 3600
    duration_sec = duration_hour * 3600
    with sf.SoundFile(audio_file) as f:
        f_s = f.samplerate
        f.seek(int(start_sec * f_s))
        audio_array = f.read(int(duration_sec * f_s), dtype="int16")
    resampled = resample_audio(audio_array, f_s, fs_new)

    return fs_new, resampled

def compute_template_dtw_costs(st_stfts, mt_stfts, bt_stfts, n_templates, audio_segment, fs_new=1000):
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

    #will store all DTW costs
    costs_st = []
    costs_mt = []
    costs_bt = []

    #calculate and save all DTW costs
    for i in range(len(st_stfts)):
        costs_st.append(dtw_calc(st_stfts[i], audio_segment))
    for i in range(len(mt_stfts)):
        costs_mt.append(dtw_calc(mt_stfts[i], audio_segment))
    for i in range(len(bt_stfts)):
        costs_bt.append(dtw_calc(bt_stfts[i], audio_segment))

    return costs_st, costs_mt, costs_bt

def dtw_costs_vectorized(st_stfts, mt_stfts, bt_stfts, n_templates, audio_segment):
    # audio_segment here is Zxx_window: (n_windows, n_freq, n_frames_per_window)

    # For each call type, compute costs of every template against every window
    #print("ST")
    st_costs = [batched_dtw_costs_for_template(t, audio_segment) for t in st_stfts]
    #print("MT")
    mt_costs = [batched_dtw_costs_for_template(t, audio_segment) for t in mt_stfts]
    #print("BT")
    bt_costs = [batched_dtw_costs_for_template(t, audio_segment) for t in bt_stfts]

    return st_costs, mt_costs, bt_costs

def load_template_segment(template, fs_new=1000):
    """
    Given one saved template/background entry ({start_time, end_time,
    wav_path, ...}), reload its audio from disk and compute its STFT.

    Pads short segments up to the minimum length required by the STFT
    window before transforming, so every spectrogram has the same number
    of frequency bins regardless of the original segment's duration.
    """
    #Extract template
    f_s, x = read_audio_file(template["wav_path"])
    start_sample = int(template["start_time"] * f_s)
    end_sample = int(template["end_time"] * f_s)
    segment = x[start_sample:end_sample]

    resampled = resample_audio(segment, f_s, fs_new)

    #Ensures window meets minimum length reqs
    min_len = int(fs_new * MIN_STFT_DURATION)
    if len(resampled) < min_len:
        resampled = np.pad(resampled, (0, min_len - len(resampled)), mode="constant")

    #Calculate STFT
    _, _, Zxx = stft_calculation(resampled, f_s, fs_new)
    return Zxx

def batch_stft_windows(audio_array, window_len, step_len, fs_new, framelength, noverlap):
    # (n_windows, window_len) view, no copying
    all_windows = np.lib.stride_tricks.sliding_window_view(audio_array, window_len)[::step_len]
    f, t, Zxx_batch = signal.stft(
        all_windows, fs=fs_new, nperseg=framelength, noverlap=noverlap,
        window="hamming", axis=-1
    )
    return f, t, Zxx_batch  # shape: (n_windows, n_freq_bins, n_time_frames)

def load_threshold(file_label, call_type, results_dir):
    """Load a previously saved threshold for this call type, or None if missing."""
    path = results_dir / f"{file_label}_{call_type}_threshold.json"
    if not path.exists():
        print(f"No saved threshold found for '{call_type}' ({file_label}).")
        return None
    with open(path) as f:
        return json.load(f)


def load_raven_selection_table(path):
    """
    Parse a Raven Pro selection table. Each annotated call appears twice
    (Waveform + Spectrogram views) with identical Begin/End times, so we
    drop the duplicate and keep one row per call.
    """
    df = pd.read_csv(path, sep="\t")
    df = df.drop_duplicates(subset=["Selection", "Begin Time (s)", "End Time (s)", "Call"])

    ground_truth = []
    for _, row in df.iterrows():
        ground_truth.append({
            "start": row["Begin Time (s)"],
            "end": row["End Time (s)"],
            "label": str(row["Call"]).strip().lower(),
        })

    ground_truth.sort(key=lambda c: c["start"])
    return ground_truth

def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end

def match_detections(detections, ground_truth, require_label_match=True):
    detections = sorted(detections, key=lambda d: d[0])
    gt_start = np.array([g["start"] for g in ground_truth])
    gt_end = np.array([g["end"] for g in ground_truth])
    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detections)

    for i, (d_start, d_end, d_label) in enumerate(detections):
        # only consider ground-truth calls that could plausibly overlap d
        lo = np.searchsorted(gt_end, d_start, side="right")
        hi = np.searchsorted(gt_start, d_end, side="left")

        best_j, best_overlap = None, 0.0
        for j in range(lo, hi):
            if gt_matched[j]:
                continue
            if require_label_match and ground_truth[j]["label"] != d_label:
                continue
            overlap = min(d_end, gt_end[j]) - max(d_start, gt_start[j])
            if overlap > best_overlap:
                best_overlap, best_j = overlap, j
        if best_j is not None:
            gt_matched[best_j] = True
            det_matched[i] = True

    tp = sum(det_matched)
    fp = len(detections) - tp
    fn = sum(not m for m in gt_matched)
    return tp, fp, fn

def windows_to_calls(times, scores, labels, threshold, step, window_len, tol=1e-6):
    mask = scores <= threshold
    if not mask.any():
        return []

    idx = np.where(mask)[0]
    t = times[idx]
    lbl = labels[idx]

    # a "break" happens where the gap isn't one step, or the label changes
    gaps = np.diff(t)
    label_changed = lbl[1:] != lbl[:-1]
    breaks = ~np.isclose(gaps, step, atol=tol) | label_changed

    break_positions = np.where(breaks)[0]
    run_starts = np.concatenate(([0], break_positions + 1))
    run_ends = np.concatenate((break_positions, [len(idx) - 1]))

    return [(t[s], t[e] + window_len, lbl[s]) for s, e in zip(run_starts, run_ends)]

def compute_pr_curve(times, scores, labels, ground_truth, step, window_len,
                      n_thresholds=100, require_label_match=True):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    precision, recall = [], []

    for t in thresholds:
        print(t)
        calls = windows_to_calls(times, scores, labels, t, step, window_len)
        tp, fp, fn = match_detections(calls, ground_truth, require_label_match)
        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    return np.array(precision), np.array(recall), thresholds

def compute_average_precision(precision, recall):
    order = np.argsort(recall)
    return np.trapezoid(precision[order], recall[order])

def plot_pr_curve(precision, recall, ap=None, marked_point=None):
    """
    Plot a precision-recall curve.

    precision, recall: arrays from compute_pr_curve, same length, same order
    ap: optional average precision value to show in the legend
    marked_point: optional (precision, recall) tuple to mark on the curve,
        e.g. your combined_calls single-point result at the deployed threshold
    """
    # sort by recall so the line is drawn left-to-right correctly
    order = np.argsort(recall)
    recall_sorted = np.array(recall)[order]
    precision_sorted = np.array(precision)[order]

    plt.figure(figsize=(6, 6))

    label = f"PR curve (AP = {ap:.3f})" if ap is not None else "PR curve"
    plt.plot(recall_sorted, precision_sorted, color="tab:blue", linewidth=2, label=label)

    if marked_point is not None:
        p, r = marked_point
        plt.scatter([r], [p], color="red", zorder=5,
                    label=f"combined_calls point (P={p:.3f}, R={r:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Call Detection")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()