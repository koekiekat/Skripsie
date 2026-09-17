import soundfile as sf
import numpy as np
from stft import resample_audio, stft_calculation
from dtw import batched_dtw_costs_for_template, dtw_calc, batched_dtw_costs_for_template
from background_functions import read_audio_file
import json
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import random

MIN_STFT_DURATION = 0.128  # seconds -- matches framelength in stft.py's short_time_calc

def merge_consecutive_detections(all_detected_t, all_detected_labels, step=0.075, window_len=0.15, tol=1e-6):
    """
    Merge consecutive per-window detections into contiguous calls.

    A run continues as long as consecutive detections are exactly one `step`
    apart (within `tol`) and share the same label; otherwise a new run starts.

    Returns:
        combined_calls: list of (start_time, end_time, label) tuples
    """
    combined_calls = []

    if len(all_detected_t) > 0:
        run_start = all_detected_t[0]
        run_end = all_detected_t[0]
        run_label = all_detected_labels[0]

        for t, label in zip(all_detected_t[1:], all_detected_labels[1:]):
            gap = t - run_end
            is_consecutive = np.isclose(gap, step, atol=tol)

            if is_consecutive and label == run_label:
                # extend the current run
                run_end = t
            else:
                # close out the current run, start a new one
                combined_calls.append((run_start, run_end + window_len, run_label))
                run_start = t
                run_end = t
                run_label = label

        # don't forget to flush the last run after the loop ends
        combined_calls.append((run_start, run_end + window_len, run_label))
    return combined_calls

def run_detection_pipeline(audio_file, n_hours, st_stfts, mt_stfts, bt_stfts,
                            st_threshold, mt_threshold, bt_threshold,
                            exclude_intervals, fs_new=1000,
                            window_len=0.15, step_len=0.075, vote_frac=0.6):
    """
    Run the DTW-based call detector over an audio file in 1-hour chunks,
    excluding windows that overlap `exclude_intervals` (e.g. tonal downsweeps).

    Returns:
        all_detected_t, all_detected_labels: times/labels of triggered detections
        all_scores, all_labels_all, all_times_all: per-window best cost/label/time
            for every evaluated window (used for PR curves)
    """
    all_detected_t = []
    all_detected_labels = []
    all_scores = []
    all_labels_all = []
    all_times_all = []

    n_templates = len(st_stfts)  # assumes st/mt/bt all use the same n_templates

    for i in range(n_hours):
        f_s, audio_array = read_audio_chunk(audio_file, i, duration_hour=1.0, fs_new=fs_new)

        f_win, t_win, Zxx_window = batch_stft_windows(
            audio_array,
            window_len=int(window_len * fs_new),
            step_len=int(window_len * fs_new / 2),
            fs_new=f_s,
            framelength=int(f_s * 0.128),
            noverlap=int(int(f_s * 0.128) * 0.75),
        )

        n_windows = Zxx_window.shape[0]
        window_times_global = np.arange(n_windows) * step_len + i * 3600

        keep = get_exclusion_mask(window_times_global, window_len=window_len, exclude_intervals=exclude_intervals)
        Zxx_window = Zxx_window[keep]
        window_times_global = window_times_global[keep]

        costs_st, costs_mt, costs_bt = dtw_costs_vectorized(
            st_stfts, mt_stfts, bt_stfts, n_templates=n_templates, audio_segment=Zxx_window
        )

        results = {}
        for costs_list, threshold, label in [(costs_st, st_threshold, "st"),
                                              (costs_mt, mt_threshold, "mt"),
                                              (costs_bt, bt_threshold, "bt")]:
            costs_arr = np.vstack(costs_list)
            n_t = costs_arr.shape[0]
            votes = (costs_arr < threshold).sum(axis=0)
            mean_cost = costs_arr.mean(axis=0)
            triggered = votes >= np.ceil(vote_frac * n_t)
            results[label] = (triggered, mean_cost)

        n_windows = Zxx_window.shape[0]
        call_detected = np.zeros(n_windows, dtype=bool)
        call_labels = np.full(n_windows, "", dtype=object)
        best_cost = np.full(n_windows, np.inf)
        best_cost_all = np.full(n_windows, np.inf)
        best_label_all = np.full(n_windows, "", dtype=object)

        for label, (triggered, mean_cost) in results.items():
            all_time_better = mean_cost < best_cost_all
            best_label_all[all_time_better] = label
            best_cost_all[all_time_better] = mean_cost[all_time_better]

            triggered_better = triggered & (mean_cost < best_cost)
            call_labels[triggered_better] = label
            best_cost[triggered_better] = mean_cost[triggered_better]
            call_detected |= triggered

        print(f"Processing hour {i+1} of {n_hours}: {call_detected.sum()} / {n_windows} windows flagged as calls")

        all_scores.extend(best_cost_all.tolist())
        all_labels_all.extend(best_label_all.tolist())
        all_times_all.extend(window_times_global.tolist())  # <-- fixed: reuse filtered times, don't recompute

        detected = np.where(call_detected)[0]
        detected_t_global = window_times_global[detected]
        all_detected_t.extend(detected_t_global.tolist())
        all_detected_labels.extend(call_labels[detected].tolist())

    all_detected_t = np.array(all_detected_t)
    all_detected_labels = np.array(all_detected_labels, dtype=object)
    all_scores = np.array(all_scores)
    all_labels_all = np.array(all_labels_all, dtype=object)
    all_times_all = np.array(all_times_all)

    return all_detected_t, all_detected_labels, all_scores, all_labels_all, all_times_all

def load_all_thresholds(results_dir, file_label):
    """
    Load previously saved thresholds for single_tone, multi_tone, and burst_tonal
    call types.

    Returns:
        st_threshold, mt_threshold, bt_threshold: float threshold values
    """
    call_types = ["single_tone", "multi_tone", "burst_tonal"]
    thresholds_by_type = {}

    for call_type in call_types:
        result = load_threshold(file_label=file_label, call_type=call_type, results_dir=results_dir)
        thresholds_by_type[call_type] = result["threshold"]

    return thresholds_by_type["single_tone"], thresholds_by_type["multi_tone"], thresholds_by_type["burst_tonal"]

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

def load_template_segment_fast(template, fs_new=1000):
    with sf.SoundFile(template["wav_path"]) as f:
        f_s = f.samplerate
        start_sample = int(template["start_time"] * f_s)
        end_sample = int(template["end_time"] * f_s)
        f.seek(start_sample)
        segment = f.read(end_sample - start_sample, dtype="int16")

    resampled = resample_audio(segment, f_s, fs_new)

    min_len = int(fs_new * MIN_STFT_DURATION)
    if len(resampled) < min_len:
        resampled = np.pad(resampled, (0, min_len - len(resampled)), mode="constant")

    _, _, Zxx = stft_calculation(resampled, f_s, fs_new)
    return Zxx

def load_all_template_stfts(results_dir, file_label, recording_label, n_templates=10, fs_new=1000):
    """
    Load single_tone, multi_tone, and burst_tonal template JSON files for a given
    file_label/recording_label, take the first n_templates from each, and compute
    their STFTs.

    Returns:
        st_stfts, mt_stfts, bt_stfts: lists of STFT arrays (Zxx), one per template
    """
    call_types = ["single_tone", "multi_tone", "burst_tonal"]
    stfts_by_type = {}

    for call_type in call_types:
        path = results_dir / f"{file_label}_{recording_label}_{call_type}_templates.json"
        with open(path) as f:
            templates = json.load(f)

        templates = templates[:n_templates]
        stfts_by_type[call_type] = [load_template_segment_fast(t, fs_new) for t in templates]

    return stfts_by_type["single_tone"], stfts_by_type["multi_tone"], stfts_by_type["burst_tonal"]

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
    """
    Same matching logic as match_detections, but also returns the actual
    false positive detections (not just the count) so they can be inspected.
    """
    detections = sorted(detections, key=lambda d: d[0])
    gt_start = np.array([g["start"] for g in ground_truth])
    gt_end = np.array([g["end"] for g in ground_truth])
    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detections)

    for i, (d_start, d_end, d_label) in enumerate(detections):
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

    false_positive_detections = [d for d, matched in zip(detections, det_matched) if not matched]
    false_negative_gts = [g for g, matched in zip(ground_truth, gt_matched) if not matched]

    return tp, fp, fn, false_positive_detections, false_negative_gts

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
        calls = windows_to_calls(times, scores, labels, t, step, window_len)
        tp, fp, fn, _, _ = match_detections(calls, ground_truth, require_label_match)
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

def plot_detection_spectrograms(detections, wav_path,
                                 fs_new=1000,
                                 framelength=None,
                                 noverlap=None,
                                 context=0.5,
                                 n_examples=12,
                                 hour_duration=3600.0,
                                 random_sample=True,
                                 seed=0,
                                 title_prefix="FP"):
    """
    Plot spectrograms of a sample of detections (e.g. false positives or
    false negatives), with time context on either side.

    detections: list of (start_time, end_time, label) in GLOBAL seconds
                (for false negatives, use (g['start'], g['end'], g['label']))
    wav_path: path to the full audio .wav file
    framelength / noverlap: STFT params in samples; defaults to the same
                             0.128s / 75% overlap used in your detection pipeline
    context: seconds of padding before/after the call to include
    n_examples: max number to plot
    random_sample: if True, randomly sample n_examples; if False, take the first n
    title_prefix: label prefix for subplot titles, e.g. "FP" or "FN"
    """
    if framelength is None:
        framelength = int(fs_new * 0.128)
    if noverlap is None:
        noverlap = int(framelength * 0.75)

    if len(detections) == 0:
        print(f"No {title_prefix} entries to plot.")
        return

    if random_sample:
        rng = random.Random(seed)
        sample = rng.sample(detections, min(n_examples, len(detections)))
    else:
        sample = detections[:n_examples]

    # group by hour so we only load each hour's audio once
    from collections import defaultdict
    by_hour = defaultdict(list)
    for start, end, label in sample:
        hour_idx = int(start // hour_duration)
        by_hour[hour_idx].append((start, end, label))

    n_plot = len(sample)
    n_cols = 3
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    ax_idx = 0
    for hour_idx, entries in by_hour.items():
        f_s, audio_array = read_audio_chunk(wav_path, hour_idx, duration_hour=1.0, fs_new=fs_new)

        for start, end, label in entries:
            local_start = start - hour_idx * hour_duration
            local_end = end - hour_idx * hour_duration

            seg_start_t = max(0, local_start - context)
            seg_end_t = min(len(audio_array) / f_s, local_end + context)
            seg_start_idx = int(seg_start_t * f_s)
            seg_end_idx = int(seg_end_t * f_s)
            segment = audio_array[seg_start_idx:seg_end_idx]

            ax = axes[ax_idx]
            if len(segment) < framelength:
                ax.set_title(f"{title_prefix}: {label} @ {start:.2f}s\n(segment too short)")
                ax.axis("off")
                ax_idx += 1
                continue

            f, t, Zxx = signal.stft(segment, fs=f_s, nperseg=framelength, noverlap=noverlap,
                                     window="hamming")
            Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

            ax.pcolormesh(t + seg_start_t, f, Zxx_db, shading="gouraud", cmap="viridis")
            ax.axvline(local_start, color="red", linestyle="--", linewidth=1)
            ax.axvline(local_end, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{title_prefix}: {label} @ {start:.2f}s (dur={end-start:.2f}s)", fontsize=10)
            ax.set_xlabel("Time (s, local)")
            ax.set_ylabel("Freq (Hz)")
            ax_idx += 1

    for j in range(ax_idx, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def get_exclusion_intervals(ground_truth, exclude_labels=("td",), pad=0.0):
    """Returns a list of (start, end) tuples for calls you want excluded, with optional padding (s)."""
    return [(g["start"] - pad, g["end"] + pad)
            for g in ground_truth if g["label"] in exclude_labels]

def get_exclusion_mask(window_times_global, window_len, exclude_intervals):
    """
    window_times_global: 1D array of each window's start time (global seconds)
    Returns a boolean array, True = keep, False = drop (window overlaps an excluded interval)
    """
    starts = window_times_global
    ends = window_times_global + window_len
    keep = np.ones(len(window_times_global), dtype=bool)
    for ex_start, ex_end in exclude_intervals:
        overlap = (starts < ex_end) & (ends > ex_start)
        keep &= ~overlap
    return keep