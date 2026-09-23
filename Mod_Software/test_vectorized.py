import soundfile as sf
import numpy as np
from stft import resample_audio
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import random
from dtw import dtw_calc_new, batched_dtw_costs_for_template_new
from calibration_calcs import load_threshold_new
from feature import load_features_from_json

def load_all_template_features_new(results_dir, file_label, recording_label,
                                   extractor, n_templates=10, fs_new=1000):
    out = []
    for call_type in ("single_tone", "multi_tone", "burst_tonal"):
        path = results_dir / f"{file_label}_{recording_label}_{call_type}_templates.json"
        out.append(load_features_from_json(path, extractor, n_templates, fs_new))
    return tuple(out)   # st_feats, mt_feats, bt_feats

def load_all_thresholds_new(results_dir, file_label, extractor):
    """Loads thresholds saved for this extractor, and checks they were made with the same settings."""
    thresholds = []
    for call_type in ("single_tone", "multi_tone", "burst_tonal"):
        r = load_threshold_new(file_label, call_type, results_dir, extractor.name)
        if r is None:
            raise FileNotFoundError(
                f"No '{extractor.name}' threshold for '{call_type}' ({file_label}) in {results_dir}")
        if r.get("feature_params") != extractor.params:
            raise ValueError(
                f"Threshold for '{call_type}' was calibrated with {r.get('feature_params')}, "
                f"but the extractor now has {extractor.params}")
        thresholds.append(r["threshold"])
    return tuple(thresholds)

def batch_feature_windows(audio_array, window_len, step_len, fs, extractor):
    """window_len / step_len in SAMPLES. Returns (n_windows, n_features, n_frames)."""
    windows = np.lib.stride_tricks.sliding_window_view(audio_array, window_len)[::step_len]
    return extractor.batch(windows, fs)

def dtw_costs_vectorized_new(st_feats, mt_feats, bt_feats, window_feats, metric):
    st = [batched_dtw_costs_for_template_new(t, window_feats, metric) for t in st_feats]
    mt = [batched_dtw_costs_for_template_new(t, window_feats, metric) for t in mt_feats]
    bt = [batched_dtw_costs_for_template_new(t, window_feats, metric) for t in bt_feats]
    return st, mt, bt

def run_detection_pipeline_new(audio_file, n_hours, st_feats, mt_feats, bt_feats,
                               st_threshold, mt_threshold, bt_threshold,
                               exclude_intervals, extractor, fs_new=1000,
                               window_len=0.15, step_len=0.075, vote_frac=0.6):
    all_detected_t, all_detected_labels = [], [] #precision adn recall passed for my thresholds
    all_scores, all_labels_all, all_times_all = [], [], [] #PR version
    all_kth = []

#converts seconds to samples
    win_samples = int(round(window_len * fs_new))
    step_samples = int(round(step_len * fs_new))

    for i in range(n_hours):
        #loads 1 hour of audio
        f_s, audio_array = read_audio_chunk(audio_file, i, duration_hour=1.0, fs_new=fs_new)

        #computes features
        feats_window = batch_feature_windows(audio_array, win_samples, step_samples, f_s, extractor)

        #stores window start times for each winodw
        n_windows = feats_window.shape[0]
        window_times_global = np.arange(n_windows) * step_len + i * 3600

        #Filter out tonal-downsweeps
        keep = get_exclusion_mask(window_times_global, window_len=window_len,
                                  exclude_intervals=exclude_intervals)
        feats_window = feats_window[keep]
        window_times_global = window_times_global[keep]

        #Costst calculated for windows against each template and stored
        costs_st, costs_mt, costs_bt = dtw_costs_vectorized_new(
            st_feats, mt_feats, bt_feats, feats_window, extractor.metric)

        #is it a call or not?
        results = {}
        kth_cols = []
        for costs_list, threshold, label in [(costs_st, st_threshold, "st"), 
                                             (costs_mt, mt_threshold, "mt"), 
                                             (costs_bt, bt_threshold, "bt")]: #loops through each call type one at a time
            costs_arr = np.vstack(costs_list) #stacks costs one under the other
            n_t = costs_arr.shape[0]
            votes = (costs_arr < threshold).sum(axis=0) #counts how many templates is below threshold
            mean_cost = costs_arr.mean(axis=0) #avg cost
            triggered = votes >= np.ceil(vote_frac * n_t) #true if 60% of templates are below thrshold
            results[label] = (triggered, mean_cost) # store results

            k = int(np.ceil(vote_frac * n_t))
            kth_cols.append(np.partition(costs_arr, k - 1, axis = 0)[k - 1])#store kth smallest cost per window

        n_windows = feats_window.shape[0]
        call_detected = np.zeros(n_windows, dtype=bool)
        call_labels = np.full(n_windows, "", dtype=object)
        best_cost = np.full(n_windows, np.inf)
        best_cost_all = np.full(n_windows, np.inf)
        best_label_all = np.full(n_windows, "", dtype=object)

        for label, (triggered, mean_cost) in results.items():
            all_time_better = mean_cost < best_cost_all #Tracks lowest mean cost across types(ignore thresholds and votes) used for PR curve
            best_label_all[all_time_better] = label
            best_cost_all[all_time_better] = mean_cost[all_time_better]

            triggered_better = triggered & (mean_cost < best_cost) #If two types triggger call lowest mean cost wins label
            call_labels[triggered_better] = label
            best_cost[triggered_better] = mean_cost[triggered_better]
            call_detected |= triggered

        print(f"Processing hour {i+1} of {n_hours}: {call_detected.sum()} / {n_windows} windows flagged as calls")

        all_scores.extend(best_cost_all.tolist())
        all_kth.append(np.stack(kth_cols, axis = 1).astype(np.float32))# (n_windows, 3) s_t, m_t, b_t
        all_labels_all.extend(best_label_all.tolist())
        all_times_all.extend(window_times_global.tolist())

        #stores only detected calls
        detected = np.where(call_detected)[0]
        all_detected_t.extend(window_times_global[detected].tolist())
        all_detected_labels.extend(call_labels[detected].tolist())

    return (np.array(all_detected_t), np.array(all_detected_labels, dtype=object),
            np.array(all_scores), np.array(all_labels_all, dtype=object),
            np.array(all_times_all), np.concatenate(all_kth, axis = 0))

def merge_consecutive_detections(all_detected_t, all_detected_labels, step=0.075,
                                 window_len=0.15, tol=1e-6, max_gap_windows=1,
                                 split_on_label=False, min_windows=1, max_windows=None):
    t = np.asarray(all_detected_t, dtype=float)
    if len(t) == 0:
        return []
    labels = np.asarray(all_detected_labels, dtype=object)

    breaks = np.diff(t) / step > 1 + max_gap_windows + tol
    if split_on_label:
        breaks |= labels[1:] != labels[:-1]

    starts = np.concatenate(([0], np.flatnonzero(breaks) + 1))
    ends = np.concatenate((starts[1:], [len(t)])) - 1          # inclusive

    if max_windows is not None:
        new_starts, new_ends = [], []
        for s, e in zip(starts, ends):
            length = e - s + 1
            for chunk_start in range(s, e + 1, max_windows):
                chunk_end = min(chunk_start + max_windows - 1, e)
                new_starts.append(chunk_start)
                new_ends.append(chunk_end)
        starts = np.array(new_starts)
        ends = np.array(new_ends)

    keep = (ends - starts + 1) >= min_windows

    uniq, codes = np.unique(labels, return_inverse=True)
    onehot = np.zeros((len(t), len(uniq)), dtype=np.int32)
    onehot[np.arange(len(t)), codes] = 1
    dominant = uniq[np.add.reduceat(onehot, starts, axis=0).argmax(axis=1)]

    return [(float(t[s]), float(t[e]) + window_len, lab)
            for s, e, lab in zip(starts[keep], ends[keep], dominant[keep])]

def read_audio_chunk(audio_file, start_hour, duration_hour = 1.0, fs_new = 1000):
    start_sec = start_hour * 3600
    duration_sec = duration_hour * 3600
    with sf.SoundFile(audio_file) as f:
        f_s = f.samplerate
        f.seek(int(start_sec * f_s))
        audio_array = f.read(int(duration_sec * f_s), dtype="int16")
    resampled = resample_audio(audio_array, f_s, fs_new)

    return fs_new, resampled

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

def windows_to_calls(times, scores, labels, threshold, step, window_len, **merge_kw):
    mask = scores <= threshold
    if not mask.any():
        return []
    return merge_consecutive_detections(times[mask], labels[mask],
                                        step=step, window_len=window_len, **merge_kw)

def compute_pr_curve_alpha(times, kth_costs, thresholds, ground_truth, step, window_len,
                           alphas, require_label_match=False, **merge_kw):
    """
    kth_costs: (n_windows, 3) k-th smallest template cost for st, mt, bt
    thresholds: (st_threshold, mt_threshold, bt_threshold), the calibrated ones
    alphas: scale factors applied to all three thresholds together (1.0 = deployed rule)
    """
    ratios = kth_costs / np.asarray(thresholds, dtype=float)      # < 1 means that type triggers
    best_ratio = ratios.min(axis=1)                               # window is flagged if any type triggers
    best_label = np.array(["st", "mt", "bt"], dtype=object)[ratios.argmin(axis=1)]

    precision, recall, used = [], [], []
    for a in alphas:
        calls = windows_to_calls(times, best_ratio, best_label, a, step, window_len, **merge_kw)
        if not calls:
            continue
        tp, fp, fn, _, _ = match_detections(calls, ground_truth, require_label_match)
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        used.append(a)
    return np.array(precision), np.array(recall), np.array(used), best_label

def compute_pr_curve(times, scores, labels, ground_truth, step, window_len,
                     n_thresholds=60, require_label_match=False,
                     q_lo=1e-4, q_hi=0.05, **merge_kw):
                     #scores = lowest mean cost across three call types
                     #ground_truth = annptated calls with td removed
                     #n_thresholds, q_lo, q_hi = control thresholds tried
                     #**merg_kw collects extra keyword arguments(window size) for merging
    #Marks which scores are real numbers(score can be inf if all costs are NAN)
    finite = np.isfinite(scores)

    #makes n_thresholds numbers btwn q_lo and q_hi spaced by a constant ratio rather than constabt difference
    #quantile turns each fractioninto a score val
    #unique sorts values from lowest cost to highest and removes duplicates
    thresholds = np.unique(np.quantile(scores[finite], np.geomspace(q_lo, q_hi, n_thresholds)))
    #the reason for quantile is due to all useful thresholds being lower

    precision, recall, used = [], [], []

    #runs through all n_thresholds thresholds
    for t in thresholds:
        #turns thresholds into a list of calls(i.e. which calls are flagged for this threshold)
        calls = windows_to_calls(times, scores, labels, t, step, window_len, **merge_kw)
        if not calls:
            continue
        #calculates tp,fp,fn
        tp, fp, fn, _, _ = match_detections(calls, ground_truth, require_label_match)
        #calculate precision and recall
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        #records which thresholds' precision and recall have been stored
        used.append(t)

    return np.array(precision), np.array(recall), np.array(used)

def compute_average_precision(precision, recall):
    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    p_interp = np.maximum.accumulate(p[::-1])[::-1]
    return np.sum(np.diff(r, prepend=0.0) * p_interp)

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