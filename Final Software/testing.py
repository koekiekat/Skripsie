import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal

from feature import(
    load_features_from_json,
    batch_feature_windows
)

from background_functions import(
     read_audio_chunk,
)

from dtw import(
    dtw_costs_vectorized,
    dtw_costs_vectorized_new
)

def load_model_config(path):
    with open(path) as f:
        return json.load(f)

def load_all_template_features_new(results_dir, file_label, recording_label,
                                   extractor, n_temps, fs_new):
    out = []
    for call_type in ("single_tone", "multi_tone", "burst_tonal"):
        path = results_dir / f"{file_label}_{recording_label}_{call_type}_templates.json"
        out.append(load_features_from_json(path, extractor, n_temps, fs_new))
    return tuple(out)   # st_feats, mt_feats, bt_feats    

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

def get_exclusion_intervals(ground_truth, exclude_labels=("td",), pad=0.0):
    """Returns a list of (start, end) tuples for calls you want excluded, with optional padding (s)."""
    return [(g["start"] - pad, g["end"] + pad)
            for g in ground_truth if g["label"] in exclude_labels]

def detection_pipeline(audio_file, n_hours, st_feats, mt_feats, bt_feats,
                               st_threshold, mt_threshold, bt_threshold,
                               exclude_intervals, extractor, fs_new,
                               window_len, step_len, vote_frac, dtw_method):
    all_detected_t, all_detected_labels = [], [] #precision adn recall passed for my thresholds
    # all_scores, all_labels_all, all_times_all = [], [], [] #PR version
    all_times_all = []
    all_kth = []

    #converts seconds to samples
    win_samples = int(round(window_len * fs_new))
    step_samples = int(round(step_len * fs_new))



    for i in range(n_hours):
        #loads 1 hour of audio
        f_s, audio_array = read_audio_chunk(audio_file, start_hour=i, duration_hour=1.0, fs_new=fs_new)
        
        #computes features
        feats_window = batch_feature_windows(audio_array, win_samples, step_samples, f_s, extractor)
        
        #stores window start times for each winodw
        n_windows = feats_window.shape[0]
        window_times_global = np.arange(n_windows) * step_len + i * 3600

        #filter out downsweeps
        keep = get_exclusion_mask(window_times_global, window_len=window_len, exclude_intervals=exclude_intervals)
        feats_window = feats_window[keep]
        window_times_global = window_times_global[keep]

        #Costs calculated for windows against each template and stored
        if dtw_method == "old":
            costs_st, costs_mt, costs_bt = dtw_costs_vectorized(st_feats, mt_feats, bt_feats, feats_window, extractor.metric)
        elif dtw_method == "new":
            costs_st, costs_mt, costs_bt = dtw_costs_vectorized_new(st_feats, mt_feats, bt_feats, feats_window, extractor.metric)

        results = {}
        kth_cols = []

        window_info = {
            "st": (costs_st, st_threshold),
            "mt": (costs_mt, mt_threshold),
            "bt": (costs_bt, bt_threshold)
        }

        for label, (costs_list, threshold) in window_info.items():
            costs_arr = np.vstack(costs_list) #stacks costs one under the other
            n_t = costs_arr.shape[0]
            votes = (costs_arr < threshold).sum(axis=0) #counts how many templates is below threshold
            mean_cost = costs_arr.mean(axis=0) #avg cost used for label detection
            triggered = votes >= np.ceil(vote_frac * n_t) #true if 60% of templates are below thrshold
            results[label] = (triggered, mean_cost) # store results

            k = int(np.ceil(vote_frac * n_t))
            kth_cols.append(np.partition(costs_arr, k - 1, axis = 0)[k - 1])#store kth smallest cost per window

        n_windows = feats_window.shape[0]
        call_detected = np.zeros(n_windows, dtype=bool)
        call_labels = np.full(n_windows, "", dtype=object)
        best_cost = np.full(n_windows, np.inf)

        for label, (triggered, mean_cost) in results.items():

            triggered_better = triggered & (mean_cost < best_cost) #If two types triggger call lowest mean cost wins label
            call_labels[triggered_better] = label
            #best_cost[triggered_better] = mean_cost[triggered_better]
            call_detected |= triggered

        print(f"Processing hour {i+1} of {n_hours}: {call_detected.sum()} / {n_windows} windows flagged as calls")
        
        all_kth.append(np.stack(kth_cols, axis = 1).astype(np.float32))# (n_windows, 3) s_t, m_t, b_t
        all_times_all.extend(window_times_global.tolist())

        detected = np.where(call_detected)[0]
        all_detected_t.extend(window_times_global[detected].tolist())
        all_detected_labels.extend(call_labels[detected].tolist())

    return(np.array(all_detected_t), np.array(all_detected_labels), np.array(all_times_all), np.concatenate(all_kth, axis = 0))
       
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

def merge_consecutive_detections(all_detected_t, all_detected_labels,
                                step, window_len, max_gap_windows,
                                min_windows, max_windows
):
    call_start_t = np.asarray(all_detected_t, dtype=float) #place all start times in a list
    labels = np.asarray(all_detected_labels, dtype=object)
    breaks_btwn_calls = np.diff(call_start_t)
    breaks =  breaks_btwn_calls / step > 1 + max_gap_windows + 1e-6 #True if one or more gaps of 75% of window
    
    is_first = np.r_[True, breaks] # True if there was a break just before that time
    first_window = np.arange(len(call_start_t))[is_first] #stores index of start time of merged calls

    is_last = np.r_[breaks, True] # True where a group finishes
    last_window = np.arange(len(call_start_t))[is_last]  #stores index of the start time of the final window of a set of merged calls
    
    if max_windows is not None:
        new_first_window, new_last_window = [], []
        for first, last in zip(first_window, last_window):
            for chunk_start in range(first, last + 1, max_windows): #(start, stop, step)
                chunk_end = min(chunk_start + max_windows - 1, last)
                new_first_window.append(chunk_start)
                new_last_window.append(chunk_end)
        first = np.array(new_first_window)
        last = np.array(new_last_window)

    keep = (last - first + 1) >= min_windows
    uniq, codes = np.unique(labels, return_inverse=True)
    onehot = np.zeros((len(call_start_t), len(uniq)), dtype=np.int32)
    onehot[np.arange(len(call_start_t)), codes] = 1
    dominant = uniq[np.add.reduceat(onehot, first, axis=0).argmax(axis=1)]

    start_t = call_start_t[first[keep]]
    end_t = call_start_t[last[keep]]+ window_len
    kept_labels = dominant[keep]

    return [(float(s), float(e), lab) for s, e, lab in zip(start_t, end_t, kept_labels)] 

def match_detections(detections, raven_table, ignore_duplicates=False):
    detections = sorted(detections, key=lambda x: x[0])  # sort by start time   
    
    rt_start_times = np.array([g["start"] for g in raven_table])
    rt_end_times = np.array([g["end"] for g in raven_table])

    #set up empty lists
    rt_calls_matched = [False] * len(raven_table)
    detections_matched = [False] * len(detections)

    duplicate = [False] * len(detections)

    for i, (det_start, det_end, _) in enumerate(detections):
        rt_lo = np.searchsorted(rt_end_times, det_start, side="right")
        rt_hi = np.searchsorted(rt_start_times, det_end, side="left")

        best_index, best_overlap, touched = None, 0.0, False
        for j in range(rt_lo, rt_hi):
            overlap = min(det_end, rt_end_times[j]) - max(det_start, rt_start_times[j])
            if overlap > 0:
                touched = True                      # overlaps *some* real call
            if rt_calls_matched[j]:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = j
        if best_index is not None:
            rt_calls_matched[best_index] = True
            detections_matched[i] = True
        elif touched and ignore_duplicates:
            duplicate[i] = True                      # real call already matched by another detection

    tp = sum(detections_matched)
    fp = len(detections) - tp - sum(duplicate)
    fn = len(raven_table) - sum(rt_calls_matched)

    false_positive_detections = [d for d, m, dup in zip(detections, detections_matched, duplicate)
                                 if not m and not dup]
    false_negative_gts = [g for g, m in zip(raven_table, rt_calls_matched) if not m]
    return tp, fp, fn, false_positive_detections, false_negative_gts



def compute_pr_curve_alpha(times, kth_costs, thresholds, ground_truth, step, window_len, alphas, max_gap_windows, min_windows, max_windows):
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
        calls = windows_to_calls(times, best_ratio, best_label, a, step, window_len, max_gap_windows, min_windows, max_windows)
        if not calls:
            continue
        tp, fp, fn, _, _ = match_detections(calls, ground_truth)
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        used.append(a)
    return np.array(precision), np.array(recall), np.array(used), best_label

def windows_to_calls(times, scores, labels, threshold, step, window_len, max_gap, min_win, max_win):
    mask = scores <= threshold
    if not mask.any():
        return []
    return merge_consecutive_detections(times[mask], labels[mask], step=step, window_len=window_len, max_gap_windows=max_gap, min_windows=min_win, max_windows=max_win)

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

def compute_average_precision(precision, recall):
    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    p_interp = np.maximum.accumulate(p[::-1])[::-1]
    return np.sum(np.diff(r, prepend=0.0) * p_interp)

def plot_detection_spectrograms(detections, wav_path,
                                 fs_new=1000,
                                 framelength=None,
                                 noverlap=None,
                                 context=0.5,
                                 n_examples=12,
                                 hour_duration=3600.0,
                                 random_sample=True,
                                 seed=0,
                                 title_prefix="FP",
                                 extractor=None):
    """
    Plot spectrograms of a sample of detections. If `extractor` is given,
    also plots that extractor's feature (e.g. MFCC) alongside the STFT
    for each detection.
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

    from collections import defaultdict
    by_hour = defaultdict(list)
    for start, end, label in sample:
        hour_idx = int(start // hour_duration)
        by_hour[hour_idx].append((start, end, label))

    n_plot = len(sample)
    n_cols = 2 if extractor is not None else 3   # STFT+MFCC pair, or STFT alone
    n_rows = n_plot if extractor is not None else int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes) if extractor is not None else np.atleast_1d(axes).flatten()

    # hop_samples = int(fs_new * extractor.params["frame_dur"] * (1 - extractor.params["overlap"])) \
    #     if extractor is not None else None

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

            if extractor is not None:
                ax_stft, ax_feat = axes[ax_idx, 0], axes[ax_idx, 1]
            else:
                ax_stft = axes[ax_idx]

            if len(segment) < framelength:
                ax_stft.set_title(f"{title_prefix}: {label} @ {start:.2f}s\n(segment too short)")
                ax_stft.axis("off")
                if extractor is not None:
                    ax_feat.axis("off")
                ax_idx += 1
                continue

            f, t, Zxx = signal.stft(segment, fs=f_s, nperseg=framelength, noverlap=noverlap,
                                     window="hamming")
            Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

            ax_stft.pcolormesh(t + seg_start_t, f, Zxx_db, shading="gouraud", cmap="viridis")
            ax_stft.axvline(local_start, color="red", linestyle="--", linewidth=1)
            ax_stft.axvline(local_end, color="red", linestyle="--", linewidth=1)
            ax_stft.set_title(f"{title_prefix}: {label} @ {start:.2f}s (dur={end-start:.2f}s)", fontsize=10)
            ax_stft.set_xlabel("Time (s, local)")
            ax_stft.set_ylabel("Freq (Hz)")

            # if extractor is not None:
            #     feat = extractor(segment.astype(np.float32), f_s)
            #     plot_mfcc(feat, f_s, hop_samples, ax=ax_feat)
            #     ax_feat.set_title(f"{extractor.name.upper()}", fontsize=10)
            #     # shift the feature's local time axis to match the STFT panel's global offset
            #     for coll in ax_feat.collections:
            #         coll.set_clim()  # no-op placeholder if you want shared color scaling later

            ax_idx += 1

        # turn off any completely unused rows/slots
    flat_axes = axes.flatten()
    used = ax_idx * n_cols
    for j in range(used, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.tight_layout()
    plt.show()
