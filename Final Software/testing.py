import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
import librosa
from scipy.signal import find_peaks
from scipy.spatial import distance as dist

from feature import(
    load_features_from_json
)

from background_functions import(
     read_audio_chunk,
)

from dtw import(
    dtw_curve
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
                       exclude_intervals, extractor, fs_new, vote_frac,
                       hop_s, frame_s, min_sep_s=0.5, alpha_max=3.0):

    templates = [st_feats, mt_feats, bt_feats]
    thresholds = [st_threshold, mt_threshold, bt_threshold]
    call_types = ["st", "mt", "bt"]

    # one entry per detected call
    all_starts, all_ends, all_scaled_costs, all_labels = [], [], [], []

    # detections closer together than this (in frames) count as one call
    min_sep_frames = max(1, round(min_sep_s / hop_s))
    print(f"calls closer than {min_sep_frames} frames are one call")

    for i in range(n_hours):
        # load 1 hour of audio and compute features for the whole hour
        f_s, audio_array = read_audio_chunk(audio_file, start_hour=i, duration_hour=1.0, fs_new=fs_new)
        feat_stream = extractor(audio_array, fs_new) # (n_freq, n_frames)

        template_costs = []     # per call type: (n_temps, n_frames), cost of each template at each frame
        scaled_cost_rows = []   # per call type: kth cost / threshold, <= 1 means "call"

        for temps, threshold in zip(templates, thresholds):
            costs_arr = np.stack([dtw_curve(t, feat_stream, extractor.metric) for t in temps])
            k = max(1, round(vote_frac * len(temps)))                 # same k as calibration
            kth_cost = np.sort(costs_arr, axis=0)[k - 1]              # kth lowest cost across templates, per frame
            template_costs.append(costs_arr)
            scaled_cost_rows.append(kth_cost / threshold) #divided by threshold so that when alpha = 1 we are at our specific threshold

        scaled_costs = np.nan_to_num(np.stack(scaled_cost_rows), nan=10.0, posinf=10.0)   # (3, n_frames)
        best_type = scaled_costs.argmin(axis=0)       # which call type fits best at each frame
        best_scaled_cost = scaled_costs.min(axis=0)   # its scaled cost

        # one frame per call: the lowest point of each dip in best_scaled_cost,
        # no higher than alpha_max, at least min_sep_frames apart
        call_end_frames, _ = find_peaks(-best_scaled_cost, height=-alpha_max, distance=min_sep_frames)

        for end_frame in call_end_frames:
            type_idx = best_type[end_frame]

            # the single template that matched best at this frame
            best_temp_idx = template_costs[type_idx][:, end_frame].argmin()
            best_template = templates[type_idx][best_temp_idx]

            # redo the DTW on a short stretch of the stream ending at end_frame, to find where the match began
            seg_lo = max(0, end_frame - 2 * best_template.shape[1])
            seg_feat = feat_stream[:, seg_lo:end_frame + 1]
            dist_mat = dist.cdist(best_template.T, seg_feat.T, extractor.metric)
            D, steps = librosa.sequence.dtw(C=dist_mat, subseq=True, backtrack=False, return_steps=True)
            warp_path = librosa.sequence.dtw_backtracking(steps, subseq=True, start=dist_mat.shape[1] - 1)
            start_frame = seg_lo + warp_path[-1, 1] # warp_path runs end -> start

            all_starts.append(i * 3600 + start_frame * hop_s - frame_s / 2)
            all_ends.append(i * 3600 + end_frame * hop_s + frame_s / 2)
            all_scaled_costs.append(best_scaled_cost[end_frame])
            all_labels.append(call_types[type_idx])

        n_detected = int((best_scaled_cost[call_end_frames] <= 1).sum())
        print(f"Processing hour {i+1} of {n_hours}: {n_detected} calls detected at the calibrated thresholds")

    all_starts, all_ends = np.array(all_starts), np.array(all_ends)
    all_scaled_costs, all_labels = np.array(all_scaled_costs), np.array(all_labels, dtype=object)

    # drop detections that overlap an excluded interval (downsweeps)
    keep = np.ones(len(all_starts), dtype=bool)
    for ex_start, ex_end in exclude_intervals:
        keep &= ~((all_starts < ex_end) & (all_ends > ex_start))

    return all_starts[keep], all_ends[keep], all_scaled_costs[keep], all_labels[keep]       

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

# def merge_consecutive_detections(all_detected_t, all_detected_labels,
#                                 step, window_len, max_gap_windows,
#                                 min_windows, max_windows
# ):
#     call_start_t = np.asarray(all_detected_t, dtype=float) #place all start times in a list
#     labels = np.asarray(all_detected_labels, dtype=object)
#     breaks_btwn_calls = np.diff(call_start_t)
#     breaks =  breaks_btwn_calls / step > 1 + max_gap_windows + 1e-6 #True if one or more gaps of 75% of window
    
#     is_first = np.r_[True, breaks] # True if there was a break just before that time
#     first_window = np.arange(len(call_start_t))[is_first] #stores index of start time of merged calls

#     is_last = np.r_[breaks, True] # True where a group finishes
#     last_window = np.arange(len(call_start_t))[is_last]  #stores index of the start time of the final window of a set of merged calls
    
#     if max_windows is not None:
#         new_first_window, new_last_window = [], []
#         for first, last in zip(first_window, last_window):
#             for chunk_start in range(first, last + 1, max_windows): #(start, stop, step)
#                 chunk_end = min(chunk_start + max_windows - 1, last)
#                 new_first_window.append(chunk_start)
#                 new_last_window.append(chunk_end)
#         first = np.array(new_first_window)
#         last = np.array(new_last_window)

#     keep = (last - first + 1) >= min_windows
#     uniq, codes = np.unique(labels, return_inverse=True)
#     onehot = np.zeros((len(call_start_t), len(uniq)), dtype=np.int32)
#     onehot[np.arange(len(call_start_t)), codes] = 1
#     dominant = uniq[np.add.reduceat(onehot, first, axis=0).argmax(axis=1)]

#     start_t = call_start_t[first[keep]]
#     end_t = call_start_t[last[keep]]+ window_len
#     kept_labels = dominant[keep]

#     return [(float(s), float(e), lab) for s, e, lab in zip(start_t, end_t, kept_labels)] 

def match_detections(detections, raven_table, ):#ignore_duplicates):
    detections = sorted(detections, key=lambda x: x[0])  # sort by start time   
    
    rt_start_times = np.array([g["start"] for g in raven_table])
    rt_end_times = np.array([g["end"] for g in raven_table])

    #set up empty lists
    det_touches = np.zeros(len(detections), dtype=bool) #detection overlaps >= 1 labelled call
    calls_touched = np.zeros(len(raven_table), dtype=bool)

    #rt_calls_matched = [False] * len(raven_table)
    #detections_matched = [False] * len(detections)

    #duplicate = [False] * len(detections)

    for i, (det_start, det_end, _) in enumerate(detections):
        rt_lo = np.searchsorted(rt_end_times, det_start, side="right") #if a label ends before the detected starts it is out ...lo
        rt_hi = np.searchsorted(rt_start_times, det_end, side="left")#if a label starts after det ends it is out ...hi

        if rt_hi > rt_lo:#if raven ends after det starts and starts before det ends
            det_touches[i] = True
            calls_touched[rt_lo:rt_hi] = True

        #best_index, best_overlap, touched = None, 0.0, False
        # for j in range(rt_lo, rt_hi):
        #     overlap = min(det_end, rt_end_times[j]) - max(det_start, rt_start_times[j])
        #     if overlap > 0:
        #         touched = True                      # overlaps *some* real call
        #     if rt_calls_matched[j]:
        #         continue
        #     if overlap > best_overlap:
        #         best_overlap = overlap
        #         best_index = j
        # if best_index is not None:
        #     rt_calls_matched[best_index] = True
        #     detections_matched[i] = True
        # elif touched and ignore_duplicates:
        #     duplicate[i] = True                      # real call already matched by another detection

    tp_det = int(sum(det_touches))
    fp = len(detections) - tp_det
    tp_calls = int(sum(calls_touched))
    fn = len(raven_table) - tp_calls

    false_positive_detections = [dets for dets, touches in zip(detections, det_touches)if not touches]
    false_negative_calls = [calls for calls, touched in zip(raven_table, calls_touched) if not touched]
    return tp_det, fp, tp_calls, fn, false_positive_detections, false_negative_calls

def compute_pr_curve_alpha_new(starts, ends, scaled_costs, labels, ground_truth, alphas):
    precision, recall, used = [], [], []
    for a in alphas:
        is_call = scaled_costs <= a
        calls = [(float(s), float(e), lab) for s, e, lab in zip(starts[is_call], ends[is_call], labels[is_call])]
        if not calls:
            continue
        tp_det, fp, tp_calls, fn, _, _ = match_detections(calls, ground_truth)
        precision.append(tp_det / (tp_det + fp))
        recall.append(tp_calls / (tp_calls + fn) if (tp_calls + fn) > 0 else 0.0)
        used.append(a)
    return np.array(precision), np.array(recall), np.array(used)

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

def plot_mfcc(m, fs_new, hop_samples, ax=None, normalise=False):
    """Plot an (n_coeffs, n_frames) MFCC matrix against real time."""
    img_data = m
    if normalise:  # display only: stops one large-magnitude row from washing out the rest
        img_data = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)

    n_coeffs, n_frames = m.shape
    t = np.arange(n_frames) * (hop_samples / fs_new)
    coeff_idx = np.arange(n_coeffs)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    img = ax.pcolormesh(t, coeff_idx, img_data, shading="auto")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC coefficient")

    if standalone:
        ax.set_title("MFCC")
        plt.colorbar(img, ax=ax, label="Coefficient value")

    return img

def plot_detection_spectrograms(detections, wav_path,
                                fs_new,
                                framelength,
                                noverlap,
                                n_examples,                          
                                extractor,
                                feat_method,
                                context=0.5,
                                hour_duration=3600.0,
                                random_sample=True,
                                seed=0,
                                title_prefix="FP",
):
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
    n_cols = 2 if feat_method == "mfcc" else 3   # STFT+MFCC pair, or STFT alone
    n_rows = n_plot if feat_method == "mfcc" else int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes) if feat_method == "mfcc" else np.atleast_1d(axes).flatten()

    hop_samples = int(fs_new * extractor.params["frame_dur"] * (1 - extractor.params["overlap"])) \
        if feat_method == "mfcc" else None

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

            if feat_method == "mfcc":
                ax_stft, ax_feat = axes[ax_idx, 0], axes[ax_idx, 1]
            else:
                ax_stft = axes[ax_idx]

            if len(segment) < framelength:
                ax_stft.set_title(f"{title_prefix}: {label} @ {start:.2f}s\n(segment too short)")
                ax_stft.axis("off")
                if feat_method == "mfcc":
                    ax_feat.axis("off")
                ax_idx += 1
                continue

            f, t, Zxx = signal.stft(segment, fs=f_s, nperseg=framelength, noverlap=noverlap, window="hamming")
            Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

            ax_stft.pcolormesh(t + seg_start_t, f, Zxx_db, shading="gouraud", cmap="viridis")
            ax_stft.axvline(local_start, color="red", linestyle="--", linewidth=1)
            ax_stft.axvline(local_end, color="red", linestyle="--", linewidth=1)
            ax_stft.set_title(f"{title_prefix}: {label} @ {start:.2f}s (dur={end-start:.2f}s)", fontsize=10)
            ax_stft.set_xlabel("Time (s, local)")
            ax_stft.set_ylabel("Freq (Hz)")

            if feat_method == "mfcc":
                feat = extractor(segment.astype(np.float32), f_s)
                plot_mfcc(feat, f_s, hop_samples, ax=ax_feat)
                ax_feat.set_title(f"{extractor.name.upper()}", fontsize=10)
                # shift the feature's local time axis to match the STFT panel's global offset
                for coll in ax_feat.collections:
                    coll.set_clim()  # no-op placeholder if you want shared color scaling later

            ax_idx += 1

    # turn off any completely unused rows/slots
    flat_axes = axes.flatten()
    used = ax_idx * n_cols
    for j in range(used, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.tight_layout()
    plt.show()
