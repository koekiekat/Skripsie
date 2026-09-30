import json
import pandas as pd
import numpy as np

from feature import(
    load_features_from_json,
    batch_feature_windows
)

from background_functions import(
     read_audio_chunk,
)

from dtw import(
    dtw_costs_vectorized
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
                               window_len, step_len, vote_frac):
    all_detected_t, all_detected_labels = [], [] #precision adn recall passed for my thresholds
    all_scores, all_labels_all, all_times_all = [], [], [] #PR version
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
        costs_st, costs_mt, costs_bt = dtw_costs_vectorized(st_feats, mt_feats, bt_feats, feats_window, extractor.metric)

        results = {}
        #kth_cols = []

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

            #k = int(np.ceil(vote_frac * n_t))
            #kth_cols.append(np.partition(costs_arr, k - 1, axis = 0)[k - 1])#store kth smallest cost per window

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
        
        #all_kth.append(np.stack(kth_cols, axis = 1).astype(np.float32))# (n_windows, 3) s_t, m_t, b_t
        #all_times_all.extend(window_times_global.tolist())

        detected = np.where(call_detected)[0]
        all_detected_t.extend(window_times_global[detected].tolist())
        all_detected_labels.extend(call_labels[detected].tolist())

    return(np.array(all_detected_t), np.array(all_detected_labels)) #np.concatenate(all_kth, axis = 0)
       
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