import json
import random
from pathlib import Path
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

from background_functions import read_audio_file, start_end_times, split_audio_segments, call_length_stats
from stft import short_time_calc, plot_spectrogram

CALL_TYPES = ["single_tone", "multi_tone", "burst_tonal", "tonal_downsweep"]

MIN_STFT_DURATION = 0.128  # seconds -- matches framelength in stft.py's short_time_calc

# Map the codes used in your "Call" column to your internal call types.
CALL_CODE_MAP = {
    "ST": "single_tone",
    "MT": "multi_tone",
    "BT": "burst_tonal",
    "TD": "tonal_downsweep",
}

def labels_from_dataframe(df, label_map=None):
    """
    Build an {index: call_type} dict from a selections dataframe that
    already has a 'Call' column with ground-truth codes, using the same
    row indices that split_audio_segments/segments will use (i.e. after
    the same de-duplication start_end_times performs).
    """
    label_map = label_map or CALL_CODE_MAP
    labels = {}
    unmapped_counts = {}
    for idx, code in enumerate(df["Call"]):
        code_clean = str(code).strip().upper()
        if code_clean in label_map:
            labels[idx] = label_map[code_clean]
        else:
            unmapped_counts[code_clean] = unmapped_counts.get(code_clean, 0) + 1

    if unmapped_counts:
        print(f"Warning: unrecognised codes left unlabeled: {unmapped_counts}")
    return labels

def load_file_calls_with_labels(wav_path, text_path, label_map=None):
    """
    Load a WAV + selections file and pull the ground-truth call type
    for each call straight out of the 'Call' column.
    """
    f_s, x = read_audio_file(wav_path)
    df, start_t, end_t = start_end_times(text_path)
    segments = split_audio_segments(x, f_s, start_t, end_t)
    labels = labels_from_dataframe(df, label_map)
    return f_s, x, start_t, end_t, segments, labels

def _template_path(file_label, recording_label, call_type, results_dir):
    return results_dir / f"{file_label}_{recording_label}_{call_type}_templates.json"

def load_templates(file_label, recording_label, call_type, results_dir):
    """Load previously saved templates for this call_type (empty list if none)."""
    path = _template_path(file_label, recording_label, call_type, results_dir)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

def save_templates(templates, file_label, recording_label, call_type, results_dir):
    path = _template_path(file_label, recording_label, call_type, results_dir)
    with open(path, "w") as f:
        json.dump(templates, f, indent=2)

def select_templates_interactive(wav_path, segments, f_s, start_t, end_t, labels,
                                  call_type, n_templates, file_label, recording_label,
                                  results_dir):
    """
    Step through every call already labeled as `call_type` and let you pick,
    via button click, which ones to keep as templates -- until `n_templates`
    have been selected or there are no more candidates left.

    Progress is written to disk after every selection, so re-running this
    with the same arguments resumes from where you left off (already-saved
    templates are skipped and don't count against the remaining candidates).

    Each saved template records: source index, call_type, start/end time,
    and the wav file it came from.
    """
    if call_type not in CALL_TYPES:
        print(f"Warning: '{call_type}' is not in CALL_TYPES {CALL_TYPES}. "
              f"Continuing anyway.")

    existing = load_templates(file_label, recording_label, call_type, results_dir)
    if len(existing) >= n_templates:
        print(f"Already have {len(existing)} template(s) for '{call_type}' "
              f"in {file_label}/{recording_label} (target {n_templates}). "
              f"Nothing to do.")
        return existing

    already_idx = {t["index"] for t in existing}
    candidate_idx = [idx for idx, lab in labels.items()
                      if lab == call_type and idx not in already_idx]

    if not candidate_idx:
        print(f"No unreviewed '{call_type}' calls left to look at.")
        return existing

    fs_new = 1000
    state = {"pos": 0}
    out = widgets.Output()
    status = widgets.Label()
    btn_select = widgets.Button(description="Select as template", button_style="success")
    btn_skip = widgets.Button(description="Skip")
    btn_stop = widgets.Button(description="Stop", button_style="warning")

    display(
        widgets.HTML(f"<b>{recording_label} — selecting '{call_type}' templates "
                      f"for {file_label} (target {n_templates})</b>"),
        status,
        widgets.HBox([btn_select, btn_skip, btn_stop]),
        out,
    )

    def update_status():
        status.value = (f"Templates saved: {len(existing)}/{n_templates} | "
                         f"Candidates remaining: {len(candidate_idx) - state['pos']}")

    def show_next():
        out.clear_output(wait=True)
        if len(existing) >= n_templates:
            finalize("Target reached.")
            return
        if state["pos"] >= len(candidate_idx):
            finalize("No more candidates.")
            return
        idx = candidate_idx[state["pos"]]
        with out:
            f_sig, t_sig, Zxx_sig, fsn = short_time_calc(segments[idx], f_s, fs_new)
            plot_spectrogram(f_sig, t_sig, Zxx_sig, fsn)
            print(f"candidate {state['pos'] + 1}/{len(candidate_idx)} | index {idx} | "
                  f"start_t {start_t[idx]:.2f} | end_t {end_t[idx]:.2f}")
        update_status()

    def on_select(b):
        idx = candidate_idx[state["pos"]]
        existing.append({
            "index": idx,
            "call_type": call_type,
            "start_time": float(start_t[idx]),
            "end_time": float(end_t[idx]),
            "wav_path": str(wav_path),
        })
        save_templates(existing, file_label, recording_label, call_type, results_dir)
        state["pos"] += 1
        show_next()

    def on_skip(b):
        state["pos"] += 1
        show_next()

    def finalize(reason):
        out.clear_output(wait=True)
        with out:
            print(f"Stopped: {reason} Saved {len(existing)}/{n_templates} "
                  f"template(s) for '{call_type}'.")
        for b in (btn_select, btn_skip, btn_stop):
            b.disabled = True

    btn_select.on_click(on_select)
    btn_skip.on_click(on_skip)
    btn_stop.on_click(lambda b: finalize("Stopped by user."))

    show_next()
    return existing

def _background_path(file_label, recording_label, results_dir):
    return results_dir / f"{file_label}_{recording_label}_background.json"

def load_background_segments(file_label, recording_label, results_dir):
    """Load previously saved background segments (empty list if none)."""
    path = _background_path(file_label, recording_label, results_dir)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

def save_background_segments(background, file_label, recording_label, results_dir):
    path = _background_path(file_label, recording_label, results_dir)
    with open(path, "w") as f:
        json.dump(background, f, indent=2)

def extract_background_segments_interactive(wav_path, text_path, n_segments,
                                              file_label, recording_label, results_dir,
                                              buffer=0.5, seed=None,
                                              candidate_pool_size=None,
                                              fs_new=1000):
    """
    Randomly generate candidate background (non-call) segments spread
    across the full duration of a (potentially very long, e.g. 72-hour)
    recording, avoiding any labeled call region, then let you visually
    accept or reject each one via button click before saving.

    n_segments: how many *accepted* background segments you want in total
    candidate_pool_size: how many candidates to generate up front to
        review (defaults to 3x n_segments -- increase this if you expect
        to reject a lot, e.g. from other unlabeled noise/signals)
    """
    if seed is not None:
        random.seed(seed)
    if candidate_pool_size is None:
        candidate_pool_size = n_segments * 3

    f_s, x = read_audio_file(wav_path)
    df, start_t, end_t = start_end_times(text_path)
    _, longest, shortest = call_length_stats(df)

    recording_duration = len(x) / f_s
    forbidden = sorted((max(0, s - buffer), min(recording_duration, e + buffer))
                        for s, e in zip(start_t, end_t))

    def overlaps(cand_start, cand_end, intervals):
        for f_start, f_end in intervals:
            if cand_start < f_end and cand_end > f_start:
                return True
        return False

    candidates = []
    max_attempts = candidate_pool_size * 200
    attempts = 0
    while len(candidates) < candidate_pool_size and attempts < max_attempts:
        attempts += 1
        duration = random.uniform(max(shortest, MIN_STFT_DURATION), longest)
        cand_start = random.uniform(0, recording_duration - duration)
        cand_end = cand_start + duration
        if overlaps(cand_start, cand_end, forbidden):
            continue
        if overlaps(cand_start, cand_end, candidates):
            continue
        candidates.append((cand_start, cand_end))

    if len(candidates) < n_segments:
        print(f"Warning: only generated {len(candidates)} candidate(s) "
              f"after {attempts} attempts -- may not reach {n_segments} "
              f"accepted segments.")

    accepted = []
    state = {"pos": 0}
    out = widgets.Output()
    status = widgets.Label()
    btn_accept = widgets.Button(description="Accept", button_style="success")
    btn_reject = widgets.Button(description="Reject (has signal)", button_style="danger")
    btn_stop = widgets.Button(description="Stop", button_style="warning")

    display(
        widgets.HTML(f"<b>{recording_label} — reviewing background candidates "
                      f"(target {n_segments})</b>"),
        status,
        widgets.HBox([btn_accept, btn_reject, btn_stop]),
        out,
    )

    def update_status():
        status.value = (f"Accepted: {len(accepted)}/{n_segments} | "
                         f"Candidates remaining: {len(candidates) - state['pos']}")

    def show_next():
        out.clear_output(wait=True)
        if len(accepted) >= n_segments:
            finalize("Target reached.")
            return
        if state["pos"] >= len(candidates):
            finalize("No more candidates.")
            return

        cand_start, cand_end = candidates[state["pos"]]
        start_sample = int(cand_start * f_s)
        end_sample = int(cand_end * f_s)
        segment = x[start_sample:end_sample]

        with out:
            f_sig, t_sig, Zxx_sig, fsn = short_time_calc(segment, f_s, fs_new)
            plot_spectrogram(f_sig, t_sig, Zxx_sig, fsn)
            print(f"candidate {state['pos'] + 1}/{len(candidates)} | "
                  f"start_t {cand_start:.2f} | end_t {cand_end:.2f} "
                  f"({cand_start / 3600:.2f} hr into recording)")
        update_status()

    def on_accept(b):
        cand_start, cand_end = candidates[state["pos"]]
        accepted.append({
            "call_type": "background",
            "start_time": cand_start,
            "end_time": cand_end,
            "wav_path": str(wav_path),
        })
        state["pos"] += 1
        show_next()

    def on_reject(b):
        state["pos"] += 1
        show_next()

    def finalize(reason):
        out.clear_output(wait=True)
        with out:
            print(f"Stopped: {reason} Accepted {len(accepted)}/{n_segments} "
                  f"background segment(s).")
        for b in (btn_accept, btn_reject, btn_stop):
            b.disabled = True
        save_background_segments(accepted, file_label, recording_label, results_dir)
        print(f"Saved {len(accepted)} background segment(s) to "
              f"{_background_path(file_label, recording_label, results_dir).name}")
        plot_background_coverage(accepted, recording_duration)

    btn_accept.on_click(on_accept)
    btn_reject.on_click(on_reject)
    btn_stop.on_click(lambda b: finalize("Stopped by user."))

    show_next()
    return accepted

def plot_background_coverage(accepted, recording_duration):
    """
    Quick sanity-check plot: shows where each accepted background segment
    falls across the recording's full duration, so you can confirm they're
    actually spread out rather than clustered.
    """
    if not accepted:
        return
    starts_hr = [a["start_time"] / 3600 for a in accepted]
    plt.figure(figsize=(10, 1.5))
    plt.scatter(starts_hr, [0] * len(starts_hr), alpha=0.7)
    plt.xlim(0, recording_duration / 3600)
    plt.yticks([])
    plt.xlabel("Time into recording (hours)")
    plt.title("Coverage of accepted background segments")
    plt.show()

def review_saved_calls(json_path, fs_new=1000, start_idx=0):
    """
    Step through every entry in a saved templates/background JSON file
    (as produced by select_templates_interactive or
    extract_background_segments), showing each one's spectrogram along
    with its metadata. Click 'Next' to advance. Purely for viewing/QA --
    nothing is saved or modified.

    json_path: path to the saved .json file (a template or background file)
    start_idx: position to start from (handy for resuming a review)
    """
    with open(json_path) as f:
        entries = json.load(f)

    if not entries:
        print(f"No entries found in {json_path}.")
        return

    order = list(range(start_idx, len(entries)))
    state = {"pos": 0}
    out = widgets.Output()
    status = widgets.Label()
    btn_next = widgets.Button(description="Next", button_style="info")
    btn_stop = widgets.Button(description="Stop", button_style="warning")

    display(
        widgets.HTML(f"<b>Reviewing {Path(json_path).name}</b>"),
        status,
        widgets.HBox([btn_next, btn_stop]),
        out,
    )

    def show_current():
        out.clear_output(wait=True)
        if state["pos"] >= len(order):
            with out:
                print("Done reviewing all entries.")
            btn_next.disabled = True
            return

        pos = order[state["pos"]]
        entry = entries[pos]

        f_s, x = read_audio_file(entry["wav_path"])
        start_sample = int(entry["start_time"] * f_s)
        end_sample = int(entry["end_time"] * f_s)
        segment = x[start_sample:end_sample]

        with out:
            f_sig, t_sig, Zxx_sig, fsn = short_time_calc(segment, f_s, fs_new)
            plot_spectrogram(f_sig, t_sig, Zxx_sig, fsn)
            label_line = (f"entry {state['pos'] + 1}/{len(order)} | "
                          f"call_type: {entry.get('call_type')} | "
                          f"start_t {entry['start_time']:.2f} | "
                          f"end_t {entry['end_time']:.2f}")
            if "index" in entry:
                label_line += f" | source index {entry['index']}"
            print(label_line)

        status.value = f"Remaining: {len(order) - state['pos'] - 1}"

    def on_next(b):
        state["pos"] += 1
        show_current()

    def on_stop(b):
        out.clear_output(wait=True)
        with out:
            print(f"Stopped at position {order[state['pos']]} "
                  f"(entry {state['pos'] + 1}/{len(order)}). "
                  f"Pass start_idx={order[state['pos']]} to resume here.")
        btn_next.disabled = True

    btn_next.on_click(on_next)
    btn_stop.on_click(on_stop)

    show_current()