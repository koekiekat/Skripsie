import json
import random

import ipywidgets as widgets
from IPython.display import display

from background_functions import read_audio_file, start_end_times, split_audio_segments, call_length_stats
from stft import short_time_calc, plot_spectrogram

CALL_TYPES = ["single_tone", "multi_tone", "burst_tonal", "tonal_downsweep"]

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


def extract_background_segments(wav_path, text_path, n_segments,
                                  file_label, recording_label, results_dir,
                                  buffer=0.5, seed=None):
    """
    Randomly extract `n_segments` background (non-call) audio segments from
    a full recording, avoiding any region within `buffer` seconds of a
    labeled call. Each segment's duration is randomized between the
    shortest and longest labeled call length in this recording, so
    background segments span the same length range as real calls.

    wav_path: path to the full recording
    text_path: path to the corresponding selections/labels file (used to
        know where calls are, and to derive the call-length range)
    n_segments: how many background segments to extract
    buffer: extra seconds of padding around each labeled call to avoid
    seed: optional random seed for reproducibility

    Saves the result to results_dir and also returns it. Each entry has
    call_type, start_time, end_time, and wav_path -- same shape as saved
    templates, so it can be loaded/used the same way downstream.
    """
    if seed is not None:
        random.seed(seed)

    f_s, x = read_audio_file(wav_path)
    df, start_t, end_t = start_end_times(text_path)
    _, longest, shortest = call_length_stats(df)

    recording_duration = len(x) / f_s
    forbidden = [(max(0, s - buffer), min(recording_duration, e + buffer))
                 for s, e in zip(start_t, end_t)]
    forbidden.sort()

    def overlaps_forbidden(cand_start, cand_end):
        for f_start, f_end in forbidden:
            if cand_start < f_end and cand_end > f_start:
                return True
        return False

    background = []
    max_attempts = n_segments * 200
    attempts = 0

    while len(background) < n_segments and attempts < max_attempts:
        attempts += 1
        duration = random.uniform(shortest, longest)
        cand_start = random.uniform(0, recording_duration - duration)
        cand_end = cand_start + duration

        if overlaps_forbidden(cand_start, cand_end):
            continue
        if any(cand_start < b["end_time"] and cand_end > b["start_time"]
               for b in background):
            continue

        background.append({
            "call_type": "background",
            "start_time": cand_start,
            "end_time": cand_end,
            "wav_path": str(wav_path),
        })

    if len(background) < n_segments:
        print(f"Warning: only found {len(background)}/{n_segments} valid "
              f"background segments after {attempts} attempts.")

    save_background_segments(background, file_label, recording_label, results_dir)
    print(f"Saved {len(background)} background segment(s) to "
          f"{_background_path(file_label, recording_label, results_dir).name}")
    return background