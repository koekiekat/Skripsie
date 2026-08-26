import json

import ipywidgets as widgets
from IPython.display import display

from background_functions import read_audio_file, start_end_times, split_audio_segments
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