import numpy as np
import json
import shutil
import ipywidgets as widgets
from IPython.display import display
from background_functions import read_audio_file, start_end_times, split_audio_segments
from stft import short_time_calc, plot_spectrogram

CALL_TYPES = ["single_tone", "multi_tone", "burst_tonal", "tonal_downsweep"]


def load_file_calls(wav_path, text_path):
    f_s, x = read_audio_file(wav_path)
    _, start_t, end_t = start_end_times(text_path)
    segments = split_audio_segments(x, f_s, start_t, end_t)
    return f_s, x, start_t, end_t, segments


def _classification_path(file_label, recording_label, results_dir):
    return results_dir / f"{file_label}_{recording_label}_classification.json"


def load_classification(file_label, recording_label, results_dir):
    """Load a previously saved (possibly partial) classification result."""
    path = _classification_path(file_label, recording_label, results_dir)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # JSON keys are always strings -> convert back to int call indices
    data["labels"] = {int(k): v for k, v in data["labels"].items()}
    return data


def save_classification(data, file_label, recording_label, results_dir):
    path = _classification_path(file_label, recording_label, results_dir)
    to_save = dict(data)
    to_save["labels"] = {str(k): v for k, v in data["labels"].items()}
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)


def summarize_classifications(file_label, results_dir):
    """
    Scan results_dir for every classification file belonging to this
    file_label (across all recordings) and print a status summary:
    how many calls are labeled per category, and whether each
    recording's pass is marked done.
    """
    paths = sorted(results_dir.glob(f"{file_label}_*_classification.json"))
    if not paths:
        print(f"{file_label}: NOT YET STARTED (no classification files found)")
        return []

    summaries = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        counts = {ct: 0 for ct in CALL_TYPES}
        for lab in data["labels"].values():
            if lab in counts:
                counts[lab] += 1
        status = "done" if data.get("done") else "in progress"
        print(f"{file_label} / {data['recording_label']}: {status} | {counts}")
        summaries.append(data)
    return summaries


def clear_call_selection(file_label, recording_label, results_dir,
                          call_type=None, confirm=False):
    """
    Clear saved classification labels for a recording, backing up the
    current file first so it can be undone with restore_call_selection().

    - call_type=None (default): clears ALL labels for this recording,
      effectively restarting it from scratch.
    - call_type="single_tone" / "multi_tone" / "burst_tonal": only clears
      calls currently labeled as that type, leaving other labels intact.

    Nothing is deleted until confirm=True. A backup (.json.bak) is written
    before any change, overwriting any previous backup for this recording.
    """
    path = _classification_path(file_label, recording_label, results_dir)
    if not path.exists():
        print(f"No saved file found for {file_label} / {recording_label}.")
        return

    if not confirm:
        scope = "ALL labels" if call_type is None else f"labels of type '{call_type}'"
        print(f"This will clear {scope} for {file_label} / {recording_label}. "
              f"Call again with confirm=True to proceed "
              f"(a backup will be kept so you can undo with restore_call_selection).")
        return

    backup_path = path.with_suffix(".json.bak")
    shutil.copy(path, backup_path)

    data = load_classification(file_label, recording_label, results_dir)

    if call_type is None:
        removed = len(data["labels"])
        data["labels"] = {}
    else:
        before = len(data["labels"])
        data["labels"] = {idx: lab for idx, lab in data["labels"].items()
                           if lab != call_type}
        removed = before - len(data["labels"])

    data["done"] = False
    save_classification(data, file_label, recording_label, results_dir)
    print(f"Cleared {removed} label(s) for {file_label} / {recording_label}. "
          f"Backup saved to {backup_path.name} — call restore_call_selection "
          f"to undo.")


def restore_call_selection(file_label, recording_label, results_dir):
    """Undo the most recent clear_call_selection() call for this recording."""
    path = _classification_path(file_label, recording_label, results_dir)
    backup_path = path.with_suffix(".json.bak")
    if not backup_path.exists():
        print(f"No backup found for {file_label} / {recording_label} — "
              f"nothing to restore.")
        return
    shutil.copy(backup_path, path)
    print(f"Restored {file_label} / {recording_label} from backup.")


def get_indices_by_type(file_label, recording_label, results_dir, call_type):
    """Convenience: pull out just the indices labeled as one call_type."""
    data = load_classification(file_label, recording_label, results_dir)
    if data is None:
        return []
    return [idx for idx, lab in data["labels"].items() if lab == call_type]


def split_template_calibration(idx_list, n_templates, n_calibration):
    """
    Given a list of call indices already known to belong to one category,
    split them into a template set and a calibration set the same way the
    old DTW-based workflow did (templates spread evenly across the list).
    Run this *after* classification, per category, if you still want that
    template/calibration split downstream.
    """
    idx_list = list(idx_list)
    n_total = min(len(idx_list), n_templates + n_calibration)
    idx_list = idx_list[:n_total]
    slots = (np.unique(np.linspace(0, len(idx_list) - 1,
                        min(n_templates, len(idx_list))).round().astype(int))
             if idx_list else np.array([], dtype=int))
    template_idx = [idx_list[i] for i in slots]
    calibration_idx = [idx_list[i] for i in range(len(idx_list)) if i not in slots]
    return template_idx, calibration_idx


def classify_calls_interactive(segments, f_s, start_t, file_label, recording_label,
                                results_dir, exclude_idx=None):
    """
    Step through every detected call in `segments` one at a time and label it
    via button click as single_tone / multi_tone / burst_tonal (or skip it).

    Progress is written to disk after *every* click, so if you stop partway
    through (or the kernel dies), just re-run this call with the same
    file_label/recording_label and it will resume from the first unlabeled
    call.

    file_label: your grouping/collection name, e.g. "20230429_Templates_5"
    recording_label: identifies which WAV these segments came from,
        e.g. "20230429_102841"
    exclude_idx: optional set of indices to skip (e.g. already used elsewhere)
    """
    existing = load_classification(file_label, recording_label, results_dir)
    if existing is None:
        existing = {
            "file_label": file_label,
            "recording_label": recording_label,
            "labels": {},
            "done": False,
        }

    if existing.get("done"):
        print(f"{file_label}/{recording_label} already fully classified "
              f"({len(existing['labels'])} calls). Delete the JSON file "
              f"if you want to redo it.")
        return existing

    exclude_idx = set(exclude_idx or [])
    fs_new = 1000
    num_calls = len(segments)
    order = [i for i in range(num_calls)
             if i not in exclude_idx and i not in existing["labels"]]

    state = {"pos": 0}
    out = widgets.Output()
    status = widgets.Label()
    btn_single = widgets.Button(description="Single-tone", button_style="info")
    btn_multi = widgets.Button(description="Multi-tone", button_style="info")
    btn_burst = widgets.Button(description="Burst-tonal", button_style="info")
    btn_skip = widgets.Button(description="Skip")
    btn_stop = widgets.Button(description="Stop / Pause", button_style="warning")

    display(
        widgets.HTML(f"<b>{recording_label} — classifying calls for {file_label}</b>"),
        status,
        widgets.HBox([btn_single, btn_multi, btn_burst, btn_skip, btn_stop]),
        out,
    )

    def show_next():
        out.clear_output(wait=True)
        if state["pos"] >= len(order):
            finalize()
            return
        idx = order[state["pos"]]
        with out:
            f_sig, t_sig, Zxx_sig, fsn = short_time_calc(segments[idx], f_s, fs_new)
            plot_spectrogram(f_sig, t_sig, Zxx_sig, fsn)
            print(f"call {state['pos'] + 1}/{len(order)} | index {idx} | "
                  f"start_t {start_t[idx]:.2f}")
        status.value = (f"Labeled so far: {len(existing['labels'])} | "
                         f"Remaining this session: {len(order) - state['pos']}")

    def label_current(call_type):
        idx = order[state["pos"]]
        existing["labels"][idx] = call_type
        save_classification(existing, file_label, recording_label, results_dir)
        state["pos"] += 1
        show_next()

    def on_skip(b):
        state["pos"] += 1
        show_next()

    def finalize():
        existing["done"] = True
        save_classification(existing, file_label, recording_label, results_dir)
        out.clear_output(wait=True)
        with out:
            counts = {ct: sum(1 for v in existing["labels"].values() if v == ct)
                      for ct in CALL_TYPES}
            print(f"Done. Totals: {counts}")
        for b in (btn_single, btn_multi, btn_burst, btn_skip, btn_stop):
            b.disabled = True

    btn_single.on_click(lambda b: label_current("single_tone"))
    btn_multi.on_click(lambda b: label_current("multi_tone"))
    btn_burst.on_click(lambda b: label_current("burst_tonal"))
    btn_skip.on_click(on_skip)
    btn_stop.on_click(lambda b: finalize())

    show_next()
    return existing

# Map the codes used in your "Call" column to your internal call types.
# Extend this if you have other codes (e.g. burst-tonal).
CALL_CODE_MAP = {
    "ST": "single_tone",
    "MT": "multi_tone",
    "BT": "burst_tonal",
    "TD": "tonal-downsweep",
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
    Same as load_file_calls, but also pulls the ground-truth call type
    for each call straight out of the 'Call' column instead of requiring
    manual classification.
    """
    f_s, x = read_audio_file(wav_path)
    df, start_t, end_t = start_end_times(text_path)
    segments = split_audio_segments(x, f_s, start_t, end_t)
    labels = labels_from_dataframe(df, label_map)
    return f_s, x, start_t, end_t, segments, labels


def classify_calls_from_labels(labels, file_label, recording_label, results_dir,
                                overwrite=False):
    """
    Non-interactive replacement for classify_calls_interactive: writes out
    the classification JSON straight from an already-known {index: call_type}
    dict (e.g. from labels_from_dataframe), so all the downstream helpers
    (summarize_classifications, get_indices_by_type, clear/restore, etc.)
    keep working unchanged.
    """
    path = _classification_path(file_label, recording_label, results_dir)
    if path.exists() and not overwrite:
        print(f"{file_label}/{recording_label} already has a saved "
              f"classification. Pass overwrite=True to replace it.")
        return load_classification(file_label, recording_label, results_dir)

    data = {
        "file_label": file_label,
        "recording_label": recording_label,
        "labels": dict(labels),
        "done": True,
    }
    save_classification(data, file_label, recording_label, results_dir)

    counts = {ct: sum(1 for v in labels.values() if v == ct) for ct in CALL_TYPES}
    print(f"{file_label}/{recording_label}: saved {len(labels)} label(s) "
          f"from file. Totals: {counts}")
    return data