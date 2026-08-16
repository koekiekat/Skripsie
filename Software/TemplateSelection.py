import ipywidgets as widgets
from IPython.display import display
import numpy as np
import json
from pathlib import Path

from Background_functions import read_audio_file, split_audio_segments, start_end_times
from STFT import short_time_calc, plot_spectrogram
from DFT import dtw_calc


def load_file_calls(wav_path, text_path):
    f_s, x = read_audio_file(wav_path)
    _, start_t, end_t = start_end_times(text_path)
    segments = split_audio_segments(x, f_s, start_t, end_t)
    return f_s, x, start_t, end_t, segments


def select_calls_interactive(segments, f_s, start_t, audio_template,
                              expected_count, n_templates, n_calibration,
                              call_type="single_tone", file_label="",
                              exclude_idx=None):
    exclude_idx = set(exclude_idx or [])
    f_temp, t_temp, Zxx_temp, fs_new = short_time_calc(audio_template, f_s)

    num_calls = len(segments)
    stft_cost = np.zeros(num_calls)
    for i in range(num_calls):
        _, _, Zxx_seg, _ = short_time_calc(segments[i], f_s)
        stft_cost[i] = dtw_calc(Zxx_temp, Zxx_seg)

    sorted_idx = np.argsort(stft_cost)
    sorted_idx = np.array([i for i in sorted_idx if i not in exclude_idx])
    candidate_idx = sorted_idx[:expected_count]

    n_total = n_templates + n_calibration
    approved_idx = []
    state = {"pos": 0}
    results = {"template_idx": [], "calibration_idx": [],
               "call_type": call_type, "file_label": file_label,
               "stft_cost": stft_cost, "done": False}

    out = widgets.Output()
    status = widgets.Label()
    btn_yes = widgets.Button(description="Keep (y)", button_style="success")
    btn_no = widgets.Button(description="Reject (n)", button_style="danger")
    btn_stop = widgets.Button(description="Stop / Finish", button_style="warning")
    display(widgets.HTML(f"<b>{file_label} — {call_type}</b>"), status,
            widgets.HBox([btn_yes, btn_no, btn_stop]), out)

    def show_next():
        out.clear_output(wait=True)
        if len(approved_idx) >= n_total or state["pos"] >= len(candidate_idx):
            finalize()
            return
        idx = int(candidate_idx[state["pos"]])
        with out:
            f_sig, t_sig, Zxx_sig, fsn = short_time_calc(segments[idx], f_s)
            plot_spectrogram(f_sig, t_sig, Zxx_sig, fsn)
            print(f"rank {state['pos']}/{len(candidate_idx)-1} | "
                  f"cost {stft_cost[idx]:.4f} | index {idx} | start_t {start_t[idx]:.2f}")
        status.value = f"Approved: {len(approved_idx)}/{n_total}"

    def finalize():
        slots = np.unique(np.linspace(0, len(approved_idx) - 1,
                           min(n_templates, len(approved_idx))).round().astype(int)) \
                 if approved_idx else np.array([], dtype=int)
        results["template_idx"] = [approved_idx[i] for i in slots]
        results["calibration_idx"] = [approved_idx[i] for i in range(len(approved_idx))
                                       if i not in slots]
        results["done"] = True
        out.clear_output(wait=True)
        with out:
            print(f"Done. Templates: {len(results['template_idx'])}, "
                  f"Calibration: {len(results['calibration_idx'])}")
        btn_yes.disabled = btn_no.disabled = btn_stop.disabled = True

    def on_yes(b):
        approved_idx.append(int(candidate_idx[state["pos"]]))
        state["pos"] += 1
        show_next()

    def on_no(b):
        state["pos"] += 1
        show_next()

    def on_stop(b):
        finalize()

    btn_yes.on_click(on_yes)
    btn_no.on_click(on_no)
    btn_stop.on_click(on_stop)
    show_next()
    return results


def package_results(res, start_t, end_t):
    tmpl = res["template_idx"]
    calib = res["calibration_idx"]
    return {
        "call_type": res["call_type"], "file": res["file_label"],
        "template_idx": tmpl, "template_start_t": start_t[tmpl], "template_end_t": end_t[tmpl],
        "calibration_idx": calib, "calibration_start_t": start_t[calib], "calibration_end_t": end_t[calib],
    }


def save_results(packaged, results_dir):
    filename = f"{packaged['file']}_{packaged['call_type']}.json"
    path = results_dir / filename
    to_save = {
        "call_type": packaged["call_type"],
        "file": packaged["file"],
        "template_idx": [int(i) for i in packaged["template_idx"]],
        "template_start_t": [float(t) for t in packaged["template_start_t"]],
        "template_end_t": [float(t) for t in packaged["template_end_t"]],
        "calibration_idx": [int(i) for i in packaged["calibration_idx"]],
        "calibration_start_t": [float(t) for t in packaged["calibration_start_t"]],
        "calibration_end_t": [float(t) for t in packaged["calibration_end_t"]],
    }
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Saved: {path}")


def load_results(file_label, call_type, results_dir):
    path = results_dir / f"{file_label}_{call_type}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    data["template_start_t"] = np.array(data["template_start_t"])
    data["template_end_t"] = np.array(data["template_end_t"])
    data["calibration_start_t"] = np.array(data["calibration_start_t"])
    data["calibration_end_t"] = np.array(data["calibration_end_t"])
    return data


def run_call_selection(file_label, call_type, segments, f_s, start_t,
                        template_idx, expected_count, n_templates, n_calibration,
                        results_dir):
    existing = load_results(file_label, call_type, results_dir)
    if existing is not None:
        print(f"Loaded existing {call_type} selections for {file_label}, skipping interactive step.")
        return {"mode": "loaded", "packaged": existing}

    audio_template = segments[template_idx]
    results = select_calls_interactive(
        segments, f_s, start_t, audio_template,
        expected_count=expected_count, n_templates=n_templates, n_calibration=n_calibration,
        call_type=call_type, file_label=file_label,
        exclude_idx=[template_idx]
    )
    return {"mode": "interactive", "results": results}


def finalize_call_selection(state, start_t, end_t, all_results, results_dir):
    if state["mode"] == "loaded":
        packaged = state["packaged"]
        if not any(r["call_type"] == packaged["call_type"] and r["file"] == packaged["file"]
                   for r in all_results):
            all_results.append(packaged)
        return packaged

    results = state["results"]
    if not results.get("done"):
        print("Not done clicking yet — finish the interactive selection above, then rerun this cell.")
        return None

    packaged = package_results(results, start_t, end_t)
    save_results(packaged, results_dir)
    if not any(r["call_type"] == packaged["call_type"] and r["file"] == packaged["file"]
               for r in all_results):
        all_results.append(packaged)
    else:
        print("Already in all_results, skipping duplicate append.")
    return packaged


def clear_call_selection(file_label, call_type, all_results, results_dir, confirm=False):
    if not confirm:
        print(f"This will permanently delete the saved selection for "
              f"{file_label} / {call_type}. Call again with confirm=True to proceed.")
        return

    path = results_dir / f"{file_label}_{call_type}.json"
    if path.exists():
        path.unlink()
        print(f"Deleted saved file: {path}")
    else:
        print(f"No saved file found for {file_label} / {call_type}.")

    before = len(all_results)
    all_results[:] = [r for r in all_results
                       if not (r["call_type"] == call_type and r["file"] == file_label)]
    if before - len(all_results):
        print(f"Removed {before - len(all_results)} entry(ies) from all_results.")
    else:
        print("No matching entry found in all_results.")