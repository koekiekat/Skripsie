import ipywidgets as widgets
from IPython.display import display
from stft import short_time_calc, plot_spectrogram

def review_calls_with_labels(segments, f_s, start_t, labels, recording_label,
                              start_idx=0):
    """
    Step through every call in `segments` one at a time, showing its
    spectrogram alongside its already-known label (from labels_from_dataframe).
    Click 'Next' to advance. Purely for viewing/QA -- nothing is saved.

    labels: {index: call_type} dict, e.g. from labels_from_dataframe
    start_idx: call index to start from (handy if you want to resume
        partway through without re-viewing everything from the start)
    """
    fs_new = 1000
    num_calls = len(segments)
    order = list(range(start_idx, num_calls))

    state = {"pos": 0}
    out = widgets.Output()
    status = widgets.Label()
    btn_next = widgets.Button(description="Next", button_style="info")
    btn_stop = widgets.Button(description="Stop", button_style="warning")

    display(
        widgets.HTML(f"<b>{recording_label} — reviewing labeled calls</b>"),
        status,
        widgets.HBox([btn_next, btn_stop]),
        out,
    )

    def show_current():
        out.clear_output(wait=True)
        if state["pos"] >= len(order):
            with out:
                print("Done reviewing all calls.")
            btn_next.disabled = True
            return
        idx = order[state["pos"]]
        label = labels.get(idx, "UNLABELED")
        with out:
            f_sig, t_sig, Zxx_sig, fsn = short_time_calc(segments[idx], f_s, fs_new)
            plot_spectrogram(f_sig, t_sig, Zxx_sig, fsn)
            print(f"call {state['pos'] + 1}/{len(order)} | index {idx} | "
                  f"start_t {start_t[idx]:.2f} | label: {label}")
        status.value = f"Remaining: {len(order) - state['pos'] - 1}"

    def on_next(b):
        state["pos"] += 1
        show_current()

    def on_stop(b):
        out.clear_output(wait=True)
        with out:
            print(f"Stopped at index {order[state['pos']]} "
                  f"(pos {state['pos']}/{len(order)}). "
                  f"Pass start_idx={order[state['pos']]} to resume here.")
        btn_next.disabled = True

    btn_next.on_click(on_next)
    btn_stop.on_click(on_stop)

    show_current()