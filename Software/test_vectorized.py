import soundfile as sf
import numpy as np
from stft import resample_audio, stft_calculation
from dtw import batched_dtw_costs_for_template, dtw_calc, batched_dtw_costs_for_template
from background_functions import read_audio_file
import json
from scipy import signal

MIN_STFT_DURATION = 0.128  # seconds -- matches framelength in stft.py's short_time_calc


def read_audio_chunk(audio_file, start_hour, duration_hour = 1.0, fs_new = 1000):
    start_sec = start_hour * 3600
    duration_sec = duration_hour * 3600
    with sf.SoundFile(audio_file) as f:
        f_s = f.samplerate
        f.seek(int(start_sec * f_s))
        audio_array = f.read(int(duration_sec * f_s), dtype="int16")
    resampled = resample_audio(audio_array, f_s, fs_new)

    return fs_new, resampled

def compute_template_dtw_costs(st_stfts, mt_stfts, bt_stfts, n_templates, audio_segment, fs_new=1000):
    """
    Load a saved templates JSON file, take the first `n_templates` entries,
    and compute the DTW cost of every template against every other template
    (all unique pairs, no self-comparisons, no duplicate reverse pairs since
    dtw_calc is symmetric).

    Returns:
        costs: 1D list of DTW costs, one per pair
        pairs: list of (i, j) index pairs (positions within the first
               n_templates entries) matching each entry in `costs`
    """

    #will store all DTW costs
    costs_st = []
    costs_mt = []
    costs_bt = []

    #calculate and save all DTW costs
    for i in range(len(st_stfts)):
        costs_st.append(dtw_calc(st_stfts[i], audio_segment))
    for i in range(len(mt_stfts)):
        costs_mt.append(dtw_calc(mt_stfts[i], audio_segment))
    for i in range(len(bt_stfts)):
        costs_bt.append(dtw_calc(bt_stfts[i], audio_segment))

    return costs_st, costs_mt, costs_bt

def dtw_costs_vectorized(st_stfts, mt_stfts, bt_stfts, n_templates, audio_segment):
    # audio_segment here is Zxx_window: (n_windows, n_freq, n_frames_per_window)

    # For each call type, compute costs of every template against every window
    #print("ST")
    st_costs = [batched_dtw_costs_for_template(t, audio_segment) for t in st_stfts]
    #print("MT")
    mt_costs = [batched_dtw_costs_for_template(t, audio_segment) for t in mt_stfts]
    #print("BT")
    bt_costs = [batched_dtw_costs_for_template(t, audio_segment) for t in bt_stfts]

    return st_costs, mt_costs, bt_costs

def load_template_segment(template, fs_new=1000):
    """
    Given one saved template/background entry ({start_time, end_time,
    wav_path, ...}), reload its audio from disk and compute its STFT.

    Pads short segments up to the minimum length required by the STFT
    window before transforming, so every spectrogram has the same number
    of frequency bins regardless of the original segment's duration.
    """
    #Extract template
    f_s, x = read_audio_file(template["wav_path"])
    start_sample = int(template["start_time"] * f_s)
    end_sample = int(template["end_time"] * f_s)
    segment = x[start_sample:end_sample]

    resampled = resample_audio(segment, f_s, fs_new)

    #Ensures window meets minimum length reqs
    min_len = int(fs_new * MIN_STFT_DURATION)
    if len(resampled) < min_len:
        resampled = np.pad(resampled, (0, min_len - len(resampled)), mode="constant")

    #Calculate STFT
    _, _, Zxx = stft_calculation(resampled, f_s, fs_new)
    return Zxx

def batch_stft_windows(audio_array, window_len, step_len, fs_new, framelength, noverlap):
    # (n_windows, window_len) view, no copying
    all_windows = np.lib.stride_tricks.sliding_window_view(audio_array, window_len)[::step_len]
    f, t, Zxx_batch = signal.stft(
        all_windows, fs=fs_new, nperseg=framelength, noverlap=noverlap,
        window="hamming", axis=-1
    )
    return f, t, Zxx_batch  # shape: (n_windows, n_freq_bins, n_time_frames)

def load_threshold(file_label, call_type, results_dir):
    """Load a previously saved threshold for this call type, or None if missing."""
    path = results_dir / f"{file_label}_{call_type}_threshold.json"
    if not path.exists():
        print(f"No saved threshold found for '{call_type}' ({file_label}).")
        return None
    with open(path) as f:
        return json.load(f)