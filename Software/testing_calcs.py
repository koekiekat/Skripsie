import soundfile as sf
import numpy as np
from stft import resample_audio, stft_calculation
from dtw import dtw_calc
from background_functions import read_audio_file
import json

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
    #error warning if too little templates
    '''if len(st_temps) < n_templates:
        print(f"Warning: file only has {len(st_temps)} template(s), "
              f"using all of them instead of {n_templates}.")

    if len(mt_temps) < n_templates:
        print(f"Warning: file only has {len(mt_temps)} template(s), "
              f"using all of them instead of {n_templates}.")

    if len(bt_temps) < n_templates:
        print(f"Warning: file only has {len(bt_temps)} template(s), "
              f"using all of them instead of {n_templates}.")'''

    #Only wokring with first n templates
    '''st_templates = st_temps[:n_templates]
    mt_templates = mt_temps[:n_templates]
    bt_templates = bt_temps[:n_templates]'''

    #Get STFTs of all templates
    '''st_stfts = [_load_template_segment(t, fs_new) for t in st_temps]
    mt_stfts = [_load_template_segment(t, fs_new) for t in mt_temps]
    bt_stfts = [_load_template_segment(t, fs_new) for t in bt_temps]'''

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

def split_into_windows(audio_array, window_len, step_len):
    """
    Slide a fixed-size window across audio_array with step_len between starts.
    Returns a list of (start_sample, end_sample, window_audio) tuples.
    """
    windows = []
    n = len(audio_array)
    start = 0
    while start + window_len <= n:
        end = start + window_len
        windows.append((start, end, audio_array[start:end]))
        start += step_len
    return windows

def load_threshold(file_label, call_type, results_dir):
    """Load a previously saved threshold for this call type, or None if missing."""
    path = results_dir / f"{file_label}_{call_type}_threshold.json"
    if not path.exists():
        print(f"No saved threshold found for '{call_type}' ({file_label}).")
        return None
    with open(path) as f:
        return json.load(f)