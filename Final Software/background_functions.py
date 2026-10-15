import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

plt.rcParams['figure.figsize'] = [10, 5]

def read_audio_file(fn):
    f_s, x = wavfile.read(fn)
    return f_s, x

def start_end_times(text):
    df = pd.read_csv(text, sep="\t") #tabs are used to seperate columns"
    df = df.drop_duplicates(subset = ["Begin Time (s)", "End Time (s)"]).reset_index(drop=True)
    pad = 0.1
    start_t = df["Begin Time (s)"].values - pad
    end_t = df["End Time (s)"].values + pad
    return df, start_t, end_t

def split_audio_segments(x, f_s, start_t, end_t):
    audio_segments = []
    for call_no in range(len(start_t)):
        start_sample = int(start_t[call_no] * f_s)
        end_sample = int(end_t[call_no] * f_s)
        audio_segment = x[start_sample:end_sample]
        audio_segments.append(audio_segment)
    return audio_segments

def call_length_stats(df):
    diff = df["End Time (s)"].values - df["Begin Time (s)"].values
    average_call_length = np.mean(diff)
    longest_call_length = np.max(diff)
    shortest_call_length = np.min(diff)
    return average_call_length, longest_call_length, shortest_call_length

def short_time_calc(sig, fs, fs_new):
    resampled_sig = resample_audio(sig, fs, fs_new)

    f_seg, t_seg, Zxx_seg = stft_calculation(resampled_sig, fs, fs_new)

    return f_seg, t_seg, Zxx_seg, fs_new

def resample_audio(audio_segments, f_s, fs_new):
    up = 1
    down = int(f_s / fs_new)
    audio_resampled = signal.resample_poly(audio_segments, up, down)
    return audio_resampled

def stft_calculation(audio_resampled, f_s, fs_new):
    framelength = int(fs_new * 0.128)
    noverlap = int(framelength * 0.75)
    window = "hamming"

    f, t, Zxx = signal.stft(audio_resampled, fs=fs_new, nperseg=framelength, noverlap=noverlap, window=window)
    return f, t, Zxx

def plot_spectrogram(f, t, Zxx, fs_new):
    plt.pcolormesh(t, 
                   f, 
                   np.log(np.abs(Zxx + 1e-16)), 
                   vmin = 0, 
                   vmax = np.max(np.log(np.abs(Zxx + 1e-16)))
                   )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram of Audio Signal")
    plt.ylim(0, fs_new/2)

def read_audio_chunk(audio_file, start_hour, duration_hour, fs_new):
    start_sec = start_hour * 3600
    duration_sec = duration_hour * 3600
    with sf.SoundFile(audio_file) as f:
        f_s = f.samplerate
        f.seek(int(start_sec * f_s))
        audio_array = f.read(int(duration_sec * f_s), dtype="int16")
    resampled = resample_audio(audio_array, f_s, fs_new)

    return fs_new, resampled