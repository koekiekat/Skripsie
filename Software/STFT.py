import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

plt.rcParams['figure.figsize'] = [10, 5]

def resample_audio(audio_segments, f_s, fs_new):
    up = 1
    down = int(f_s / fs_new)
    audio_resampled = signal.resample_poly(audio_segments, up, down)
    return audio_resampled

def stft_calculation(audio_resampled, f_s, fs_new):
    framelength = 256
    noverlap = int(framelength * 0.8)
    window = "hamming"

    f, t, Zxx = signal.stft(audio_resampled, fs=fs_new, nperseg=framelength, noverlap=noverlap, window=window)
    return f, t, Zxx

def plot_spectrogram(f, t, Zxx, fs_new):
    plt.pcolormesh(t, f, np.log(np.abs(Zxx + 1e-16)), vmin = 0, vmax = np.max(np.log(np.abs(Zxx + 1e-16))));
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram of Audio Signal")
    plt.ylim(0, fs_new/2)