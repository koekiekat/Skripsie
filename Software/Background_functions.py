import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]

def start_end_times(text):
    df = pd.read_csv(text, sep="\t") #tabs are used to seperate columns"
    pad = 0.1
    start_t = df["Begin Time (s)"].values - pad
    end_t = df["End Time (s)"].values + pad
    return df, start_t, end_t

def read_audio_file(fn):
    f_s, x = wavfile.read(fn)
    return f_s, x

def split_audio_segments(x, f_s, start_t, end_t):
    audio_segments = []
    for call_no in range(len(start_t)):
        start_sample = int(start_t[call_no] * f_s)
        end_sample = int(end_t[call_no] * f_s)
        audio_segment = x[start_sample:end_sample]
        audio_segments.append(audio_segment)
    return audio_segments

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

def call_length_stats(df):
    diff = df["End Time (s)"].values - df["Begin Time (s)"].values
    average_call_length = np.mean(diff)
    longest_call_length = np.max(diff)
    shortest_call_length = np.min(diff)
    return average_call_length, longest_call_length, shortest_call_length

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)