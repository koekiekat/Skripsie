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

def call_length_stats(df):
    diff = df["End Time (s)"].values - df["Begin Time (s)"].values
    average_call_length = np.mean(diff)
    longest_call_length = np.max(diff)
    shortest_call_length = np.min(diff)
    return average_call_length, longest_call_length, shortest_call_length