import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt 
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from stft import resample_audio
from dtw import dtw_calc_new


@dataclass(frozen=True)
class FeatureExtractor:
    name: str                                   #STFT / MFCC 
    fn: Callable[[np.ndarray, int], np.ndarray] #per segment feature extraction function
    min_duration: float                         #shortest audio duration
    metric: str = "cosine"                      #metric
    params: dict = field(default_factory=dict)  #store extractor configuration
    batch_fn: Optional[Callable] = None         # (n_windows, window_len), fs -> (n_windows, n_feat, n_frames)

    #allows for extractor(audio, fs) instead of extractor.fn(audio,fs)
    def __call__(self, audio, fs):
        return self.fn(audio, fs)

    #if a fast vectorized batch function is supplied you must use it
    def batch(self, 
              windows, 
              fs
              ):
        """(n_windows, window_len) -> (n_windows, n_features, n_frames)"""
        if self.batch_fn is not None:
            return self.batch_fn(windows, fs)
        return np.stack([self.fn(w, fs) for w in windows])   # slow fallback

def make_stft_extractor(frame_dur=0.128, overlap=0.75, window="hamming"):
    def fn(audio, fs):
        #number of samples per STFT segment
        nperseg = int(fs * frame_dur) 
        #Compute STFT
        _, _, Zxx = signal.stft(audio, 
                                fs=fs, 
                                nperseg=nperseg,
                                noverlap=int(nperseg * overlap), 
                                window=window
                                )
        #return real STFT magnitudes instead of complex magnitudes
        return np.abs(Zxx) #(n_freq_bins, n_frames)

    def batch_fn(windows, fs):
        nperseg = int(fs * frame_dur)
        #axis = -1: compute STFT along last axis independently for each row
        _, _, Zxx = signal.stft(windows,
                                fs=fs, 
                                nperseg=nperseg,
                                noverlap=int(nperseg * overlap), 
                                window=window, 
                                axis=-1
                                )
        return np.abs(Zxx) #(n_windows, n_freq_bins, n_frames)

    return FeatureExtractor("stft", 
                            fn, 
                            min_duration=frame_dur, 
                            metric="cosine",
                            params=dict(frame_dur=frame_dur, 
                                        overlap=overlap, 
                                        window=window),
                            batch_fn=batch_fn
                            )

def make_mfcc_extractor(n_mfcc=13, n_mels=13, frame_dur=0.128, overlap=0.75, drop_c0 = True, metric = "euclidean"):                       # one scalar scale

    def normalize_rows(feat, eps=1e-9):
        mean = feat.mean(axis=-1, keepdims=True)
        std = feat.std(axis=-1, keepdims=True)
        return (feat - mean) / (std + eps)

    def safe_delta(m, order=1, max_width=9):
        n_frames = m.shape[-1]
        width = min(max_width, n_frames if n_frames % 2 == 1 else n_frames - 1)
        width = max(width, 3)
        return librosa.feature.delta(m, order=order, width=width)

    def fn(audio, fs):
        n_fft = int(fs * frame_dur)
        hop = int(n_fft * (1 - overlap))
        S = librosa.feature.melspectrogram(y=audio.astype(np.float32), 
                                           sr=fs, 
                                           n_fft=n_fft, 
                                           hop_length=hop,
                                           window="hamming", 
                                           n_mels=n_mels
                                           )
        m = librosa.feature.mfcc(S=librosa.power_to_db(S, top_db=None), 
                                 n_mfcc=n_mfcc + int(drop_c0)
                                 )
        m = m[1:] if drop_c0 else m 

        #delta = librosa.feature.delta(m, order=1)
        #delta = safe_delta(m, order=1)
        #m_d_1 = np.concatenate([m, delta], axis=0)
        return m

    def batch_windows(audio, fs):
            n_fft = int(fs * frame_dur)
            hop = int(n_fft * (1 - overlap))
            S = librosa.feature.melspectrogram(y=audio.astype(np.float32), 
                                               sr=fs, 
                                               n_fft=n_fft, 
                                               hop_length=hop,
                                               window="hamming", 
                                               n_mels=n_mels
                                               )
            m = librosa.feature.mfcc(S=librosa.power_to_db(S, top_db=None), 
                                     n_mfcc=n_mfcc + int(drop_c0)
                                    )
                   
            m = m[..., 1:, :] if drop_c0 else m      

            #delta = librosa.feature.delta(m, order=1)
            #delta = safe_delta(m, order=1)
            #m_d_1 = np.concatenate([m, delta], axis=-2)
            return m

    return FeatureExtractor("mfcc", 
                            fn, 
                            min_duration=frame_dur, 
                            metric=metric,
                            batch_fn=batch_windows,
                            params=dict(n_mfcc=n_mfcc, 
                                        n_mels=n_mels, 
                                        frame_dur=frame_dur,
                                        overlap=overlap, 
                                        drop_c0=drop_c0)
                            )

# ---- single shared loader (replaces all three copies) ----

def plot_log_mel(audio, fs, n_mels=13, frame_dur=0.128, overlap=0.75):
    n_fft = int(fs * frame_dur)
    hop = int(n_fft * (1 - overlap))
    S = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=fs,
                                        n_fft=n_fft, hop_length=hop,
                                        window="hamming", n_mels=n_mels)
    S_db = librosa.power_to_db(S, top_db=None)

    t = np.arange(S.shape[1]) * (hop / fs)
    mel_idx = np.arange(n_mels)

    plt.pcolormesh(t, mel_idx, S_db, shading="auto")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel band")
    plt.title("Log-Mel Spectrogram")
    plt.colorbar(label="dB")

def plot_mfcc_grid(feats_list, fs_new, hop_samples, title_prefix, n_cols=4):
    n = len(feats_list)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(feats_list):
        plot_mfcc(feat, fs_new, hop_samples, ax=axes[i])
        axes[i].set_title(f"{title_prefix} {i}")

    # turn off any unused subplot slots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_mfcc(m, fs_new, hop_samples, ax=None, normalise=False):
    """Plot an (n_coeffs, n_frames) MFCC matrix against real time."""
    img_data = m
    if normalise:  # display only: stops one large-magnitude row from washing out the rest
        img_data = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)

    n_coeffs, n_frames = m.shape
    t = np.arange(n_frames) * (hop_samples / fs_new)
    coeff_idx = np.arange(n_coeffs)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    img = ax.pcolormesh(t, coeff_idx, img_data, shading="auto")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC coefficient")

    if standalone:
        ax.set_title("MFCC")
        plt.colorbar(img, ax=ax, label="Coefficient value")

    return img

def plot_mfcc_pair(feat_a, feat_b, fs_new, hop_samples, metric="euclidean",
                   label_a="Template", label_b="Comparison"):
    """Plot two MFCC feature arrays side by side and show their DTW cost."""
    cost = dtw_calc_new(feat_a, feat_b, metric)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_mfcc(feat_a, fs_new, hop_samples, ax=axes[0])
    axes[0].set_title(label_a)

    plot_mfcc(feat_b, fs_new, hop_samples, ax=axes[1])
    axes[1].set_title(label_b)

    fig.suptitle(f"DTW cost = {cost:.4f}  (metric = {metric})")
    plt.tight_layout()
    plt.show()

    return cost

def load_segment_features(entry, extractor, fs_new=1000):

    with sf.SoundFile(entry["wav_path"]) as f:          # reads only the segment, not the whole WAV
        f_s = f.samplerate
        start, end = int(entry["start_time"] * f_s), int(entry["end_time"] * f_s)
        f.seek(start)
        segment = f.read(end - start, dtype="int16")

    audio = resample_audio(segment, f_s, fs_new)
    min_len = int(fs_new * extractor.min_duration)
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)), mode="constant")
    return extractor(audio, fs_new)

def load_features_from_json(path, extractor, n=None, fs_new=1000):

    with open(path) as f:
        entries = json.load(f)
    if n is not None and len(entries) < n:
        print(f"Warning: {path} has only {len(entries)} entries, using all instead of {n}.")
    return [load_segment_features(e, extractor, fs_new) for e in entries[:n]]