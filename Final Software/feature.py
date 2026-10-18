import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import librosa


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

def make_stft_extractor(frame_dur, overlap=0.75, window="hamming"):

    def fn(audio, fs):
        nperseg = int(fs * frame_dur)
        _, _, Zxx = signal.stft(audio, fs=fs, nperseg=nperseg,
                                noverlap=int(nperseg * overlap), window=window)
        return np.abs(Zxx)                 # (n_freq_kept, n_frames)

    return FeatureExtractor("stft", fn, min_duration = frame_dur, metric="cosine",
                            params=dict(frame_dur=frame_dur, overlap=overlap),)

def make_mfcc_extractor(frame_dur, n_mfcc=14, n_mels=26, overlap=0.75, drop_c0 = True, derv_1 = False, derv_2 = False):                       # one scalar scale

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
                                           n_mels=n_mels, fmin = 0.0, fmax = 500.0
                                           )
        m = librosa.feature.mfcc(S=librosa.power_to_db(S), 
                                 n_mfcc=n_mfcc + int(drop_c0)
                                 )
        m = m[1:] if drop_c0 else m 
        delta = safe_delta(m, order=1)
        m_d_1 = np.concatenate([m, delta], axis=0)
        delta_2 = safe_delta(m, order=2)
        m_d_2 = np.concatenate([m_d_1, delta_2], axis=0)

        if derv_1:
            return m_d_1
        elif derv_2:
            return m_d_2
        else:
            return m

    if derv_1:
        call = "mfcc_d_1"
    elif derv_2:
        call = "mfcc_d_2"
    else:
        call = "mfcc"

    return FeatureExtractor(call, 
                            fn, 
                            min_duration=frame_dur, 
                            metric="euclidean",
                            params=dict(n_mfcc=n_mfcc, 
                                        n_mels=n_mels, 
                                        frame_dur=frame_dur,
                                        overlap=overlap, 
                                        drop_c0=drop_c0)
                            )

def lpc_to_lpcc(a, n_ceps):
    """
    a: (order + 1, n_frames) from librosa.lpc, with a[0] == 1
       (convention: A(z) = 1 + a1 z^-1 + ... + ap z^-p)
    returns: (n_ceps, n_frames) cepstral coefficients c1..c_n_ceps
    """
    p = a.shape[0] - 1
    n_frames = a.shape[1]
    c = np.zeros((n_ceps + 1, n_frames))          # index 0 unused (gain term)

    for n in range(1, n_ceps + 1):
        if n <= p:
            acc = -a[n]
        else:
            acc = np.zeros(n_frames)
        for k in range(max(1, n - p), n):
            acc = acc - (k / n) * c[k] * a[n - k]
        c[n] = acc
    return c[1:]


def make_lpc_extractor(frame_dur, order=4, n_ceps=8, overlap=0.75):
    n_ceps = n_ceps or order          # often 1.0-1.5x the LPC order

    def fn(audio, fs):
        n_fft = int(fs * frame_dur)
        hop = max(1, int(n_fft * (1 - overlap)))

        audio = np.pad(audio.astype(np.float64), n_fft // 2, mode="reflect")  # centre frames
        frames = librosa.util.frame(audio, frame_length=n_fft, hop_length=hop)
        frames = frames * np.hamming(n_fft)[:, None]

        # tiny white-noise floor so silent/degenerate frames don't give NaN or huge poles
        frames = frames + 1e-6 * np.random.default_rng(0).standard_normal(frames.shape)

        a = librosa.lpc(frames, order=order, axis=0)       # (order + 1, n_frames)
        a = np.nan_to_num(a)
        c = lpc_to_lpcc(a, n_ceps)                         # (n_ceps, n_frames)
        return c

    return FeatureExtractor("lpcc",
                            fn,
                            min_duration=frame_dur,
                            metric="euclidean",
                            params=dict(order=order,
                                        n_ceps=n_ceps,
                                        frame_dur=frame_dur,
                                        overlap=overlap))

def feature_axes(feat, extractor, fs_new=1000):
    p = extractor.params
    nperseg = int(fs_new * p["frame_dur"])
    hop = nperseg - int(nperseg * p["overlap"])
    n_freq, n_frames = feat.shape
    f = np.fft.rfftfreq(nperseg, d=1 / fs_new)
    f = f[f >= p.get("f_min", 0.0)]              # keep axis in step with the sliced features
    t = np.arange(n_frames) * hop / fs_new
    return f, t

def load_segment_features(entry, extractor, fs_new):

    with sf.SoundFile(entry["wav_path"]) as f:          # reads only the segment, not the whole WAV
        f_s = f.samplerate
        start, end = int(entry["start_time"] * f_s), int(entry["end_time"] * f_s)

        min_samples = int((extractor.min_duration + 0.01)*f_s)
        if end - start < min_samples:
            centre = (start + end)//2
            start = max(0, min(centre - min_samples // 2, f.frames - min_samples))
            end = start + min_samples
        
        f.seek(start)
        segment = f.read(end - start, dtype="int16")

    #resample segment to 1kHz
    audio = resample_audio(segment, f_s, fs_new)
    # min_len = int(fs_new * extractor.min_duration)
    # if len(audio) < min_len:
    #     audio = np.pad(audio, (0, min_len - len(audio)), mode="constant")
    #     print(f"{len(audio)} < {min_len}")
    return extractor(audio, fs_new)

def load_features_from_json(path, extractor, n, fs_new):

    with open(path) as f:
        entries = json.load(f)
    if n is not None and len(entries) < n:
        print(f"Warning: {path} has only {len(entries)} entries, using all instead of {n}.")
    return [load_segment_features(e, extractor, fs_new) for e in entries[:n]]

def resample_audio(audio_segments, f_s, fs_new):
    up = 1
    down = int(f_s / fs_new)
    audio_resampled = signal.resample_poly(audio_segments, up, down)
    return audio_resampled

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

def batch_feature_windows(audio_array, window_len, step_len, fs, extractor):
    """window_len / step_len in SAMPLES. Returns (n_windows, n_features, n_frames)."""
    windows = np.lib.stride_tricks.sliding_window_view(audio_array, window_len)[::step_len]
    return extractor.batch(windows, fs)