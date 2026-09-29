import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


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

def resample_audio(audio_segments, f_s, fs_new):
    up = 1
    down = int(f_s / fs_new)
    audio_resampled = signal.resample_poly(audio_segments, up, down)
    return audio_resampled

def feature_axes(feat, extractor, fs_new=1000):
    p = extractor.params
    nperseg = int(fs_new * p["frame_dur"])
    noverlap = int(nperseg * p["overlap"])
    hop = nperseg - noverlap

    n_freq, n_frames = feat.shape
    f = np.fft.rfftfreq(nperseg, d=1 / fs_new)   # length nperseg//2 + 1 == n_freq
    t = np.arange(n_frames) * hop / fs_new       # scipy's default boundary padding puts frame 0 at t = 0
    return f, t

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