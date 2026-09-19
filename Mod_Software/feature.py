import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf
from scipy import signal

from stft import resample_audio


@dataclass(frozen=True)
class FeatureExtractor:
    name: str                                          # used in filenames / saved JSON
    fn: Callable[[np.ndarray, int], np.ndarray]        # (audio, fs) -> (n_features, n_frames)
    min_duration: float                                # seconds; shorter audio gets zero-padded
    metric: str = "cosine"                             # distance metric DTW should use
    params: dict = field(default_factory=dict)         # recorded alongside saved thresholds

    def __call__(self, audio, fs):
        return self.fn(audio, fs)

def make_stft_extractor(frame_dur=0.128, overlap=0.75, window="hamming"):
    def fn(audio, fs):
        nperseg = int(fs * frame_dur)
        _, _, Zxx = signal.stft(audio, fs=fs, nperseg=nperseg,
                                noverlap=int(nperseg * overlap), window=window)
        return np.abs(Zxx)          # magnitude here, NOT inside DTW

    return FeatureExtractor("stft", fn, min_duration=frame_dur, metric="cosine",
                            params=dict(frame_dur=frame_dur, overlap=overlap, window=window))

def make_mfcc_extractor(n_mfcc=13, n_mels=20, frame_dur=0.128, overlap=0.75,
                        fmin=0.0, fmax=None, drop_c0=True):
    import librosa

    def fn(audio, fs):
        n_fft = int(fs * frame_dur)
        hop = int(n_fft * (1 - overlap))
        m = librosa.feature.mfcc(
            y=audio.astype(np.float32), sr=fs,
            n_mfcc=n_mfcc + int(drop_c0),
            n_fft=n_fft, hop_length=hop, window="hamming",
            n_mels=n_mels, fmin=fmin, fmax=fmax,
        )
        return m[1:] if drop_c0 else m

    return FeatureExtractor("mfcc", fn, min_duration=frame_dur, metric="euclidean",
                            params=dict(n_mfcc=n_mfcc, n_mels=n_mels, frame_dur=frame_dur,
                                        overlap=overlap, fmin=fmin, fmax=fmax, drop_c0=drop_c0))

# ---- single shared loader (replaces all three copies) ----
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