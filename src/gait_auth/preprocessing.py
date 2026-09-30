"""Preprocessing for smartphone inertial gait windows."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from .config import PreprocessConfig

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


@dataclass
class Normalizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> "Normalizer":
        if windows.ndim != 3 or windows.shape[2] != len(CHANNELS):
            raise ValueError("windows must have shape (n_windows, n_samples, 6)")
        self.mean_ = windows.reshape(-1, windows.shape[-1]).mean(axis=0)
        self.scale_ = windows.reshape(-1, windows.shape[-1]).std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("normalizer must be fitted before transform")
        return (windows - self.mean_) / self.scale_

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)


def _lowpass(values: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if len(values) < max(12, config.filter_order * 4):
        return values.copy()
    nyquist = config.sampling_rate_hz / 2.0
    cutoff = min(config.lowpass_cutoff_hz / nyquist, 0.99)
    b, a = butter(config.filter_order, cutoff, btype="low")
    return filtfilt(b, a, values, axis=0)


def clean_and_filter(frame: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """Validate, sort, filter, and add orientation-reduced magnitudes."""
    required = {"subject_id", "activity", "timestamp", *CHANNELS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = frame.copy()
    for channel in CHANNELS:
        out[channel] = pd.to_numeric(out[channel], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=CHANNELS).sort_values("timestamp")
    if out.empty:
        return out.reset_index(drop=True)
    values = _lowpass(out[CHANNELS].to_numpy(dtype=float), config)
    out.loc[:, CHANNELS] = values
    out["accel_magnitude"] = np.linalg.norm(values[:, :3], axis=1)
    out["gyro_magnitude"] = np.linalg.norm(values[:, 3:], axis=1)
    return out.reset_index(drop=True)


def segment_windows(frame: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Segment a cleaned frame into overlapping six-channel windows.

    Returns ``(windows, labels)``. Windows shorter than the configured length
    are not padded, preventing artificial gait observations.
    """
    if frame.empty:
        return np.empty((0, config.window_samples, 6)), np.empty((0,), dtype=int)
    n = len(frame)
    size, stride = config.window_samples, config.stride_samples
    windows, labels = [], []
    for start in range(0, max(0, n - size + 1), stride):
        chunk = frame.iloc[start:start + size]
        if len(chunk) != size:
            continue
        valid = chunk[CHANNELS].notna().all(axis=1).mean()
        if valid < config.min_valid_fraction:
            continue
        windows.append(chunk[CHANNELS].to_numpy(dtype=np.float32))
        labels.append(int(chunk["subject_id"].mode().iloc[0]))
    if not windows:
        return np.empty((0, size, 6), dtype=np.float32), np.empty((0,), dtype=int)
    return np.stack(windows), np.asarray(labels)
