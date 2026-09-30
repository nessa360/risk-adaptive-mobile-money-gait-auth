"""Handcrafted features for classical gait classifiers."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks

FEATURE_NAMES = [
    "step_time_mean", "step_time_std", "cadence", "stride_variability",
    "peak_acceleration", "dominant_frequency", "spectral_energy",
    "accel_mean", "accel_std", "gyro_mean", "gyro_std",
]


def _one_window(window: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError("window must have shape (samples, 6)")
    accel_mag = np.linalg.norm(window[:, :3], axis=1)
    gyro_mag = np.linalg.norm(window[:, 3:], axis=1)
    centered = accel_mag - np.mean(accel_mag)
    distance = max(1, int(sampling_rate_hz * 0.25))
    peaks, _ = find_peaks(centered, distance=distance, prominence=max(np.std(centered) * 0.2, 1e-8))
    if len(peaks) >= 2:
        step_times = np.diff(peaks) / sampling_rate_hz
        step_mean = float(np.mean(step_times))
        step_std = float(np.std(step_times))
        cadence = float(60.0 / step_mean) if step_mean > 0 else 0.0
        variability = float(step_std / step_mean) if step_mean > 0 else 0.0
    else:
        step_mean = step_std = cadence = variability = 0.0
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate_hz)
    if len(spectrum) > 1:
        idx = int(np.argmax(spectrum[1:]) + 1)
        dominant = float(frequencies[idx])
        energy = float(np.sum(spectrum[1:]) / max(1, len(spectrum) - 1))
    else:
        dominant = energy = 0.0
    return np.asarray([
        step_mean, step_std, cadence, variability, float(np.max(accel_mag)),
        dominant, energy, float(np.mean(accel_mag)), float(np.std(accel_mag)),
        float(np.mean(gyro_mag)), float(np.std(gyro_mag)),
    ], dtype=np.float32)


def extract_features(windows: np.ndarray, sampling_rate_hz: float = 20.0) -> np.ndarray:
    """Extract one fixed-length feature vector per normalized or raw window."""
    if windows.ndim != 3 or windows.shape[2] != 6:
        raise ValueError("windows must have shape (n_windows, samples, 6)")
    if len(windows) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([_one_window(window, sampling_rate_hz) for window in windows])
