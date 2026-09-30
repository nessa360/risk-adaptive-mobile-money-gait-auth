"""Data loading utilities for WISDM-style raw inertial files.

The loader accepts rows of the form subject,activity,timestamp,x,y,z; and returns
validated phone accelerometer/gyroscope records. It never invents missing sensor
values; unmatched modalities remain missing and are handled explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np

COLUMNS = ["subject_id", "activity", "timestamp", "x", "y", "z"]


def read_wisdm_file(path: str | Path, sensor: str) -> pd.DataFrame:
    """Read one WISDM raw file and attach the sensor name.

    Invalid rows are discarded and counted in ``attrs['invalid_rows']``.
    """
    if sensor not in {"accel", "gyro"}:
        raise ValueError("sensor must be 'accel' or 'gyro'")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    invalid = 0
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip().rstrip(";")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            invalid += 1
            continue
        try:
            rows.append([int(parts[0]), parts[1], int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
        except (ValueError, TypeError):
            invalid += 1
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["sensor"] = sensor
    frame.attrs["invalid_rows"] = invalid
    return frame


def load_wisdm_subject(raw_root: str | Path, subject_id: int, activity: str = "A") -> pd.DataFrame:
    """Load and inner-align phone accelerometer and gyro data for one subject/activity.

    WISDM files are stored under ``phone/accel`` and ``phone/gyro``. Timestamps
    are retained in source units and aligned by nearest timestamp within a
    conservative tolerance inferred from the median sampling interval.
    """
    root = Path(raw_root)
    accel_path = next(root.glob(f"**/data_{subject_id}_accel_phone.txt"), None)
    gyro_path = next(root.glob(f"**/data_{subject_id}_gyro_phone.txt"), None)
    if accel_path is None or gyro_path is None:
        raise FileNotFoundError(f"phone accel/gyro files for subject {subject_id} not found below {root}")
    accel = read_wisdm_file(accel_path, "accel").query("activity == @activity").copy()
    gyro = read_wisdm_file(gyro_path, "gyro").query("activity == @activity").copy()
    if accel.empty or gyro.empty:
        return pd.DataFrame(columns=["subject_id", "activity", "timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
    accel = accel.rename(columns={"x": "ax", "y": "ay", "z": "az"}).drop(columns="sensor")
    gyro = gyro.rename(columns={"x": "gx", "y": "gy", "z": "gz"}).drop(columns="sensor")
    accel = accel.sort_values("timestamp")
    gyro = gyro.sort_values("timestamp")
    diffs = np.diff(accel["timestamp"].to_numpy())
    tolerance = int(np.nanmedian(diffs) * 0.75) if len(diffs) else 1
    tolerance = max(tolerance, 1)
    aligned = pd.merge_asof(accel, gyro, on="timestamp", by=["subject_id", "activity"], direction="nearest", tolerance=tolerance)
    aligned = aligned.dropna(subset=["ax", "ay", "az", "gx", "gy", "gz"]).reset_index(drop=True)
    return aligned


def subject_split(subject_ids: Iterable[int], train_fraction: float = 0.6, validation_fraction: float = 0.2, seed: int = 42) -> dict[str, list[int]]:
    """Create deterministic subject-disjoint train/validation/test partitions."""
    subjects = np.array(sorted(set(int(s) for s in subject_ids)))
    if len(subjects) < 3:
        raise ValueError("at least three subjects are required for disjoint splits")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must be positive and leave a non-empty test split")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_train = max(1, int(len(subjects) * train_fraction))
    n_val = max(1, int(len(subjects) * validation_fraction))
    return {"train": subjects[:n_train].tolist(), "validation": subjects[n_train:n_train+n_val].tolist(), "test": subjects[n_train+n_val:].tolist()}
