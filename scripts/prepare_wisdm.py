"""Prepare subject-disjoint NPZ windows from an extracted WISDM archive.

The output is compatible with scripts/run_experiments.py. WISDM has no
controlled-condition, calendar-session, mimicry, or survey fields, so this
script deliberately exports only sensor windows and subject labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from gait_auth.config import PreprocessConfig
from gait_auth.data import load_wisdm_subject, subject_split
from gait_auth.preprocessing import clean_and_filter, segment_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Extracted WISDM root")
    parser.add_argument("--output", default="data/processed/wisdm_subject_disjoint.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0,
                        help="Use 10 seconds by default so evaluation windows do not overlap")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    files = sorted(raw_root.glob("**/data_*_accel_phone.txt"))
    subject_ids = []
    for path in files:
        try:
            subject_ids.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    subject_ids = sorted(set(subject_ids))
    if len(subject_ids) < 3:
        raise RuntimeError(f"Found only {len(subject_ids)} subject files below {raw_root}; need at least 3")

    split = subject_split(subject_ids, args.train_fraction, args.validation_fraction, args.seed)
    config = PreprocessConfig(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    windows_by_subject: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subject_id in subject_ids:
        try:
            frame = load_wisdm_subject(raw_root, subject_id, activity="A")
            cleaned = clean_and_filter(frame, config)
            windows, labels = segment_windows(cleaned, config)
            if len(windows):
                windows_by_subject[subject_id] = (windows, labels)
        except FileNotFoundError:
            continue

    def collect(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        chunks = [windows_by_subject[s] for s in ids if s in windows_by_subject]
        if not chunks:
            return np.empty((0, config.window_samples, 6), dtype=np.float32), np.empty((0,), dtype=int)
        return np.concatenate([x for x, _ in chunks]), np.concatenate([y for _, y in chunks])

    train_windows, train_labels = collect(split["train"])
    test_windows, test_labels = collect(split["test"])
    if len(train_windows) == 0 or len(test_windows) == 0:
        raise RuntimeError("No windows in train or test split; verify WISDM paths and walking activity code A")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_windows=train_windows,
        train_labels=train_labels,
        test_windows=test_windows,
        test_labels=test_labels,
        train_subjects=np.asarray(split["train"], dtype=int),
        validation_subjects=np.asarray(split["validation"], dtype=int),
        test_subjects=np.asarray(split["test"], dtype=int),
    )
    print(f"wrote {output}")
    print(f"subjects: train={split['train']} validation={split['validation']} test={split['test']}")
    print(f"windows: train={len(train_windows)} test={len(test_windows)} shape={train_windows.shape[1:]}")


if __name__ == "__main__":
    main()
