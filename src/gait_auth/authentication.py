"""Enrollment and verification services."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import numpy as np
from .config import ConfidenceThresholds, GaitConfidence
from .features import extract_features


@dataclass
class VerificationResult:
    claimed_subject: int
    predicted_subject: int
    score: float
    confidence: str
    accepted: bool

    def to_dict(self) -> dict:
        return asdict(self)


class GaitAuthenticator:
    def __init__(self, model, feature_mode: str = "classical", sampling_rate_hz: float = 20.0, thresholds: ConfidenceThresholds | None = None):
        self.model = model
        self.feature_mode = feature_mode
        self.sampling_rate_hz = sampling_rate_hz
        self.thresholds = thresholds or ConfidenceThresholds()
        self.enrolled_subjects: set[int] = set()

    def _represent(self, windows: np.ndarray) -> np.ndarray:
        if self.feature_mode == "classical":
            return extract_features(windows, self.sampling_rate_hz)
        if self.feature_mode == "raw":
            return windows
        raise ValueError("feature_mode must be classical or raw")

    def enroll(self, windows: np.ndarray, subject_ids: np.ndarray) -> None:
        if len(windows) == 0 or len(windows) != len(subject_ids):
            raise ValueError("enrollment requires non-empty windows and matching subject IDs")
        self.model.fit(self._represent(windows), subject_ids)
        self.enrolled_subjects.update(int(s) for s in np.unique(subject_ids))

    def verify(self, windows: np.ndarray, claimed_subject: int) -> list[VerificationResult]:
        if len(windows) == 0:
            raise ValueError("verification requires at least one window")
        if int(claimed_subject) not in self.enrolled_subjects:
            raise ValueError("claimed subject is not enrolled")
        prediction = self.model.predict(self._represent(windows))
        result = []
        for label, score in zip(prediction.labels, prediction.confidence):
            score = float(score)
            band = self.thresholds.band(score)
            result.append(VerificationResult(int(claimed_subject), int(label), score, band.value, int(label) == int(claimed_subject)))
        return result

    def save_metadata(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"feature_mode": self.feature_mode, "sampling_rate_hz": self.sampling_rate_hz, "enrolled_subjects": sorted(self.enrolled_subjects)}, handle, indent=2)
