"""Configuration and domain enums for the gait authentication prototype."""
from dataclasses import dataclass
from enum import Enum


class GaitConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TransactionRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    STEP_UP_AUTHENTICATION = "STEP_UP_AUTHENTICATION"
    DENY = "DENY"


@dataclass(frozen=True)
class PreprocessConfig:
    sampling_rate_hz: float = 20.0
    lowpass_cutoff_hz: float = 5.0
    filter_order: int = 3
    window_seconds: float = 10.0
    stride_seconds: float = 5.0
    min_valid_fraction: float = 0.8

    @property
    def window_samples(self) -> int:
        return max(2, round(self.window_seconds * self.sampling_rate_hz))

    @property
    def stride_samples(self) -> int:
        return max(1, round(self.stride_seconds * self.sampling_rate_hz))


@dataclass(frozen=True)
class ConfidenceThresholds:
    medium: float = 0.60
    high: float = 0.85

    def band(self, score: float) -> GaitConfidence:
        if not 0.0 <= score <= 1.0:
            raise ValueError("confidence score must be between 0 and 1")
        if score >= self.high:
            return GaitConfidence.HIGH
        if score >= self.medium:
            return GaitConfidence.MEDIUM
        return GaitConfidence.LOW
