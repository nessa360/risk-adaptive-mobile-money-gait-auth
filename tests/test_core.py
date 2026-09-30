import numpy as np
import pandas as pd
import pytest

from gait_auth.config import ConfidenceThresholds, Decision, GaitConfidence, TransactionRisk
from gait_auth.data import read_wisdm_file, subject_split
from gait_auth.features import FEATURE_NAMES, extract_features
from gait_auth.preprocessing import clean_and_filter, segment_windows, PreprocessConfig
from gait_auth.models import ClassicalModel
from gait_auth.authentication import GaitAuthenticator
from gait_auth.risk_engine import classify_transaction_risk, decide


def windows(n=12, samples=200):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, samples, 6)).astype(np.float32)
    x[:, :, 0] += np.sin(np.arange(samples) / 5.0)
    return x


def test_wisdm_loader_discards_invalid_rows(tmp_path):
    path = tmp_path / "data_1600_accel_phone.txt"
    path.write_text("1600,A,1,1,2,3;\ninvalid\n1600,A,2,bad,2,3;\n")
    frame = read_wisdm_file(path, "accel")
    assert len(frame) == 1
    assert frame.attrs["invalid_rows"] == 2


def test_preprocessing_filters_and_segments():
    n = 240
    frame = pd.DataFrame({"subject_id": 1, "activity": "A", "timestamp": np.arange(n),
        "ax": np.sin(np.arange(n)/4), "ay": 0.2, "az": 9.8,
        "gx": 0.1, "gy": 0.2, "gz": 0.3})
    cleaned = clean_and_filter(frame)
    x, y = segment_windows(cleaned, PreprocessConfig(window_seconds=5, stride_seconds=2.5))
    assert x.ndim == 3 and x.shape[2] == 6
    assert len(x) == len(y) and len(x) > 0


def test_missing_data_is_removed_not_fabricated():
    frame = pd.DataFrame({"subject_id": [1, 1], "activity": ["A", "A"], "timestamp": [1, 2],
        "ax": [1, np.nan], "ay": [1, 1], "az": [1, 1], "gx": [1, 1], "gy": [1, 1], "gz": [1, 1]})
    assert len(clean_and_filter(frame)) == 1


def test_features_have_expected_dimension():
    result = extract_features(windows(3))
    assert result.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(result).all()


def test_subject_split_is_disjoint():
    split = subject_split(range(10), seed=2)
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_classical_model_prediction():
    x = np.vstack([extract_features(windows(6)[:3]), extract_features(windows(6)[3:])])
    y = np.array([1, 1, 1, 2, 2, 2])
    model = ClassicalModel("svm").fit(x, y)
    prediction = model.predict(x)
    assert len(prediction.labels) == len(y)
    assert np.all((prediction.confidence >= 0) & (prediction.confidence <= 1))


def test_authentication_workflow():
    x = windows(8)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    auth = GaitAuthenticator(ClassicalModel("random_forest"))
    auth.enroll(x, y)
    results = auth.verify(x[:2], 1)
    assert len(results) == 2
    assert all(0 <= r.score <= 1 for r in results)
    with pytest.raises(ValueError):
        auth.verify(x[:1], 99)


def test_confidence_boundaries():
    t = ConfidenceThresholds(medium=0.6, high=0.85)
    assert t.band(0.85) == GaitConfidence.HIGH
    assert t.band(0.6) == GaitConfidence.MEDIUM
    assert t.band(0.5999) == GaitConfidence.LOW
    with pytest.raises(ValueError):
        t.band(1.1)


def test_policy_all_cells():
    assert decide("high", "low") == Decision.ALLOW
    assert decide("high", "medium") == Decision.ALLOW
    assert decide("high", "high") == Decision.STEP_UP_AUTHENTICATION
    assert decide("medium", "low") == Decision.ALLOW
    assert decide("medium", "medium") == Decision.STEP_UP_AUTHENTICATION
    assert decide("medium", "high") == Decision.STEP_UP_AUTHENTICATION
    assert decide("low", "low") == Decision.STEP_UP_AUTHENTICATION
    assert decide("low", "medium") == Decision.STEP_UP_AUTHENTICATION
    assert decide("low", "high") == Decision.DENY


def test_risk_classification_and_invalid_amount():
    assert classify_transaction_risk(0) == TransactionRisk.LOW
    assert classify_transaction_risk(1500) == TransactionRisk.HIGH
    with pytest.raises(ValueError):
        classify_transaction_risk(-1)
