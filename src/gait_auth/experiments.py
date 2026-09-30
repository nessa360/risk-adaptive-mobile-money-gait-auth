"""Experiment runners and honest status reporting for E1-E6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .evaluation import classifier_metrics, verification_metrics


@dataclass
class ExperimentStatus:
    experiment: str
    status: str
    interpretation: str
    limitation: str = ""

    def to_dict(self):
        return asdict(self)


def save_json(payload, path: str | Path) -> None:
    """Save a Python object as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )


# ============================================================
# E1 — CLOSED-SET CLASSIFICATION BASELINE
# ============================================================

def run_e1(
    models: dict,
    x_test,
    y_test,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Run the closed-set classification baseline.

    This experiment is retained only as a comparison baseline.

    The development/train subjects and test subjects are subject-disjoint.
    Therefore, a conventional closed-set classifier is expected to perform
    poorly because the test identities were not present during training.
    """

    rows = []

    for name, model in models.items():
        prediction = model.predict(x_test)

        row = {
            "experiment": "E1_CLASSIFICATION_BASELINE",
            "model": name,
            **classifier_metrics(
                y_test,
                prediction.labels,
            ),
            "eer": np.nan,
            "far": np.nan,
            "frr": np.nan,
            "status": "MEASURED_CLASSIFICATION_ONLY",
        }

        rows.append(row)

    frame = pd.DataFrame(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output_dir / "e1_classification_baseline.csv",
        index=False,
    )

    return frame


# ============================================================
# E1 — VERIFICATION HELPERS
# ============================================================

def _cosine_scores(
    templates: dict[int, np.ndarray],
    windows: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate genuine and impostor cosine similarity scores.

    Each verification window is associated with a claimed subject.

    Genuine score:
        Similarity between the verification window and the claimed
        subject's enrolled template.

    Impostor scores:
        Similarity between the verification window and every other
        enrolled subject template.

    Scores are converted from cosine similarity [-1, 1] to [0, 1].
    """

    vectors = np.asarray(
        windows,
        dtype=float,
    )

    labels = np.asarray(labels)

    genuine_scores = []
    impostor_scores = []

    for vector, subject in zip(vectors, labels):

        subject_id = int(subject)

        norm = np.linalg.norm(vector)

        if norm == 0:
            continue

        if subject_id not in templates:
            continue

        normalized_vector = vector / norm

        # Genuine comparison.
        own_template = templates[subject_id]

        genuine_score = (
            np.dot(
                normalized_vector,
                own_template,
            )
            + 1.0
        ) / 2.0

        genuine_scores.append(
            float(genuine_score)
        )

        # Impostor comparisons.
        for other_subject, other_template in templates.items():

            if other_subject == subject_id:
                continue

            impostor_score = (
                np.dot(
                    normalized_vector,
                    other_template,
                )
                + 1.0
            ) / 2.0

            impostor_scores.append(
                float(impostor_score)
            )

    return (
        np.asarray(genuine_scores),
        np.asarray(impostor_scores),
    )


def _split_enrollment_verification(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split each subject's windows into disjoint sets.

    Alternating windows are assigned to the first and second sets.
    This guarantees that no individual window appears in both sets.

    The function is used both for:
        1. Development template/calibration separation.
        2. Enrollment/verification separation.
    """

    features = np.asarray(
        features,
        dtype=float,
    )

    labels = np.asarray(labels)

    first_x = []
    first_y = []

    second_x = []
    second_y = []

    for subject in sorted(np.unique(labels)):

        indices = np.flatnonzero(
            labels == subject
        )

        if len(indices) < 2:
            continue

        first_indices = indices[::2]
        second_indices = indices[1::2]

        if (
            len(first_indices) == 0
            or len(second_indices) == 0
        ):
            continue

        first_x.append(
            features[first_indices]
        )

        first_y.append(
            labels[first_indices]
        )

        second_x.append(
            features[second_indices]
        )

        second_y.append(
            labels[second_indices]
        )

    if not first_x:

        empty_features = np.empty(
            (
                0,
                features.shape[1],
            ),
            dtype=float,
        )

        empty_labels = np.empty(
            0,
            dtype=labels.dtype,
        )

        return (
            empty_features,
            empty_labels,
            empty_features,
            empty_labels,
        )

    return (
        np.concatenate(first_x),
        np.concatenate(first_y),
        np.concatenate(second_x),
        np.concatenate(second_y),
    )


def _templates(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Create one normalized template for each subject."""

    features = np.asarray(
        features,
        dtype=float,
    )

    labels = np.asarray(labels)

    templates = {}

    for subject in np.unique(labels):

        subject_features = features[
            labels == subject
        ]

        if len(subject_features) == 0:
            continue

        centroid = np.mean(
            subject_features,
            axis=0,
        )

        norm = np.linalg.norm(
            centroid
        )

        if norm > 0:
            templates[int(subject)] = (
                centroid / norm
            )

    return templates


def _eer_threshold(
    genuine: np.ndarray,
    impostor: np.ndarray,
) -> float:
    """Select an operating threshold close to equal error rate.

    The threshold is selected from ROC operating points where the
    false acceptance rate and false rejection rate are closest.
    """

    genuine = np.asarray(
        genuine,
        dtype=float,
    )

    impostor = np.asarray(
        impostor,
        dtype=float,
    )

    if (
        len(genuine) == 0
        or len(impostor) == 0
    ):
        return 0.5

    scores = np.concatenate(
        [
            genuine,
            impostor,
        ]
    )

    labels = np.concatenate(
        [
            np.ones(
                len(genuine)
            ),
            np.zeros(
                len(impostor)
            ),
        ]
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    fnr = 1.0 - tpr

    index = int(
        np.nanargmin(
            np.abs(
                fpr - fnr
            )
        )
    )

    threshold = float(
        thresholds[index]
    )

    return float(
        np.clip(
            threshold,
            0.0,
            1.0,
        )
    )


# ============================================================
# E1 — SUBJECT VERIFICATION
# ============================================================

def run_e1_verification(
    development_features: np.ndarray,
    development_labels: np.ndarray,
    enrollment_features: np.ndarray,
    enrollment_labels: np.ndarray,
    verification_features: np.ndarray,
    verification_labels: np.ndarray,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Evaluate subject authentication using disjoint verification data.

    Protocol:

    1. Development subjects are separated into:
       - template windows
       - threshold-calibration windows

    2. Development templates are created only from the template windows.

    3. The authentication threshold is calibrated using the independent
       development calibration windows.

    4. Enrollment subjects were not used during development.

    5. Enrollment windows are separate from verification windows.

    6. Final E1 metrics are calculated only on the verification windows.

    This prevents the previous implementation from calibrating the threshold
    using the same development windows that were used to construct the
    development templates.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # STEP 1 — Separate development template/calibration data
    # --------------------------------------------------------

    (
        development_template_x,
        development_template_y,
        development_calibration_x,
        development_calibration_y,
    ) = _split_enrollment_verification(
        development_features,
        development_labels,
    )

    # --------------------------------------------------------
    # STEP 2 — Build development templates
    # --------------------------------------------------------

    development_templates = _templates(
        development_template_x,
        development_template_y,
    )

    # --------------------------------------------------------
    # STEP 3 — Calibrate threshold on independent windows
    # --------------------------------------------------------

    development_genuine_scores, development_impostor_scores = (
        _cosine_scores(
            development_templates,
            development_calibration_x,
            development_calibration_y,
        )
    )

    threshold = _eer_threshold(
        development_genuine_scores,
        development_impostor_scores,
    )

    # --------------------------------------------------------
    # STEP 4 — Build templates for unseen enrollment subjects
    # --------------------------------------------------------

    enrolled_templates = _templates(
        enrollment_features,
        enrollment_labels,
    )

    # --------------------------------------------------------
    # STEP 5 — Score final verification windows
    # --------------------------------------------------------

    genuine_scores, impostor_scores = _cosine_scores(
        enrolled_templates,
        verification_features,
        verification_labels,
    )

    # --------------------------------------------------------
    # STEP 6 — Construct binary verification labels
    # --------------------------------------------------------

    y_true = np.concatenate(
        [
            np.ones(
                len(genuine_scores),
                dtype=int,
            ),
            np.zeros(
                len(impostor_scores),
                dtype=int,
            ),
        ]
    )

    scores = np.concatenate(
        [
            genuine_scores,
            impostor_scores,
        ]
    )

    # --------------------------------------------------------
    # STEP 7 — Calculate final metrics
    # --------------------------------------------------------

    metrics = verification_metrics(
        y_true,
        scores,
        threshold,
    )

    # --------------------------------------------------------
    # STEP 8 — Create result row
    # --------------------------------------------------------

    row = {
        "experiment": "E1",
        "model": "template_cosine",

        "development_subjects": int(
            len(development_templates)
        ),

        "enrollment_subjects": int(
            len(enrolled_templates)
        ),

        "genuine_scores": int(
            len(genuine_scores)
        ),

        "impostor_scores": int(
            len(impostor_scores)
        ),

        **metrics,

        "status": "MEASURED_VERIFICATION",
    }

    frame = pd.DataFrame(
        [row]
    )

    # --------------------------------------------------------
    # STEP 9 — Save E1 verification results
    # --------------------------------------------------------

    frame.to_csv(
        output_dir / "e1_verification.csv",
        index=False,
    )

    # --------------------------------------------------------
    # STEP 10 — Save protocol information
    # --------------------------------------------------------

    save_json(
        {
            "experiment": "E1",

            "protocol": (
                "independent development template/calibration; "
                "unseen-subject enrollment; "
                "window-disjoint verification"
            ),

            "threshold": metrics["threshold"],

            "development_subjects": sorted(
                development_templates
            ),

            "enrollment_subjects": sorted(
                enrolled_templates
            ),

            "development_template_windows": int(
                len(development_template_x)
            ),

            "development_calibration_windows": int(
                len(development_calibration_x)
            ),

            "development_calibration_genuine_scores": int(
                len(development_genuine_scores)
            ),

            "development_calibration_impostor_scores": int(
                len(development_impostor_scores)
            ),

            "genuine_scores": int(
                len(genuine_scores)
            ),

            "impostor_scores": int(
                len(impostor_scores)
            ),

            "status": "MEASURED_VERIFICATION",
        },
        output_dir / "e1_verification_protocol.json",
    )

    return frame


# ============================================================
# E2 — ENVIRONMENTAL / CONTEXT ROBUSTNESS
# ============================================================

def status_e2(
    metadata_columns: set[str],
) -> ExperimentStatus:
    """Report whether environmental-condition metadata is available."""

    required = {
        "speed",
        "surface",
        "footwear",
        "carrying_position",
    }

    available = (
        required & metadata_columns
    )

    if not available:

        return ExperimentStatus(
            experiment="E2",
            status="NOT_CONDUCTED",
            interpretation=(
                "No condition metadata is available "
                "in the supplied dataset."
            ),
            limitation=(
                "WISDM activity data does not provide "
                "the requested controlled speed, surface, "
                "footwear, and carrying-position factors."
            ),
        )

    return ExperimentStatus(
        experiment="E2",
        status="PARTIAL",
        interpretation=(
            f"Only these condition fields are available: "
            f"{sorted(available)}."
        ),
        limitation=(
            "Unavailable condition factors must not be simulated."
        ),
    )


# ============================================================
# E3 — TEMPLATE AGEING
# ============================================================

def status_e3(
    metadata_columns: set[str],
) -> ExperimentStatus:
    """Report whether calendar-based ageing can be measured."""

    if "session_date" not in metadata_columns:

        return ExperimentStatus(
            experiment="E3",
            status="NOT_CONDUCTED",
            interpretation=(
                "Calendar-based template ageing cannot be "
                "measured without session dates."
            ),
            limitation=(
                "WISDM subject/activity files do not provide "
                "enrolment-to-verification intervals for "
                "same-day, one-, two-, and four-week comparisons."
            ),
        )

    return ExperimentStatus(
        experiment="E3",
        status="READY",
        interpretation=(
            "Session dates are available for genuine ageing analysis."
        ),
    )


# ============================================================
# E4 — IMPOSTOR / MIMICRY EVALUATION
# ============================================================

def status_e4(
    has_mimicry_labels: bool,
) -> ExperimentStatus:
    """Report the available impostor evaluation conditions."""

    if not has_mimicry_labels:

        return ExperimentStatus(
            experiment="E4",
            status="PARTIAL",
            interpretation=(
                "Zero-effort impostor evaluation can be derived "
                "from subject-disjoint data; mimicry is not conducted."
            ),
            limitation=(
                "No labelled self-collected mimicry recordings "
                "were supplied."
            ),
        )

    return ExperimentStatus(
        experiment="E4",
        status="READY",
        interpretation=(
            "Both zero-effort and labelled mimicry conditions "
            "can be evaluated."
        ),
    )


# ============================================================
# E5 — INFERENCE PERFORMANCE
# ============================================================

def benchmark_inference(
    model,
    x,
    output_dir: str | Path,
    model_name: str,
) -> dict:
    """Measure host inference latency and peak process memory.

    Battery consumption is deliberately reported as not measured because
    this benchmark does not directly measure smartphone battery usage.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracemalloc.start()

    start = time.perf_counter()

    model.predict(x)

    elapsed = (
        time.perf_counter()
        - start
    )

    _, peak = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    result = {
        "experiment": "E5",
        "model": model_name,

        "n_windows": int(
            len(x)
        ),

        "latency_ms_total": (
            elapsed * 1000
        ),

        "latency_ms_per_window": (
            elapsed
            * 1000
            / max(
                1,
                len(x),
            )
        ),

        "peak_memory_mb": (
            peak
            / (1024**2)
        ),

        "status": "MEASURED_HOST_BENCHMARK",

        "battery": "NOT_MEASURED",
    }

    save_json(
        result,
        output_dir
        / f"e5_{model_name}.json",
    )

    return result


# ============================================================
# E6 — USABILITY / FALSE STEP-UP
# ============================================================

def status_e6(
    has_survey: bool,
) -> ExperimentStatus:
    """Report whether perceived-friction data is available."""

    if not has_survey:

        return ExperimentStatus(
            experiment="E6",
            status="PARTIAL",
            interpretation=(
                "False step-up can be computed from genuine "
                "labelled decisions; perceived friction is "
                "not conducted."
            ),
            limitation=(
                "No participant survey responses were supplied; "
                "use docs/usability_questionnaire.md for future collection."
            ),
        )

    return ExperimentStatus(
        experiment="E6",
        status="READY",
        interpretation=(
            "Survey responses are available for "
            "perceived-friction analysis."
        ),
    )