# Risk-Adaptive Continuous Authentication for Mobile-Money Transactions Using Smartphone-Based Gait Recognition

This repository implements the prototype specified by Assignment #1. It uses smartphone accelerometer and gyroscope data as an additional behavioural signal within a mobile-money authentication workflow. The system does not replace PIN or OTP authentication. It produces a gait confidence and combines that confidence with transaction risk to return **ALLOW**, **STEP_UP_AUTHENTICATION**, or **DENY**.

## Architecture

The implementation follows the proposal’s five layers:

| Layer | Prototype component | Responsibility |
|---|---|---|
| Sensing | `data.py` | Load phone accelerometer and gyroscope observations. |
| Preprocessing | `preprocessing.py` | Validate, filter, derive orientation-reduced magnitudes, normalize, and segment windows. |
| Feature/Representation | `features.py` and raw-window path | Handcrafted classical features or normalized raw six-channel windows. |
| Matching | `models.py`, `authentication.py` | SVM, Random Forest, k-NN, or CNN-LSTM prediction and verification confidence. |
| Risk and Decision | `risk_engine.py` | Independent risk classification and proposal-defined decision matrix. |

## Dataset

The primary supported source is the [UCI WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset). UCI describes 51 subjects, phone and smartwatch accelerometer/gyroscope streams, 20 Hz sampling, subject identifiers, activity labels, and raw rows containing subject, activity, timestamp, x, y, and z. The prototype uses phone walking activity code `A` and does not assume that the dataset contains controlled footwear, surface, carrying-position, calendar-ageing, mimicry, survey, or battery labels.

The proposal also identifies the [OU-ISIR inertial gait dataset](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html). It is supported as a future adapter, but its download requires a signed institutional release agreement and password. No OU-ISIR data is claimed as locally available unless the user supplies it.

## Installation

```bash
cd gait-mobile-money-authentication
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Install PyTorch separately if the CNN-LSTM experiment is to be run:

```bash
pip install torch
```

## Dataset preparation

Download the WISDM archive from UCI and extract it under `data/raw/wisdm/`. The loader searches recursively for files named like `data_1600_accel_phone.txt` and `data_1600_gyro_phone.txt`. It reads only walking records (`activity == A`), validates numeric observations, aligns phone modalities by timestamp, and discards unmatched or invalid rows rather than imputing fabricated sensor values.

Create the subject-disjoint NPZ consumed by the experiment runner:

```bash
PYTHONPATH=src python scripts/prepare_wisdm.py \
  --raw-root data/raw/wisdm \
  --output data/processed/wisdm_subject_disjoint.npz
```

The preparation script uses the default 60/20/20 subject split, 10-second windows, and 10-second stride. The non-overlapping stride is intentional for authentication evaluation: enrollment and verification windows must not share raw sensor samples. It saves only train/test windows and labels; the validation subject list is retained for auditability.

## Risk-policy usage

```bash
PYTHONPATH=src python -m gait_auth.cli decision --confidence high --amount 50
PYTHONPATH=src python -m gait_auth.cli decision --confidence high --amount 1500
PYTHONPATH=src python -m gait_auth.cli decision --confidence low --amount 1500
```

The default policy is:

| Gait confidence | Low risk | Medium risk | High risk |
|---|---|---|---|
| High | Allow | Allow | Step-up authentication |
| Medium | Allow | Step-up authentication | Step-up authentication |
| Low | Step-up authentication | Step-up authentication | Deny |

Gait never independently authorizes a high-risk transaction.

## Testing

Run the automated software tests with:

```bash
PYTHONPATH=src pytest -q
```

The tests cover data parsing, invalid and missing observations, filtering and segmentation, feature dimensions, model prediction, authentication scoring, risk classification, all policy cells, invalid inputs, and edge cases.

## Research experiments

The `gait_auth.experiments` module provides reproducible output paths for:

| Experiment | Implementation status in the public-data prototype |
|---|---|
| E1 | Executable after preparing subject-disjoint windows; classification metrics are measured, while FAR/FRR/EER require a verification-score protocol. |
| E2 | Not conducted for WISDM unless condition metadata is supplied; no artificial condition labels are created. |
| E3 | Not conducted without genuine session dates at same-day, 1-week, 2-week, and 4-week intervals. |
| E4 | Zero-effort impostor analysis is possible from held-out subjects; mimicry is not conducted without labelled self-collected mimicry data. |
| E5 | Host inference latency and process memory can be benchmarked; smartphone battery impact is not measured in this sandbox. |
| E6 | False step-up rate can be computed from labelled decisions; perceived friction is not conducted without participant survey responses. |

All result files must distinguish **measured**, **expected**, **not conducted**, and **limited** outcomes. The repository must never report fabricated participants, observations, metrics, attacks, survey answers, or battery measurements.

## Security and privacy

Gait is sensitive biometric information. Raw signals and templates should be de-identified, access-controlled, retained only as long as necessary, and separated from names, phone numbers, and mobile-money account identifiers. The prototype contains no real mobile-money integration and no transaction execution capability.

## References

[1]: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset "UCI WISDM Smartphone and Smartwatch Activity and Biometrics Dataset"
[2]: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html "OU-ISIR Gait Database, Inertial Sensor Dataset"
