# Risk-Adaptive Mobile Money Gait Authentication

A risk-adaptive continuous behavioral biometric authentication system for mobile money transactions.

The architecture couples a continuous gait verification pipeline (using a CNN-LSTM deep learning sequence model exported to ONNX) with a contextual Identity and Access Management (IAM) policy engine. Rather than treating gait as a fragile binary gatekeeper, the system leverages gait as a dynamic confidence modifier within a defense-in-depth framework—balancing everyday user friction against transaction risk.

---

## Core Research Thesis

> Continuous gait biometrics provide a passive, continuous confidence signal rather than an absolute binary authorization gate. By fusing biometric confidence with transaction-level risk context (transfer value, counterparty familiarity, and device anomalies), authentication can dynamically adapt—authorizing routine payments frictionlessly, stepping up high-consequence transfers, or denying anomalous activity outright.

---

## System Architecture

[ Android Client ]

Accelerometer / Gyroscope Sampling (50 Hz)

Thread-Safe Circular Ring Buffer

Retrofit HTTP Client / Android BiometricPrompt
│
▼  (10-second window / 200 samples)
[ FastAPI Inference Service ]

Sliding-Window Preprocessing & Normalization

ONNX Runtime Evaluation (CNN-LSTM Model)

Cosine Similarity vs. Enrolled Template
│
▼  (Biometric Confidence: High / Medium / Low)
[ Risk-Adaptive Policy Engine ]

Evaluates: Amount, New Recipient, Geolocation Anomaly

Resolves IAM Policy Decision:
├── ALLOW           (Frictionless execution)
├── STEP_UP_LIGHT   (Contextual PIN / Device Biometric)
└── DENY            (Transaction blocked & audited)


### End-to-End Decision Matrix

| Biometric Confidence | Transaction Risk Context | Engine Decision | User Experience |
| :--- | :--- | :--- | :--- |
| **High** ($\ge 0.998$) | Low (Routine amount, trusted contact) | **ALLOW** | Frictionless (Instant execution) |
| **Medium / High** | Elevated (New recipient or high value) | **STEP_UP_LIGHT** | Contextual PIN / Biometric prompt |
| **Low** ($< 0.995$) | High / Anomalous location | **DENY** | Transaction blocked & audit logged |

---

## Empirical Verification Highlights

Evaluated on the WISDM benchmark dataset under strict **subject-disjoint partitions** (30 calibration subjects, 11 unseen enrollment subjects) with non-overlapping 10-second evaluation windows to eliminate temporal autocorrelation leakage:

* **Biometric Verification (E1):**
  * **Equal Error Rate (EER):** 20.17%
  * **Overall Accuracy:** 77.62%
  * **Operating Threshold:** 0.9971
  * **False Acceptance Rate (FAR):** 23.19% (across 1,190 zero-effort impostor evaluations)
  * **False Rejection Rate (FRR):** 14.29% (across 119 genuine evaluations)
* **Multiclass Baseline Contrast:**
  * Conventional closed-set classifiers (SVM, Random Forest, KNN) achieved **0.0% accuracy** and **0.0 macro F1** on the subject-disjoint test set, demonstrating the limitation of closed-set classification and the requirement for open-set template matching in open-world biometrics.
* **Host Resource Profiling (E5):**
  * Sub-millisecond host inference latency ($0.055\text{–}0.913\text{ ms}$ per evaluation window) with peak memory allocation $<0.4\text{ MB}$.

---

## Experimental Scope & Evaluation Status

| Experiment | Status | Scope & Rationale |
| :--- | :--- | :--- |
| **E1: Biometric Verification** | **Completed** | Established verification baseline: 20.17% EER, 77.62% accuracy on unseen subjects. |
| **E2: Environmental Invariance** | **Excluded** | Benchmark dataset lacks ground-truth annotations for terrain, footwear, and carry variations. |
| **E3: Longitudinal Ageing** | **Excluded** | Benchmark dataset lacks multi-week session timestamps. |
| **E4: Adversarial Mimicry** | **Partial** | Quantified against 1,190 zero-effort impostor attempts; active physical mimicry reserved for future human trials. |
| **E5: Resource Profiling** | **Completed** | Host inference profiling across models ($<1\text{ ms}$ latency, $<0.4\text{ MB}$ RAM). |
| **E6: Friction Adaptation** | **Demonstrated** | Validated end-to-end through `ALLOW`, `STEP_UP_LIGHT`, and `DENY` states on Android emulator. |

---

## Repository Structure

```text
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI server endpoints
│   │   ├── models/                   # ONNX runtime wrapper & gait sequence models
│   │   └── services/                 # Risk engine & policy evaluation
│   ├── gait_model.onnx               # Exported CNN-LSTM ONNX model
│   └── requirements.txt              # Backend dependencies
├── gait-mobile-money-authentication/
│   ├── scripts/
│   │   ├── prepare_wisdm.py          # Data ingestion, windowing & subject-disjoint split
│   │   ├── run_experiments.py        # E1-E6 verification, baseline & benchmark runner
│   │   └── plot_evaluation_figures.py# Pure NumPy/Matplotlib ROC & score distribution generator
│   ├── src/gait_auth/
│   │   ├── data.py                   # WISDM signal parser & interpolation
│   │   ├── preprocessing.py          # Filtering & temporal windowing
│   │   ├── features.py               # Time/frequency domain feature extraction
│   │   ├── models.py                 # Classical models & CNN-LSTM PyTorch definition
│   │   ├── risk_engine.py            # Contextual risk classification logic
│   │   └── evaluation.py             # Open-set template matching & EER calculations
│   └── results/
│       ├── tables/                   # e1_verification.csv, e1_classification_baseline.csv
│       ├── json/                     # e5 host latency & memory benchmarks
│       └── figures/                  # e1_roc_curve.png, e1_score_distribution.png
└── android/ (MobileMoneyGaitAuth)
    └── app/src/main/
        ├── AndroidManifest.xml       # Hardware sensor & foreground service permissions
        ├── java/com/momo/gaitauth/
        │   ├── MainActivity.kt       # Transaction UI & step-up biometric prompt logic
        │   ├── api/                  # Retrofit networking client
        │   └── sensor/
        │       ├── GaitSensorService.kt # Continuous background sensor listener
        │       └── GaitDataBuffer.kt    # Synchronized circular ring buffer
        └── res/layout/activity_main.xml
Installation & Quickstart
1. Backend Inference Service
PowerShell
# Navigate to backend directory
cd backend

# Initialize and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Launch FastAPI service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
API documentation will be accessible at http://localhost:8000/docs.

2. Android Client Application
Open the MobileMoneyGaitAuth project directory in Android Studio.

Synchronize project Gradle dependencies.

Configure the Retrofit API base URL:

Android Emulator: http://10.0.2.2:8000/

Physical Device: http://<HOST_MACHINE_LOCAL_IP>:8000/

Build and deploy to an emulator or USB-connected device running Android API 34+.

Evaluation & Reproduction
To reproduce the subject-disjoint windowing, evaluation tables, and figures:

PowerShell
cd gait-mobile-money-authentication

# 1. Window raw WISDM data into subject-disjoint splits
python scripts/prepare_wisdm.py --raw-root path/to/WISDM/raw --output data/processed/wisdm_subject_disjoint.npz

# 2. Run verification pipeline and host profiling
python scripts/run_experiments.py --npz data/processed/wisdm_subject_disjoint.npz --output results

# 3. Generate ROC and score distribution plots
python scripts/plot_evaluation_figures.py
Generated publication plots:

results/figures/e1_roc_curve.png

results/figures/e1_score_distribution.png
