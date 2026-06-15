# lts-monolithic-ddos-detection

A **single-machine, real-time DDoS detection system** using Scapy for live packet capture and a trained Random Forest classifier for threat classification. Built during **Practice School I (PS-1)** as the core ML engineering deliverable.

---

## How This Fits Into the Larger Work

This is the **second phase** of the LTS project series — a self-contained monolithic implementation where packet capture, feature extraction, and ML inference all run on one machine:

```
lts-coursework-experiments       (early classifiers and Scapy experiments)
        ↓
lts-monolithic-ddos-detection   ← you are here
        ↓
lts-distributed-ddos-detection   (split into capture agents + central inference)

healthcare-prediction            (separate parallel project)
```

---

## What's in This Repo

### Code (top-level)
| File | Description |
|---|---|
| `real_time_ddos_detection.py` | **Main entrypoint.** Sniffs live packets via Scapy, extracts 28 flow features, runs local ML inference, and logs alerts |
| `train.py` | Trains the Random Forest model on the CIC-DDoS2019 dataset |
| `predict_ddos.py` | Offline batch prediction on captured traffic records |
| `concat.py` | Concatenates multiple raw CSV capture files |
| `reducer.py` / `rereducer.py` | Balances and reduces the large DDoS dataset for training |
| `random_sample_gen.py` | Samples records from the dataset for quick testing |
| `create_test_records.py` | Generates test records for validating inference |
| `ddosattack.py` | Simulates basic DDoS traffic patterns |

### Outputs / Artifacts (included for reference)
| File | Description |
|---|---|
| `model_metadata.txt` | Feature list and model parameters used at training time |
| `model_training_summary.txt` | Training accuracy and class distribution summary |
| `confusion_matrix.png` | Model evaluation — confusion matrix |
| `ddos_model_performance.png` | ROC / accuracy visualisation |
| `feature_importance.png` | Top-28 feature importance chart |
| `enhanced_ddos_alerts.json` | Sample alert log output from a live detection run |

### Documents (`documents/`)
| Subfolder | Contents |
|---|---|
| `documents/reports/` | Software Architecture Document (PDF + DOCX), system comparison report |
| `documents/presentations/` | Final assignment presentation, real-time system overview PPTX |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install scapy scikit-learn pandas joblib numpy matplotlib seaborn
```

**Run live detection** (requires root for packet sniffing):
```bash
sudo python real_time_ddos_detection.py
```

**Train model** (requires dataset — see below):
```bash
python train.py
```

---

## Data and Models Not Included

Raw datasets and trained model binaries are **not** in this repo. They live locally under `raw data collected for project/`:

| File | Size | Used by |
|---|---|---|
| `CIC_DDoS2019_To_Use.csv` | 54 MB | `train.py` |
| `CIC_DDoS2019_balanced_reduced.csv` | 43 MB | `reducer.py`, `random_sample_gen.py` |
| `optimized_ddos_model.pkl` | 37 MB | `real_time_ddos_detection.py` |
| `ddos_detection_model.pkl` | 255 MB | Earlier training iteration |
| `improved_ddos_model.pkl` | 37 MB | Intermediate model |

The code resolves model/data paths dynamically. Update the path constants if running outside the original layout.

---

## See Also
- [`lts-coursework-experiments`](https://github.com/nikunj-gupta-1/lts-coursework-experiments) — precursor assignments and Scapy experiments
- [`lts-distributed-ddos-detection`](https://github.com/nikunj-gupta-1/lts-distributed-ddos-detection) — distributed multi-device version of this system
- [`healthcare-prediction`](https://github.com/nikunj-gupta-1/healthcare-prediction) — separate Streamlit ML dashboard
