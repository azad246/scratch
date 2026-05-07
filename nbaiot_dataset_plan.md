# N-BaIoT Dataset Plan — PQC-IoT Sentinel

> **Scope:** Five devices only — Danmini Doorbell, Ecobee Thermostat, Philips Baby Monitor, Provision Security Camera, Samsung Webcam.

---

## 1. Folder Structure

```
N-BaIoT/
│
├── raw/                                  # Original, untouched CSVs from the dataset
│   ├── danmini_doorbell/
│   │   ├── benign/
│   │   │   └── benign_traffic.csv
│   │   └── attack/
│   │       ├── gafgyt_combo.csv
│   │       ├── gafgyt_junk.csv
│   │       ├── gafgyt_scan.csv
│   │       ├── gafgyt_tcp.csv
│   │       ├── gafgyt_udp.csv
│   │       ├── mirai_ack.csv
│   │       ├── mirai_scan.csv
│   │       ├── mirai_syn.csv
│   │       ├── mirai_udp.csv
│   │       └── mirai_udpplain.csv
│   │
│   ├── ecobee_thermostat/
│   │   ├── benign/
│   │   │   └── benign_traffic.csv
│   │   └── attack/
│   │       ├── gafgyt_combo.csv
│   │       ├── gafgyt_junk.csv
│   │       ├── gafgyt_scan.csv
│   │       ├── gafgyt_tcp.csv
│   │       └── gafgyt_udp.csv
│   │
│   ├── philips_baby_monitor/
│   │   ├── benign/
│   │   │   └── benign_traffic.csv
│   │   └── attack/
│   │       ├── gafgyt_combo.csv
│   │       ├── gafgyt_junk.csv
│   │       ├── gafgyt_scan.csv
│   │       ├── gafgyt_tcp.csv
│   │       └── gafgyt_udp.csv
│   │
│   ├── provision_security_camera/
│   │   ├── benign/
│   │   │   └── benign_traffic.csv
│   │   └── attack/
│   │       ├── gafgyt_combo.csv
│   │       ├── gafgyt_junk.csv
│   │       ├── gafgyt_scan.csv
│   │       ├── gafgyt_tcp.csv
│   │       └── gafgyt_udp.csv
│   │
│   └── samsung_webcam/
│       ├── benign/
│       │   └── benign_traffic.csv
│       └── attack/
│           ├── gafgyt_combo.csv
│           ├── gafgyt_junk.csv
│           ├── gafgyt_scan.csv
│           ├── gafgyt_tcp.csv
│           └── gafgyt_udp.csv
│
├── processed/                            # Preprocessed, normalized, labeled CSVs
│   ├── danmini_doorbell/
│   │   ├── benign.csv                   # All benign rows, label=0
│   │   ├── attack.csv                   # All attack rows, label=1
│   │   ├── train.csv                    # Benign-only split (80%)
│   │   ├── val.csv                      # Mixed split (10%) — benign + attack
│   │   └── test.csv                     # Mixed split (10%) — benign + attack
│   │
│   ├── ecobee_thermostat/
│   │   ├── benign.csv
│   │   ├── attack.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   ├── philips_baby_monitor/
│   │   ├── benign.csv
│   │   ├── attack.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   ├── provision_security_camera/
│   │   ├── benign.csv
│   │   ├── attack.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── samsung_webcam/
│       ├── benign.csv
│       ├── attack.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
└── fl_clients/                           # Federated Learning client-ready data (symlinks or copies)
    ├── client_1__danmini_doorbell/
    │   ├── train.csv                     # → copy/symlink from processed/danmini_doorbell/train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── client_2__ecobee_thermostat/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── client_3__philips_baby_monitor/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── client_4__provision_security_camera/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── client_5__samsung_webcam/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

---

## 2. File Naming Convention

| Pattern | Rule |
|---|---|
| `benign_traffic.csv` | Raw benign traffic per device — unchanged from source |
| `<attack_family>_<variant>.csv` | Raw attack file names — preserve original N-BaIoT naming |
| `benign.csv` | Processed, normalized benign rows — label column = `0` |
| `attack.csv` | Processed, normalized attack rows — label column = `1`, attack_type column preserved |
| `train.csv` | 80% of benign only — no attack rows — used for autoencoder unsupervised training |
| `val.csv` | 10% benign + all attack rows (resampled to balance) — used for threshold tuning |
| `test.csv` | 10% benign + remaining attack rows — final evaluation |
| `client_N__<device_slug>/` | FL client folders — N is the 1-indexed client number, device slug matches the processed folder name |

---

## 3. Standard Column Schema

All five devices must share **identical column names** after preprocessing. The N-BaIoT dataset has 115 network traffic features; add the following metadata columns:

```
device_id        (string)  — e.g., "danmini_doorbell"
client_id        (int)     — 1 through 5
label            (int)     — 0 = benign, 1 = attack
attack_type      (string)  — e.g., "gafgyt_combo", "benign" — kept for test-time analysis only
<feature_1>      (float32)
<feature_2>      (float32)
...
<feature_115>    (float32)
```

> [!IMPORTANT]
> `attack_type` and `device_id` and `client_id` are **metadata columns only**. They must be **dropped** before feeding data into the autoencoder or FL training loop to prevent data leakage.

---

## 4. Preprocessing Plan

### Step 1 — Concatenate Raw Files per Device

For each device:
- Load all CSVs inside `raw/<device>/benign/` → concatenate → tag `label=0`, `attack_type="benign"`
- Load all CSVs inside `raw/<device>/attack/` → concatenate → tag `label=1`, derive `attack_type` from filename
- Concatenate both → full device DataFrame

### Step 2 — Drop Leakage-Prone Columns

- Drop any columns that are all-zero, all-constant, or near-zero variance across the full dataset
- Drop any columns that directly encode device identity (if present in raw features)
- Apply the same column drop list to **all five devices** so every device ends up with the same feature set

### Step 3 — Handle Missing Values

- Replace `NaN` and `Inf` / `-Inf` with `0.0`
- Flag rows where more than 50% of features are zero — log count but keep them

### Step 4 — Normalize Features

- Fit a `MinMaxScaler` (range `[0, 1]`) on the **training split of benign data only** per device
- Apply the fitted scaler to val, test, and attack splits of the same device
- Save the fitted scaler as `processed/<device>/scaler.pkl` for reproducibility
- **Never fit the scaler on attack data** — prevents contamination of the anomaly signal

### Step 5 — Split Benign Data

```
benign rows
    → 80% → train.csv      (autoencoder unsupervised training)
    → 10% → val_benign     (merged into val.csv)
    → 10% → test_benign    (merged into test.csv)
```

Use a fixed `random_state=42` seed for all splits.

### Step 6 — Assemble attack.csv, val.csv, test.csv

- **attack.csv** — all attack rows (normalized), with `label=1` and `attack_type` preserved
- **val.csv** — `val_benign` rows + 50% of `attack.csv` rows (randomly sampled, seed=42)
- **test.csv** — `test_benign` rows + remaining 50% of `attack.csv` rows

> [!NOTE]
> val.csv and test.csv are intentionally **imbalanced** (more benign than attack), reflecting realistic IoT network conditions. Do not artificially balance them — let your threshold search handle this.

### Step 7 — Save Processed Files

Save per device under `processed/<device>/`:

| File | Contents |
|---|---|
| `benign.csv` | All benign rows, normalized, label=0 |
| `attack.csv` | All attack rows, normalized, label=1, attack_type preserved |
| `train.csv` | 80% benign, normalized, label=0 |
| `val.csv` | 10% benign + 50% attack, normalized |
| `test.csv` | 10% benign + 50% attack, normalized |
| `scaler.pkl` | Fitted MinMaxScaler (benign-only fit) |

### Step 8 — Populate fl_clients/

Copy or symlink `train.csv`, `val.csv`, `test.csv` from each device's processed folder into the corresponding `fl_clients/client_N__<device>/` folder.

Each FL client folder contains **only its own device's data** — no cross-device mixing.

---

## 5. Label Convention

| Value | Meaning | Used In |
|---|---|---|
| `0` | Benign / Normal | train, val, test, benign.csv |
| `1` | Attack / Anomaly | val, test, attack.csv |

The autoencoder is trained **only on label=0** rows (train.csv).  
Labels appear in val.csv and test.csv solely for evaluation (threshold tuning and F1 scoring).

---

## 6. Device ↔ Client Mapping

| Client ID | Device Slug | Folder |
|---|---|---|
| 1 | `danmini_doorbell` | `client_1__danmini_doorbell/` |
| 2 | `ecobee_thermostat` | `client_2__ecobee_thermostat/` |
| 3 | `philips_baby_monitor` | `client_3__philips_baby_monitor/` |
| 4 | `provision_security_camera` | `client_4__provision_security_camera/` |
| 5 | `samsung_webcam` | `client_5__samsung_webcam/` |

---

## 7. Reproducibility Checklist

- [ ] All random splits use `random_state=42`
- [ ] Scaler fitted on benign training data only — scaler saved as `scaler.pkl` per device
- [ ] Same column drop list applied to all five devices
- [ ] Column order is identical across all five devices
- [ ] `attack_type`, `device_id`, `client_id` are present in CSVs for traceability but **dropped in the training loop**
- [ ] `fl_clients/` mirrors `processed/` — no additional transformation applied

---

## 8. Compatibility Notes for FL Training Code

- Each client loads its own `fl_clients/client_N__<device>/train.csv` — no shared data loader
- Feature columns are assumed to be all columns **except** `label`, `attack_type`, `device_id`, `client_id`
- The autoencoder input dimension is fixed to `N_FEATURES` (number of columns after dropping metadata) — must be consistent across all clients
- Threshold tuning uses `val.csv` — threshold is computed per client locally, not globally
- Global model weights are aggregated by the server after local training on `train.csv`
