"""
preprocess_nbaiot.py
────────────────────
N-BaIoT preprocessing pipeline for PQC-IoT Sentinel.

Source layout (flat numbered files in G:/scratch/):
  <N>.benign.csv
  <N>.gafgyt.combo.csv  /  <N>.mirai.ack.csv  etc.

Device → number mapping (from device_info.csv):
  1 → Danmini Doorbell
  2 → Ecobee Thermostat
  4 → Philips Baby Monitor
  5 → Provision PT-737E Security Camera
  7 → Samsung Webcam

Outputs
-------
N-BaIoT/raw/<device_slug>/benign/benign_traffic.csv
N-BaIoT/raw/<device_slug>/attack/<attack_name>.csv
N-BaIoT/processed/<device_slug>/benign.csv
N-BaIoT/processed/<device_slug>/attack.csv
N-BaIoT/processed/<device_slug>/train.csv
N-BaIoT/processed/<device_slug>/val.csv
N-BaIoT/processed/<device_slug>/test.csv
N-BaIoT/processed/<device_slug>/scaler.pkl
N-BaIoT/fl_clients/client_<id>__<device_slug>/train.csv
N-BaIoT/fl_clients/client_<id>__<device_slug>/val.csv
N-BaIoT/fl_clients/client_<id>__<device_slug>/test.csv
"""

import os
import glob
import shutil
import pickle
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Where the flat *.csv files currently live
SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..")   # G:/scratch/

# Output root  →  G:/scratch/N-BaIoT/
BASE_DIR = os.path.dirname(__file__)                          # G:/scratch/N-BaIoT/
RAW_DIR  = os.path.join(BASE_DIR, "raw")
PROC_DIR = os.path.join(BASE_DIR, "processed")
FL_DIR   = os.path.join(BASE_DIR, "fl_clients")

RANDOM_SEED = 42

META_COLS = ["device_id", "client_id", "label", "attack_type"]

TRAIN_RATIO      = 0.80
VAL_RATIO        = 0.10
TEST_RATIO       = 0.10
ATTACK_VAL_RATIO = 0.50

# ── Device mapping ───────────────────────────────────────────────────────────
# (FL client id, source file prefix, device slug)
DEVICES = [
    (1, "1", "danmini_doorbell"),
    (2, "2", "ecobee_thermostat"),
    (3, "4", "philips_baby_monitor"),
    (4, "5", "provision_security_camera"),
    (5, "7", "samsung_webcam"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    log.info("    saved %d rows → %s", len(df), os.path.relpath(path, BASE_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — Stage raw files into the folder structure
# ──────────────────────────────────────────────────────────────────────────────

def stage_raw_files(prefix: str, device_slug: str) -> dict:
    """
    Copy flat source files  G:/scratch/<prefix>.*.csv
    into  N-BaIoT/raw/<device_slug>/benign/  and  .../attack/

    Returns a dict with keys 'benign' and 'attack' mapping to lists of
    destination paths.
    """
    benign_dst_dir = os.path.join(RAW_DIR, device_slug, "benign")
    attack_dst_dir = os.path.join(RAW_DIR, device_slug, "attack")
    ensure_dir(benign_dst_dir)
    ensure_dir(attack_dst_dir)

    staged = {"benign": [], "attack": []}

    pattern = os.path.join(SOURCE_DIR, f"{prefix}.*.csv")
    src_files = sorted(glob.glob(pattern))

    if not src_files:
        raise FileNotFoundError(
            f"No source files matched pattern: {pattern}\n"
            f"Check that SOURCE_DIR is correct: {os.path.abspath(SOURCE_DIR)}"
        )

    for src in src_files:
        basename = os.path.basename(src)                   # e.g. "1.benign.csv"
        # Strip the leading "<prefix>." to get the semantic name
        semantic  = basename[len(prefix) + 1:]             # e.g. "benign.csv"
        stem      = os.path.splitext(semantic)[0]          # e.g. "benign"
        clean_stem = stem.replace(".", "_")                # e.g. "gafgyt_combo"

        if stem == "benign":
            dst = os.path.join(benign_dst_dir, "benign_traffic.csv")
            staged["benign"].append(dst)
        else:
            dst = os.path.join(attack_dst_dir, f"{clean_stem}.csv")
            staged["attack"].append(dst)

        if not os.path.exists(dst):
            log.info("  staging %s → raw/%s", basename, os.path.relpath(dst, RAW_DIR))
            shutil.copy2(src, dst)
        else:
            log.info("  already staged: raw/%s", os.path.relpath(dst, RAW_DIR))

    return staged


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load staged CSVs
# ──────────────────────────────────────────────────────────────────────────────

def load_benign(device_slug: str, client_id: int) -> pd.DataFrame:
    benign_dir = os.path.join(RAW_DIR, device_slug, "benign")
    files = sorted(glob.glob(os.path.join(benign_dir, "*.csv")))

    if not files:
        raise FileNotFoundError(f"No benign CSVs found in: {benign_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, header=0)
        frames.append(df)
        log.info("    [benign] %d rows ← %s", len(df), os.path.basename(f))

    combined = pd.concat(frames, ignore_index=True)
    combined["label"]       = 0
    combined["attack_type"] = "benign"
    combined["device_id"]   = device_slug
    combined["client_id"]   = client_id
    return combined


def load_attack(device_slug: str, client_id: int) -> pd.DataFrame:
    attack_dir = os.path.join(RAW_DIR, device_slug, "attack")
    files = sorted(glob.glob(os.path.join(attack_dir, "*.csv")))

    if not files:
        raise FileNotFoundError(f"No attack CSVs found in: {attack_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, header=0)
        stem = os.path.splitext(os.path.basename(f))[0]   # e.g. "gafgyt_combo"
        df["attack_type"] = stem
        frames.append(df)
        log.info("    [attack/%s] %d rows", stem, len(df))

    combined = pd.concat(frames, ignore_index=True)
    combined["label"]     = 1
    combined["device_id"] = device_slug
    combined["client_id"] = client_id
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Clean and align columns
# ──────────────────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df[feature_cols] = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


def drop_low_variance(df: pd.DataFrame, feature_cols: list, threshold: float = 1e-8) -> list:
    """Return list of columns to DROP (near-zero variance on benign data)."""
    variances = df[feature_cols].var()
    return variances[variances < threshold].index.tolist()


def align_to_reference(df: pd.DataFrame, ref_cols: list) -> pd.DataFrame:
    """
    Keep only ref_cols + META_COLS.
    Fill any missing columns with 0.0 (handles devices with fewer attack types).
    """
    target = ref_cols + META_COLS
    for col in target:
        if col not in df.columns:
            df[col] = np.float32(0.0)
    return df[target].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Normalize
# ──────────────────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame, feature_cols: list) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols]).astype(np.float32)
    return out


def save_scaler(scaler: MinMaxScaler, out_dir: str) -> None:
    path = os.path.join(out_dir, "scaler.pkl")
    with open(path, "wb") as fh:
        pickle.dump(scaler, fh)
    log.info("    scaler saved → processed/%s/scaler.pkl",
             os.path.basename(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Split
# ──────────────────────────────────────────────────────────────────────────────

def split_benign(df: pd.DataFrame):
    train, temp = train_test_split(
        df, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED, shuffle=True
    )
    rel_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val, test = train_test_split(
        temp, test_size=(1 - rel_val), random_state=RANDOM_SEED, shuffle=True
    )
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def split_attack(df: pd.DataFrame):
    val_atk, test_atk = train_test_split(
        df, test_size=(1 - ATTACK_VAL_RATIO), random_state=RANDOM_SEED, shuffle=True
    )
    return val_atk.reset_index(drop=True), test_atk.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Save all processed outputs for one device
# ──────────────────────────────────────────────────────────────────────────────

def save_device_outputs(device_slug, benign_df, attack_df,
                        train_df, val_df, test_df, scaler):
    out_dir = os.path.join(PROC_DIR, device_slug)
    ensure_dir(out_dir)

    save_csv(benign_df, os.path.join(out_dir, "benign.csv"))
    save_csv(attack_df, os.path.join(out_dir, "attack.csv"))
    save_csv(train_df,  os.path.join(out_dir, "train.csv"))
    save_csv(val_df,    os.path.join(out_dir, "val.csv"))
    save_csv(test_df,   os.path.join(out_dir, "test.csv"))
    save_scaler(scaler, out_dir)

    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Populate FL client folder
# ──────────────────────────────────────────────────────────────────────────────

def populate_fl_client(client_id: int, device_slug: str, proc_dir: str) -> None:
    client_dir = os.path.join(FL_DIR, f"client_{client_id}__{device_slug}")
    ensure_dir(client_dir)
    for split in ("train", "val", "test"):
        src = os.path.join(proc_dir, f"{split}.csv")
        dst = os.path.join(client_dir, f"{split}.csv")
        shutil.copy2(src, dst)
        log.info("    FL client copy → fl_clients/client_%d__%s/%s.csv",
                 client_id, device_slug, split)


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — Column consistency check
# ──────────────────────────────────────────────────────────────────────────────

def verify_column_consistency() -> bool:
    log.info("Verifying column consistency across all five devices …")
    ref_cols = None
    ref_slug = None
    all_ok   = True

    for client_id, _, device_slug in DEVICES:
        path = os.path.join(PROC_DIR, device_slug, "train.csv")
        if not os.path.exists(path):
            log.warning("  MISSING: %s", path)
            all_ok = False
            continue

        cols = [c for c in pd.read_csv(path, nrows=0).columns if c not in META_COLS]

        if ref_cols is None:
            ref_cols = cols
            ref_slug = device_slug
            log.info("  [%s] reference: %d feature columns", device_slug, len(cols))
        else:
            extra   = set(cols) - set(ref_cols)
            missing = set(ref_cols) - set(cols)
            if extra or missing:
                log.error("  [%s] MISMATCH vs [%s] — extra: %s  missing: %s",
                          device_slug, ref_slug, extra, missing)
                all_ok = False
            else:
                log.info("  [%s] OK — %d feature columns", device_slug, len(cols))

    if all_ok:
        log.info("Column consistency check PASSED ✓")
    else:
        log.error("Column consistency check FAILED ✗")
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Per-device pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_device(client_id: int, prefix: str, device_slug: str,
                   shared_feature_cols: list | None) -> list:
    log.info("")
    log.info("━" * 60)
    log.info("  Device [%d/5]: %s  (source prefix=%s)", client_id, device_slug, prefix)
    log.info("━" * 60)

    # 0. Stage raw files
    log.info("  → staging raw files …")
    stage_raw_files(prefix, device_slug)

    # 1. Load
    log.info("  → loading staged CSVs …")
    benign_raw = load_benign(device_slug, client_id)
    attack_raw = load_attack(device_slug, client_id)

    # 2. Clean
    benign_raw = clean_dataframe(benign_raw)
    attack_raw = clean_dataframe(attack_raw)

    # 3. Resolve feature columns
    if shared_feature_cols is None:
        all_feat   = get_feature_cols(benign_raw)
        drop_cols  = drop_low_variance(benign_raw, all_feat)
        feat_cols  = [c for c in all_feat if c not in drop_cols]
        if drop_cols:
            log.info("  dropped %d low-variance cols", len(drop_cols))
    else:
        feat_cols = shared_feature_cols

    benign_aligned = align_to_reference(benign_raw, feat_cols)
    attack_aligned = align_to_reference(attack_raw, feat_cols)

    # 4. Split benign
    train_raw, val_b, test_b = split_benign(benign_aligned)

    # 5. Fit scaler on benign train only
    scaler = fit_scaler(train_raw, feat_cols)

    # 6. Normalize all splits
    benign_norm = apply_scaler(benign_aligned, scaler, feat_cols)
    attack_norm = apply_scaler(attack_aligned, scaler, feat_cols)
    train_norm  = apply_scaler(train_raw,      scaler, feat_cols)
    val_b_norm  = apply_scaler(val_b,          scaler, feat_cols)
    test_b_norm = apply_scaler(test_b,         scaler, feat_cols)

    # 7. Split attack
    val_a_norm, test_a_norm = split_attack(attack_norm)

    # 8. Assemble val/test (benign + attack, shuffled)
    val_df = (pd.concat([val_b_norm, val_a_norm], ignore_index=True)
                .sample(frac=1, random_state=RANDOM_SEED)
                .reset_index(drop=True))
    test_df = (pd.concat([test_b_norm, test_a_norm], ignore_index=True)
                 .sample(frac=1, random_state=RANDOM_SEED)
                 .reset_index(drop=True))

    log.info(
        "  split → train: %d | val: %d (%d b+%d a) | test: %d (%d b+%d a)",
        len(train_norm),
        len(val_df),  len(val_b_norm),  len(val_a_norm),
        len(test_df), len(test_b_norm), len(test_a_norm),
    )

    # 9. Save processed CSVs
    log.info("  → saving processed files …")
    out_dir = save_device_outputs(
        device_slug, benign_norm, attack_norm,
        train_norm, val_df, test_df, scaler
    )

    # 10. Populate FL client
    log.info("  → populating FL client folder …")
    populate_fl_client(client_id, device_slug, out_dir)

    return feat_cols


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   N-BaIoT Preprocessing Pipeline — PQC-IoT Sentinel     ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("Source dir : %s", os.path.abspath(SOURCE_DIR))
    log.info("Output dir : %s", os.path.abspath(BASE_DIR))
    log.info("")

    shared_feature_cols = None

    for client_id, prefix, device_slug in DEVICES:
        feat_cols = process_device(client_id, prefix, device_slug, shared_feature_cols)
        if shared_feature_cols is None:
            shared_feature_cols = feat_cols
            log.info("  Feature reference locked: %d columns", len(shared_feature_cols))

    log.info("")
    verify_column_consistency()

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   Pipeline complete.                                     ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║   processed/  → %s", os.path.abspath(PROC_DIR).ljust(39) + "║")
    log.info("║   fl_clients/ → %s", os.path.abspath(FL_DIR).ljust(39)  + "║")
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
