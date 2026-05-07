"""
eda_phase3.py
─────────────
Exploratory Data Analysis for N-BaIoT processed splits.

Processed data lives at:
    N-BaIoT/N-BaIoT/processed/<device_slug>/
        benign.csv, attack.csv, train.csv, val.csv, test.csv

Outputs saved to:
    outputs/eda/
        eda_summary.json
        eda_summary.csv
        <device>_feature_stats.csv
        plots/<device>_label_distribution.png
        plots/<device>_<feature>.png
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================

# Correct path: preprocess_nbaiot.py writes to N-BaIoT/N-BaIoT/processed/
BASE_DIR    = os.path.join("N-BaIoT", "N-BaIoT", "processed")
OUTPUT_DIR  = os.path.join("outputs", "eda")

DEVICE_FOLDERS = [
    "danmini_doorbell",
    "ecobee_thermostat",
    "philips_baby_monitor",
    "provision_security_camera",
    "samsung_webcam",
]

# Priority order: the first file found in each device folder will be used
CSV_CANDIDATES = ["train.csv", "val.csv", "test.csv", "benign.csv", "attack.csv"]

# For very large CSVs, read only this many rows to keep EDA fast.
# Set to None to read everything (may be slow / OOM on large attack files).
SAMPLE_ROWS = 200_000

# Max numeric features to plot histograms for
MAX_HIST_FEATURES = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

sns.set_theme(style="whitegrid", palette="viridis")


# =============================================================================
# HELPERS
# =============================================================================

def find_first_existing_csv(folder_path: str, candidates: list[str]):
    """Return path of the first candidate CSV that exists, or None."""
    for fname in candidates:
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            return fpath
    return None


def find_all_existing_csvs(folder_path: str, candidates: list[str]) -> dict:
    """Return {filename: full_path} for every candidate that exists."""
    found = {}
    for fname in candidates:
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            found[fname] = fpath
    return found


def load_csv_sampled(path: str, max_rows: int | None) -> pd.DataFrame:
    """Load a CSV, optionally capping at max_rows via chunked sampling."""
    if max_rows is None:
        return pd.read_csv(path, low_memory=False)

    # First pass: get total row count without loading data
    total = sum(1 for _ in open(path, "r", encoding="utf-8", errors="replace")) - 1  # minus header

    if total <= max_rows:
        return pd.read_csv(path, low_memory=False)

    # Reservoir / skip-row sampling: keep every k-th row
    skip_every = max(1, total // max_rows)
    rows_to_skip = set(range(1, total + 1, skip_every))          # 1-based (header=0)
    # We want to KEEP evenly-spaced rows, so skip the complement
    keep_idx = set(range(1, total + 1)) - rows_to_skip
    # Simpler: use pandas skiprows with a lambda
    df = pd.read_csv(
        path,
        skiprows=lambda i: i > 0 and i not in range(1, total + 1, skip_every),
        low_memory=False,
    )
    return df.head(max_rows)


def infer_label_column(df: pd.DataFrame) -> str | None:
    """Return the first matching label column name, or None."""
    for col in ["label", "Label", "class", "Class", "target", "Target"]:
        if col in df.columns:
            return col
    return None


def summarize_dataframe(df: pd.DataFrame, label_col: str | None, source_file: str, sampled: bool) -> dict:
    summary = {
        "rows_in_sample": int(df.shape[0]),
        "cols":           int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "sampled":        sampled,
        "source_file":    source_file,
    }
    if label_col and label_col in df.columns:
        summary["label_column"] = label_col
        summary["label_counts"] = {
            str(k): int(v)
            for k, v in df[label_col].value_counts(dropna=False).items()
        }
    return summary


def plot_label_distribution(df: pd.DataFrame, label_col: str, title: str, save_path: str) -> None:
    counts = df[label_col].value_counts(dropna=False)
    plot_df = pd.DataFrame({"label": counts.index.astype(str), "count": counts.values})
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 1.2), 5))
    sns.barplot(data=plot_df, x="label", y="count", ax=ax, hue="label", palette="viridis", legend=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, feature_cols: list[str],
                                device_name: str, save_dir: str,
                                max_features: int = MAX_HIST_FEATURES) -> None:
    selected = feature_cols[:max_features]
    for col in selected:
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            sns.histplot(df[col].dropna(), bins=50, kde=True, ax=ax, color="steelblue")
        except Exception:
            ax.text(0.5, 0.5, f"Could not plot {col}", ha="center", va="center")
        ax.set_title(f"{device_name} — {col}", fontsize=11)
        plt.tight_layout()
        safe_name = col.replace("/", "_").replace(" ", "_").replace("\\", "_")[:60]
        plt.savefig(os.path.join(save_dir, f"{device_name}_{safe_name}.png"), dpi=120)
        plt.close(fig)


def feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    stats_df = df[feature_cols].describe().T
    stats_df["missing"] = df[feature_cols].isna().sum().values
    return stats_df


def per_split_summary(folder_path: str, all_csvs: dict) -> dict:
    """Return {split_name: {rows, cols, label_counts}} for every found CSV."""
    split_info = {}
    for fname, fpath in all_csvs.items():
        try:
            # Just peek at shape without loading everything
            peek = pd.read_csv(fpath, nrows=0)
            total_rows = sum(1 for _ in open(fpath, encoding="utf-8", errors="replace")) - 1
            label_col = infer_label_column(peek)
            info = {"rows": total_rows, "cols": int(len(peek.columns))}
            if label_col:
                # Sample to get label distribution
                sample = load_csv_sampled(fpath, min(50_000, SAMPLE_ROWS))
                info["label_counts"] = {
                    str(k): int(v)
                    for k, v in sample[label_col].value_counts(dropna=False).items()
                }
            split_info[fname] = info
        except Exception as exc:
            split_info[fname] = {"error": str(exc)}
    return split_info


# =============================================================================
# MAIN EDA
# =============================================================================

all_reports = []

print(f"\n{'='*60}")
print(f"  N-BaIoT EDA - Phase 3")
print(f"  BASE_DIR : {os.path.abspath(BASE_DIR)}")
print(f"  OUTPUT   : {os.path.abspath(OUTPUT_DIR)}")
print(f"{'='*60}\n")

for device in DEVICE_FOLDERS:
    folder_path = os.path.join(BASE_DIR, device)
    print(f"[{device}]")

    # ── Folder existence check ────────────────────────────────────────────────
    if not os.path.isdir(folder_path):
        msg = f"Folder not found: {os.path.abspath(folder_path)}"
        print(f"  [SKIP] {msg}\n")
        all_reports.append({"device": device, "error": msg})
        continue

    # ── Discover CSVs ─────────────────────────────────────────────────────────
    all_csvs = find_all_existing_csvs(folder_path, CSV_CANDIDATES)
    if not all_csvs:
        msg = (f"No CSV found in {folder_path}. "
               f"Expected one of: {CSV_CANDIDATES}")
        print(f"  [SKIP] {msg}\n")
        all_reports.append({"device": device, "error": msg})
        continue

    print(f"  Found splits: {list(all_csvs.keys())}")

    try:
        # ── Per-split summary (fast, no full load) ────────────────────────────
        split_summaries = per_split_summary(folder_path, all_csvs)

        # ── Load primary split for deep analysis ──────────────────────────────
        primary_fname = next(iter(all_csvs))          # first found (priority order)
        primary_path  = all_csvs[primary_fname]

        total_rows = split_summaries.get(primary_fname, {}).get("rows", "?")
        sampled    = isinstance(total_rows, int) and total_rows > (SAMPLE_ROWS or total_rows + 1)
        print(f"  Loading '{primary_fname}' ({total_rows} rows) "
              f"{'[SAMPLED]' if sampled else '[FULL]'} …")

        df        = load_csv_sampled(primary_path, SAMPLE_ROWS)
        label_col = infer_label_column(df)

        # ── Feature columns ───────────────────────────────────────────────────
        meta_cols    = ["device_id", "client_id", "label", "attack_type"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        numeric_feat = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

        # ── Save feature stats ────────────────────────────────────────────────
        if numeric_feat:
            stats_df = feature_stats(df, numeric_feat)
            stats_df.to_csv(os.path.join(OUTPUT_DIR, f"{device}_feature_stats.csv"))

        # ── Plot label distribution ───────────────────────────────────────────
        if label_col:
            plot_label_distribution(
                df, label_col,
                f"Label Distribution — {device} ({primary_fname})",
                os.path.join(OUTPUT_DIR, "plots", f"{device}_label_distribution.png")
            )
            print(f"  Label counts ({label_col}): "
                  f"{df[label_col].value_counts(dropna=False).to_dict()}")

        # ── Plot feature histograms ───────────────────────────────────────────
        if numeric_feat:
            plot_feature_distributions(
                df, numeric_feat, device,
                os.path.join(OUTPUT_DIR, "plots"),
                max_features=MAX_HIST_FEATURES
            )

        # ── Build report ──────────────────────────────────────────────────────
        report = {
            "device":          device,
            "primary_file":    primary_fname,
            "splits_found":    list(all_csvs.keys()),
            "split_summaries": split_summaries,
            **summarize_dataframe(df, label_col, primary_path, sampled),
            "num_feature_cols":    len(feature_cols),
            "num_numeric_features": len(numeric_feat),
        }

        all_reports.append(report)
        print(f"  [OK] Done - {df.shape[0]} rows x {df.shape[1]} cols\n")

    except Exception as exc:
        import traceback
        err_detail = traceback.format_exc()
        print(f"  [ERROR] {exc}\n")
        all_reports.append({"device": device, "error": str(exc), "traceback": err_detail})

# =============================================================================
# SAVE REPORTS
# =============================================================================

eda_json_path = os.path.join(OUTPUT_DIR, "eda_summary.json")
with open(eda_json_path, "w") as f:
    json.dump(all_reports, f, indent=4, default=str)

# Flatten for CSV (top-level keys only; nested dicts become JSON strings)
flat_reports = []
for r in all_reports:
    flat = {}
    for k, v in r.items():
        flat[k] = json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
    flat_reports.append(flat)

pd.DataFrame(flat_reports).to_csv(
    os.path.join(OUTPUT_DIR, "eda_summary.csv"), index=False
)

print(f"\n{'='*60}")
print(f"  EDA complete.")
print(f"  JSON  -> {eda_json_path}")
print(f"  CSV   -> {os.path.join(OUTPUT_DIR, 'eda_summary.csv')}")
print(f"  Plots -> {os.path.join(OUTPUT_DIR, 'plots')}")
print(f"{'='*60}\n")