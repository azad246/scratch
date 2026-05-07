import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =========================
# CONFIG
# =========================
BASE_DIR    = os.path.join("N-BaIoT", "N-BaIoT", "processed")
OUTPUT_DIR  = os.path.join("outputs", "final_processed")


DEVICE_FOLDERS = [
    "danmini_doorbell",
    "ecobee_thermostat",
    "philips_baby_monitor",
    "provision_security_camera",
    "samsung_webcam"
]

CSV_CANDIDATES = ["train.csv", "val.csv", "test.csv", "benign.csv", "attack.csv"]
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def find_first_existing_csv(folder_path, candidates):
    for fname in candidates:
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            return fpath
    return None


def infer_label_column(df):
    candidates = ["label", "Label", "class", "Class", "target", "Target"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clean_dataframe(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(axis=0, how="all")
    return df


def standardize_labels(series):
    if series.dtype == object:
        s = series.astype(str).str.strip().str.lower()
        mapping = {
            "benign": 0,
            "normal": 0,
            "0": 0,
            "attack": 1,
            "malicious": 1,
            "1": 1
        }
        return s.map(lambda x: mapping.get(x, x))
    return series


def prepare_device_data(device_path):
    csv_path = find_first_existing_csv(device_path, CSV_CANDIDATES)
    if csv_path is None:
        raise FileNotFoundError(f"No CSV found in {device_path}")

    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)

    label_col = infer_label_column(df)
    if label_col is None:
        raise ValueError(f"No label column found in {csv_path}")

    df[label_col] = standardize_labels(df[label_col])

    # Drop rows with missing label after mapping
    df = df.dropna(subset=[label_col])

    # Separate features and labels
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    # Keep only numeric columns for modeling
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols].copy()

    # Fill missing values in numeric columns
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, label_col, csv_path


def save_split(df_X, y, device_name, split_name, output_dir):
    out_df = df_X.copy()
    out_df["label"] = y.values
    out_path = os.path.join(output_dir, device_name, f"{split_name}.csv")
    out_df.to_csv(out_path, index=False)
    return out_path


# =========================
# MAIN PROCESSING
# =========================
summary = []

for device in DEVICE_FOLDERS:
    device_path = os.path.join(BASE_DIR, device)
    device_out = os.path.join(OUTPUT_DIR, device)
    os.makedirs(device_out, exist_ok=True)

    X, y, label_col, source_file = prepare_device_data(device_path)

    # Stratified split: train+val/test first
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Now split train/val from remaining data
    val_ratio_of_temp = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_of_temp,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # Scale using train only
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    # Save processed splits
    train_path = save_split(X_train_scaled, y_train, device, "train", OUTPUT_DIR)
    val_path = save_split(X_val_scaled, y_val, device, "val", OUTPUT_DIR)
    test_path = save_split(X_test_scaled, y_test, device, "test", OUTPUT_DIR)

    # Save scaler metadata
    with open(os.path.join(device_out, "metadata.json"), "w") as f:
        json.dump({
            "source_file": source_file,
            "label_column": label_col,
            "num_features": int(X.shape[1]),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "class_distribution_train": y_train.value_counts().to_dict(),
            "class_distribution_val": y_val.value_counts().to_dict(),
            "class_distribution_test": y_test.value_counts().to_dict()
        }, f, indent=4)

    summary.append({
        "device": device,
        "source_file": source_file,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test))
    })

# Save overall summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "preprocessing_summary.csv"), index=False)

print("Phase 4 preprocessing completed successfully.")
print(summary_df)
