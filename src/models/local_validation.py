import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.join("outputs", "final_processed")
MODEL_DIR = os.path.join("outputs", "phase5_autoencoder", "models")
OUTPUT_DIR = os.path.join("outputs", "phase6_validation")

DEVICE_FOLDERS = [
    "danmini_doorbell",
    "ecobee_thermostat",
    "philips_baby_monitor",
    "provision_security_camera",
    "samsung_webcam"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)


# =========================
# HELPERS
# =========================
def load_split(device_name, split_name):
    path = os.path.join(BASE_DIR, device_name, f"{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def split_xy(df):
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


def reconstruction_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=1)


def evaluate_device(device_name):
    model_path = os.path.join(MODEL_DIR, f"{device_name}_autoencoder.keras")
    threshold_path = os.path.join(MODEL_DIR, f"{device_name}_threshold.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Missing threshold file: {threshold_path}")

    with open(threshold_path, "r") as f:
        th_data = json.load(f)
    threshold = th_data["threshold"]

    model = tf.keras.models.load_model(model_path)

    test_df = load_split(device_name, "test")
    X_test, y_test = split_xy(test_df)

    errors = reconstruction_errors(model, X_test)
    y_pred = (errors > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    pred_df = pd.DataFrame({
        "true_label": y_test,
        "pred_label": y_pred,
        "reconstruction_error": errors
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"{device_name}_predictions.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "reports", f"{device_name}_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    return {
        "device": device_name,
        "threshold": threshold,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_test))
    }


# =========================
# MAIN
# =========================
results = []

for device in DEVICE_FOLDERS:
    print(f"Evaluating {device} ...")
    try:
        res = evaluate_device(device)
        results.append(res)
    except Exception as e:
        print(f"  [ERROR] Failed to evaluate {device}: {e}")

if results:
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "phase6_local_validation_summary.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "phase6_local_validation_summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Phase 6 completed successfully ===")
    print(summary_df)
else:
    print("\n[ERROR] No evaluation results generated.")
