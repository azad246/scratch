import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

# =========================
# CONFIG
# =========================
OUTPUT_DIR       = "outputs/phase10_evaluation"
PHASE5_PRED_DIR  = "outputs/phase5_autoencoder/predictions"
PHASE6_PRED_DIR  = "outputs/phase6_validation"
DP_LOG_DIR       = "outputs/dp_logs"

DEVICE_FOLDERS = [
    "danmini_doorbell",
    "ecobee_thermostat",
    "philips_baby_monitor",
    "provision_security_camera",
    "samsung_webcam",
]

# Short names used by the DP log files (danmini_epsilon.json, etc.)
DEVICE_SHORT_NAME = {
    "danmini_doorbell":          "danmini",
    "ecobee_thermostat":         "ecobee",
    "philips_baby_monitor":      "philips",
    "provision_security_camera": "provision",
    "samsung_webcam":            "samsung",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


# =========================
# HELPERS
# =========================
def load_predictions(pred_path):
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    return pd.read_csv(pred_path)


def compute_metrics_from_df(df):
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

    unique_true = np.unique(y_true)
    single_class = len(unique_true) == 1

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "single_class_eval": single_class,
    }

    # For benign-only validation data, report false positive rate instead
    if single_class and unique_true[0] == 0:
        # All true labels are 0 (benign); pred=1 means false alarm
        false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
        true_negatives  = int(np.sum((y_true == 0) & (y_pred == 0)))
        fpr = false_positives / len(y_true) if len(y_true) > 0 else 0.0
        metrics["false_positive_rate"] = fpr
        metrics["false_positives"]     = false_positives
        metrics["true_negatives"]      = true_negatives

    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return metrics, cm, report


def save_confusion_matrix_plot(cm, device_name, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {device_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_dp_epsilon(device_name):
    """
    DP log files are saved as <short_name>_epsilon.json
    e.g.  danmini_epsilon.json  (not  danmini_doorbell.json)
    Fall back to full device name if short name not found.
    """
    short = DEVICE_SHORT_NAME.get(device_name, device_name)
    candidates = [
        os.path.join(DP_LOG_DIR, f"{short}_epsilon.json"),   # e.g. danmini_epsilon.json
        os.path.join(DP_LOG_DIR, f"{device_name}.json"),     # fallback: full name
        os.path.join(DP_LOG_DIR, f"{device_name}_epsilon.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if "latest_epsilon" in data:
                return float(data["latest_epsilon"])
            if "epsilon_history" in data and data["epsilon_history"]:
                return float(data["epsilon_history"][-1])
    return None


# =========================
# MAIN EVALUATION LOOP
# =========================
results = []

for device in DEVICE_FOLDERS:
    pred_path_phase5 = os.path.join(PHASE5_PRED_DIR, f"{device}_predictions.csv")
    pred_path_phase6 = os.path.join(PHASE6_PRED_DIR, f"{device}_predictions.csv")

    if os.path.exists(pred_path_phase6):
        chosen_path = pred_path_phase6
        method_name = "local_validation"
    elif os.path.exists(pred_path_phase5):
        chosen_path = pred_path_phase5
        method_name = "autoencoder_baseline"
    else:
        print(f"[SKIP] {device}: no predictions file found.")
        continue

    print(f"[OK]   {device} -> {method_name}")
    df = load_predictions(chosen_path)
    metrics, cm, report = compute_metrics_from_df(df)
    epsilon = load_dp_epsilon(device)

    # Save JSON classification report
    report_path = os.path.join(OUTPUT_DIR, "reports", f"{device}_{method_name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "reports", f"{device}_{method_name}_confusion_matrix.csv"))

    # Save confusion matrix plot
    cm_plot_path = os.path.join(OUTPUT_DIR, "plots", f"{device}_{method_name}_cm.png")
    save_confusion_matrix_plot(cm, device, cm_plot_path)

    row = {
        "device":      device,
        "method":      method_name,
        "source_file": chosen_path,
        "epsilon":     epsilon,
    }
    row.update(metrics)
    results.append(row)

# =========================
# SUMMARY TABLE
# =========================
summary_df = pd.DataFrame(results)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "phase10_summary.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "phase10_summary.json"), "w") as f:
    json.dump(results, f, indent=4)

# =========================
# COMPARISON PLOTS
# =========================
if not summary_df.empty:
    # F1 comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary_df, x="device", y="f1", hue="method")
    plt.title("F1 Score Comparison Across Devices")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "f1_comparison.png"), dpi=300)
    plt.close()

    # Accuracy comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary_df, x="device", y="accuracy", hue="method")
    plt.title("Accuracy Comparison Across Devices")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "accuracy_comparison.png"), dpi=300)
    plt.close()

    # Epsilon budget per device (if available)
    eps_df = summary_df.dropna(subset=["epsilon"])
    if not eps_df.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=eps_df, x="device", y="epsilon", color="coral")
        plt.title("DP Privacy Budget (epsilon) per Device  [lower = more private]")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "epsilon_budget.png"), dpi=300)
        plt.close()

# =========================
# PRINT RESULTS
# =========================
print("\nPhase 10 evaluation completed successfully.")
print("=" * 80)
display_cols = ["device", "method", "accuracy", "precision", "recall", "f1",
                "false_positive_rate", "epsilon"]
print(summary_df[[c for c in display_cols if c in summary_df.columns]].to_string(index=False))

# Interpret results
if "single_class_eval" in summary_df.columns and summary_df["single_class_eval"].any():
    print("\n[NOTE] Validation data is benign-only (single class).")
    print("       F1/Precision/Recall = 0 is EXPECTED — no attack samples in val set.")
    print("       Key metrics: Accuracy = correct benign classification rate")
    print("                    False Positive Rate = fraction of benign flagged as attack")
print("=" * 80)
print(f"Reports : {os.path.abspath(os.path.join(OUTPUT_DIR, 'reports'))}")
print(f"Plots   : {os.path.abspath(os.path.join(OUTPUT_DIR, 'plots'))}")
