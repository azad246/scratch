import os
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
PROJECT_ROOT     = "."
OUTPUT_DIR       = "outputs/phase11_final"
FINAL_BUNDLE_DIR = "final_bundle"

PHASE10_SUMMARY    = "outputs/phase10_evaluation/phase10_summary.csv"
PHASE5_RESULTS     = "outputs/phase5_autoencoder/phase5_results.csv"
PREPROCESS_SUMMARY = "outputs/final_processed/preprocessing_summary.csv"   # corrected path

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"),  exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tables"), exist_ok=True)
os.makedirs(FINAL_BUNDLE_DIR, exist_ok=True)

sns.set(style="whitegrid")


# =========================
# HELPERS
# =========================
def safe_copy(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def load_csv_if_exists(path):
    if os.path.exists(path):
        print(f"  [LOAD] {path}")
        return pd.read_csv(path)
    print(f"  [SKIP] Not found: {path}")
    return pd.DataFrame()


def plot_metric_comparison(df, metric, save_path, title):
    if df.empty or metric not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    hue_col = "method" if "method" in df.columns else None
    sns.barplot(data=df, x="device", y=metric, hue=hue_col)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [PLOT] {save_path}")


def write_text_report(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# LOAD DATA
# =========================
print("\nLoading data...")
phase10_df = load_csv_if_exists(PHASE10_SUMMARY)
phase5_df  = load_csv_if_exists(PHASE5_RESULTS)
prep_df    = load_csv_if_exists(PREPROCESS_SUMMARY)

# =========================
# SAVE TABLES
# =========================
print("\nSaving clean tables...")
if not phase10_df.empty:
    p = os.path.join(OUTPUT_DIR, "tables", "phase10_summary_clean.csv")
    phase10_df.to_csv(p, index=False)
    print(f"  [TABLE] {p}")

if not phase5_df.empty:
    p = os.path.join(OUTPUT_DIR, "tables", "phase5_results_clean.csv")
    phase5_df.to_csv(p, index=False)
    print(f"  [TABLE] {p}")

if not prep_df.empty:
    p = os.path.join(OUTPUT_DIR, "tables", "preprocessing_summary_clean.csv")
    prep_df.to_csv(p, index=False)
    print(f"  [TABLE] {p}")

# =========================
# PLOTS
# =========================
print("\nGenerating plots...")
plot_metric_comparison(
    phase10_df, "accuracy",
    os.path.join(OUTPUT_DIR, "plots", "accuracy_comparison.png"),
    "Accuracy Comparison Across Devices",
)

plot_metric_comparison(
    phase10_df, "f1",
    os.path.join(OUTPUT_DIR, "plots", "f1_comparison.png"),
    "F1 Score Comparison Across Devices",
)

if not phase10_df.empty and "false_positive_rate" in phase10_df.columns:
    plot_metric_comparison(
        phase10_df, "false_positive_rate",
        os.path.join(OUTPUT_DIR, "plots", "fpr_comparison.png"),
        "False Positive Rate (lower is better)",
    )

if not phase10_df.empty and "epsilon" in phase10_df.columns:
    eps_df = phase10_df.dropna(subset=["epsilon"])
    if not eps_df.empty:
        plt.figure(figsize=(10, 5))
        hue_col = "method" if "method" in eps_df.columns else None
        sns.barplot(data=eps_df, x="device", y="epsilon", hue=hue_col, color="coral")
        plt.title("DP Privacy Budget (epsilon) per Device  [lower = more private]")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, "plots", "epsilon_comparison.png")
        plt.savefig(p, dpi=300)
        plt.close()
        print(f"  [PLOT] {p}")

# =========================
# FINAL REPORT
# =========================
print("\nWriting final report...")
report_lines = [
    "PQC-IoT Sentinel — Final Project Report\n",
    "=" * 50 + "\n",
    "\nProject: Post-Quantum Cryptography for IoT Intrusion Detection\n",
    "Pipeline: EDA → Preprocessing → Autoencoder → Federated Learning (PQC + DP)\n",
]

if not prep_df.empty:
    report_lines.append("\n--- Preprocessing Summary ---\n")
    report_lines.append(prep_df.to_string(index=False))
    report_lines.append("\n")

if not phase5_df.empty:
    report_lines.append("\n--- Autoencoder Baseline (Phase 5) ---\n")
    report_lines.append(phase5_df.to_string(index=False))
    report_lines.append("\n")

if not phase10_df.empty:
    report_lines.append("\n--- Phase 10 Evaluation Summary ---\n")
    cols = [c for c in ["device","method","accuracy","false_positive_rate","epsilon"] if c in phase10_df.columns]
    report_lines.append(phase10_df[cols].to_string(index=False))
    report_lines.append("\n")
    report_lines.append("\nNote: Validation data is benign-only (anomaly detection baseline).\n")
    report_lines.append("      Accuracy reflects correct benign classification rate.\n")
    report_lines.append("      False Positive Rate = fraction of benign traffic incorrectly flagged.\n")
    report_lines.append("      Epsilon = DP privacy budget consumed (lower = stronger privacy).\n")

report_path = os.path.join(OUTPUT_DIR, "final_report.txt")
write_text_report(report_path, "\n".join(report_lines))
print(f"  [REPORT] {report_path}")

# =========================
# ORGANIZE FINAL BUNDLE
# =========================
print("\nBuilding final bundle...")
bundle_structure = [
    ("outputs/phase11_final",                 os.path.join(FINAL_BUNDLE_DIR, "results")),
    ("src",                                   os.path.join(FINAL_BUNDLE_DIR, "src_backup")),
    ("outputs/phase10_evaluation/plots",      os.path.join(FINAL_BUNDLE_DIR, "evaluation_plots")),
    ("outputs/pqc_logs",                      os.path.join(FINAL_BUNDLE_DIR, "pqc_logs")),
    ("outputs/dp_logs",                       os.path.join(FINAL_BUNDLE_DIR, "dp_logs")),
]

for src_dir, dst_dir in bundle_structure:
    if os.path.exists(src_dir):
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"  [BUNDLE] {src_dir} -> {dst_dir}")
    else:
        print(f"  [SKIP]   {src_dir} not found")

# Manifest
manifest = {
    "project": "PQC-IoT Sentinel",
    "outputs": {
        "phase5_results":    PHASE5_RESULTS     if os.path.exists(PHASE5_RESULTS)     else None,
        "phase10_summary":   PHASE10_SUMMARY    if os.path.exists(PHASE10_SUMMARY)    else None,
        "preprocessing":     PREPROCESS_SUMMARY if os.path.exists(PREPROCESS_SUMMARY) else None,
    },
    "final_bundle_dir": FINAL_BUNDLE_DIR,
    "final_report":     report_path,
}
manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)
print(f"  [MANIFEST] {manifest_path}")

# =========================
# DONE
# =========================
print("\n" + "=" * 60)
print("Phase 11 packaging completed successfully.")
print(f"Final report : {os.path.abspath(report_path)}")
print(f"Bundle dir   : {os.path.abspath(FINAL_BUNDLE_DIR)}/")
print("=" * 60)
