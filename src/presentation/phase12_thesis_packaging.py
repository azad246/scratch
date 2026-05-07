import os
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
PROJECT_ROOT = "."
OUTPUT_DIR   = "outputs/phase12_thesis"
BUNDLE_DIR   = "thesis_bundle"

PHASE10_SUMMARY    = "outputs/phase10_evaluation/phase10_summary.csv"
PHASE11_REPORT     = "outputs/phase11_final/final_report.txt"
PHASE5_RESULTS     = "outputs/phase5_autoencoder/phase5_results.csv"
PREPROCESS_SUMMARY = "outputs/final_processed/preprocessing_summary.csv"  # corrected path

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"),  exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tables"), exist_ok=True)
os.makedirs(BUNDLE_DIR, exist_ok=True)

sns.set(style="whitegrid")


# =========================
# HELPERS
# =========================
def load_csv(path):
    if os.path.exists(path):
        print(f"  [LOAD] {path}")
        return pd.read_csv(path)
    print(f"  [SKIP] Not found: {path}")
    return pd.DataFrame()


def save_plot(df, metric, title, save_path):
    if df.empty or metric not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    hue_col = "method" if "method" in df.columns else None
    sns.barplot(data=df, x="device", y=metric, hue=hue_col)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {save_path}")


def safe_copy(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# LOAD RESULTS
# =========================
print("\nLoading data...")
phase10_df = load_csv(PHASE10_SUMMARY)
phase5_df  = load_csv(PHASE5_RESULTS)
prep_df    = load_csv(PREPROCESS_SUMMARY)

# =========================
# TABLES
# =========================
print("\nSaving tables...")
if not phase10_df.empty:
    p = os.path.join(OUTPUT_DIR, "tables", "phase10_summary.csv")
    phase10_df.to_csv(p, index=False)
    print(f"  [TABLE] {p}")

if not phase5_df.empty:
    p = os.path.join(OUTPUT_DIR, "tables", "phase5_results.csv")
    phase5_df.to_csv(p, index=False)
    print(f"  [TABLE] {p}")

if not prep_df.empty:
    p = os.path.join(OUTPUT_DIR, "tables", "preprocessing_summary.csv")
    prep_df.to_csv(p, index=False)
    print(f"  [TABLE] {p}")

# =========================
# PLOTS
# =========================
print("\nGenerating plots...")
save_plot(
    phase10_df, "accuracy",
    "Accuracy Comparison Across Devices",
    os.path.join(OUTPUT_DIR, "plots", "accuracy_comparison.png"),
)

save_plot(
    phase10_df, "f1",
    "F1 Score Comparison Across Devices",
    os.path.join(OUTPUT_DIR, "plots", "f1_comparison.png"),
)

if not phase10_df.empty and "false_positive_rate" in phase10_df.columns:
    save_plot(
        phase10_df, "false_positive_rate",
        "False Positive Rate Across Devices  (lower = better)",
        os.path.join(OUTPUT_DIR, "plots", "fpr_comparison.png"),
    )

if not phase10_df.empty and "epsilon" in phase10_df.columns:
    eps_df = phase10_df.dropna(subset=["epsilon"])
    if not eps_df.empty:
        plt.figure(figsize=(10, 5))
        hue_col = "method" if "method" in eps_df.columns else None
        sns.barplot(data=eps_df, x="device", y="epsilon", hue=hue_col,
                    palette="flare")
        plt.title("DP Privacy Budget (epsilon) Across Devices  [lower = more private]")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, "plots", "epsilon_comparison.png")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [PLOT] {p}")

if not phase5_df.empty and "accuracy" in phase5_df.columns:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=phase5_df, x="device", y="accuracy", palette="Blues_d")
    plt.title("Baseline Autoencoder Accuracy per Device")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "plots", "baseline_accuracy.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {p}")

# =========================
# THESIS REPORT
# =========================
print("\nWriting thesis report...")
report = [
    "PQC-IoT Sentinel — Thesis Package\n",
    "=" * 50 + "\n",
    "\nProject: Post-Quantum Cryptography for Federated IoT Intrusion Detection\n",
    "Pipeline:\n",
    "  Phase 3 : Exploratory Data Analysis (5 IoT devices)\n",
    "  Phase 4 : Preprocessing & Feature Engineering\n",
    "  Phase 5 : Local Autoencoder Baseline Training\n",
    "  Phase 6 : Local Validation & Anomaly Detection\n",
    "  Phase 7 : Federated Learning (FedAvg)\n",
    "  Phase 7b: PQC-Secured Federation (Dilithium2 Signatures)\n",
    "  Phase 9 : Differentially-Private Federation (Opacus)\n",
    "  Phase 10: Evaluation & Metrics\n",
    "  Phase 11: Final Packaging\n",
    "  Phase 12: Thesis Packaging (this output)\n",
]

if not prep_df.empty:
    report.append("\n--- Dataset Preprocessing Summary ---\n")
    report.append(prep_df.to_string(index=False))
    report.append("\n")

if not phase5_df.empty:
    report.append("\n--- Autoencoder Baseline Results (Phase 5) ---\n")
    report.append(phase5_df.to_string(index=False))
    report.append("\n")

if not phase10_df.empty:
    report.append("\n--- Final Evaluation Summary (Phase 10) ---\n")
    cols = [c for c in ["device", "method", "accuracy", "false_positive_rate", "epsilon"]
            if c in phase10_df.columns]
    report.append(phase10_df[cols].to_string(index=False))
    report.append("\n")
    report.append("\nInterpretation:\n")
    report.append("  - Validation set is benign-only -> F1/Precision/Recall = 0 (expected).\n")
    report.append("  - Accuracy reflects correct benign classification (~93-96%).\n")
    report.append("  - False Positive Rate: benign traffic incorrectly flagged as attack (~5%).\n")
    report.append("  - Epsilon: DP privacy budget consumed per client (lower = stronger privacy).\n")
    report.append("  - All client updates signed with Dilithium2 (2420-byte PQC signatures).\n")

# Embed phase 11 final report if available
if os.path.exists(PHASE11_REPORT):
    report.append("\n--- Phase 11 Final Report (embedded) ---\n")
    with open(PHASE11_REPORT, encoding="utf-8") as f:
        report.append(f.read())

report_path = os.path.join(OUTPUT_DIR, "thesis_report.txt")
write_text(report_path, "\n".join(report))
print(f"  [REPORT] {report_path}")

# =========================
# BUNDLE
# =========================
print("\nBuilding thesis bundle...")
bundle_items = [
    ("outputs/phase12_thesis",            os.path.join(BUNDLE_DIR, "results")),
    ("outputs/phase10_evaluation/plots",  os.path.join(BUNDLE_DIR, "evaluation_plots")),
    ("outputs/phase11_final",             os.path.join(BUNDLE_DIR, "phase11_final")),
    ("outputs/pqc_logs",                  os.path.join(BUNDLE_DIR, "pqc_logs")),
    ("outputs/dp_logs",                   os.path.join(BUNDLE_DIR, "dp_logs")),
    ("src",                               os.path.join(BUNDLE_DIR, "src_backup")),
]

for src_dir, dst_dir in bundle_items:
    if os.path.exists(src_dir):
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"  [BUNDLE] {src_dir} -> {dst_dir}")
    else:
        print(f"  [SKIP]   {src_dir} not found")

manifest = {
    "project": "PQC-IoT Sentinel",
    "bundle_dir": BUNDLE_DIR,
    "artifacts": {
        "phase10_summary":      PHASE10_SUMMARY    if os.path.exists(PHASE10_SUMMARY)    else None,
        "phase11_report":       PHASE11_REPORT     if os.path.exists(PHASE11_REPORT)     else None,
        "preprocessing_summary": PREPROCESS_SUMMARY if os.path.exists(PREPROCESS_SUMMARY) else None,
        "phase5_results":       PHASE5_RESULTS     if os.path.exists(PHASE5_RESULTS)     else None,
    },
}

manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)
print(f"  [MANIFEST] {manifest_path}")

# =========================
# DONE
# =========================
print("\n" + "=" * 60)
print("Phase 12 thesis packaging completed successfully.")
print(f"Thesis report : {os.path.abspath(report_path)}")
print(f"Bundle dir    : {os.path.abspath(BUNDLE_DIR)}/")
print("=" * 60)
