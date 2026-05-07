import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =========================
# CONFIG
# =========================
# Correcting BASE_DIR to point to where phase 4 actually output the data
BASE_DIR    = os.path.join("outputs", "final_processed")
OUTPUT_DIR  = os.path.join("outputs", "phase5_autoencoder")
DEVICE_FOLDERS = [
    "danmini_doorbell",
    "ecobee_thermostat",
    "philips_baby_monitor",
    "provision_security_camera",
    "samsung_webcam"
]

RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 40
THRESHOLD_PERCENTILE = 95

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)


# =========================
# HELPERS
# =========================
def load_split(device_name, split_name):
    path = os.path.join(BASE_DIR, device_name, f"{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def split_xy(df):
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the processed CSV.")
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y


def build_autoencoder(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def reconstruction_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - X_pred), axis=1)
    return errors


def evaluate_predictions(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    }


# =========================
# MAIN LOOP
# =========================
all_results = []

for device in DEVICE_FOLDERS:
    print(f"\n=== Training autoencoder for {device} ===")

    try:
        train_df = load_split(device, "train")
        val_df = load_split(device, "val")
        test_df = load_split(device, "test")

        X_train, y_train = split_xy(train_df)
        X_val, y_val = split_xy(val_df)
        X_test, y_test = split_xy(test_df)

        # Train only on benign rows
        X_train_benign = X_train[y_train == 0]
        X_val_benign = X_val[y_val == 0]

        if len(X_train_benign) == 0:
            print(f"  [ERROR] No benign samples found in train split for {device}")
            continue
        if len(X_val_benign) == 0:
            print(f"  [ERROR] No benign samples found in val split for {device}")
            continue

        input_dim = X_train.shape[1]
        model = build_autoencoder(input_dim)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            X_train_benign,
            X_train_benign,
            validation_data=(X_val_benign, X_val_benign),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=callbacks,
            verbose=1
        )

        # Threshold from benign validation reconstruction errors
        val_errors = reconstruction_errors(model, X_val_benign)
        threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))

        # Evaluate on test set
        test_errors = reconstruction_errors(model, X_test)
        y_pred = (test_errors > threshold).astype(int)

        metrics = evaluate_predictions(y_test, y_pred, test_errors)
        cm = confusion_matrix(y_test, y_pred)

        # Save model
        model_path = os.path.join(OUTPUT_DIR, "models", f"{device}_autoencoder.keras")
        model.save(model_path)

        # Save threshold
        threshold_path = os.path.join(OUTPUT_DIR, "models", f"{device}_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({
                "device": device,
                "threshold_percentile": THRESHOLD_PERCENTILE,
                "threshold": threshold
            }, f, indent=4)

        # Save predictions
        pred_df = pd.DataFrame({
            "true_label": y_test,
            "pred_label": y_pred,
            "reconstruction_error": test_errors
        })
        pred_path = os.path.join(OUTPUT_DIR, "predictions", f"{device}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        # Save history
        hist_path = os.path.join(OUTPUT_DIR, "models", f"{device}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history.history, f, indent=4)

        # Save metrics
        result = {
            "device": device,
            "model_path": model_path,
            "threshold_path": threshold_path,
            "prediction_path": pred_path,
            "threshold": threshold,
            "confusion_matrix": cm.tolist(),
            **metrics
        }
        all_results.append(result)

    except Exception as e:
        print(f"  [ERROR] Failed to process {device}: {e}")

# Overall summary
if all_results:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "phase5_results.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "phase5_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n=== Phase 5 completed successfully ===")
    print(results_df)
else:
    print("\n[ERROR] No results were generated.")
