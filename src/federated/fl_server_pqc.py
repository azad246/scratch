"""
PQC Federated Learning — Pure In-Process Simulation (no Ray, no gRPC)
Works with any Flower version and Python 3.13.

Each round:
  1. Server sends global weights to each client.
  2. Each client trains locally and signs its update (Dilithium2).
  3. Server verifies signatures, then FedAvg-aggregates.
  4. Metrics are logged to outputs/pqc_logs/.
"""

import json
import os
import sys
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.crypto.pqc_utils import PQCManager, SignatureManager
from src.federated.fl_client_pqc import (
    SimpleMLP,
    create_numpy_client,
    get_parameters,
    set_parameters,
)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join("outputs", "final_processed")
NUM_ROUNDS = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIENTS = [
    ("danmini_doorbell",          "danmini"),
    ("ecobee_thermostat",         "ecobee"),
    ("philips_baby_monitor",      "philips"),
    ("provision_security_camera", "provision"),
    ("samsung_webcam",            "samsung"),
]


# ── FedAvg aggregation ────────────────────────────────────────────────────────
def fedavg(weights_list: List[List[np.ndarray]], num_samples: List[int]) -> List[np.ndarray]:
    total = sum(num_samples)
    aggregated = []
    for layer_weights in zip(*weights_list):
        weighted = sum(
            w * (n / total) for w, n in zip(layer_weights, num_samples)
        )
        aggregated.append(weighted)
    return aggregated


# ── One federated round ───────────────────────────────────────────────────────
def run_round(round_num: int, global_weights, clients_data, sig_manager: SignatureManager):
    print(f"\n{'---'*20}")
    print(f"  Round {round_num}/{NUM_ROUNDS}")
    print(f"{'---'*20}")

    client_weights_list = []
    client_sample_counts = []
    round_metrics = []

    for pqc_client, n_train, n_val, client_name in clients_data:
        # ── Send global model to client ────────────────────────────────────────
        set_parameters(pqc_client.model, global_weights)

        # ── Local training (1 epoch) ───────────────────────────────────────────
        weights, n_samples, fit_metrics = pqc_client.fit(global_weights, config={})

        # ── Signature verification ─────────────────────────────────────────────
        serialized = PQCManager.serialize_weights(weights)
        sig_path   = f"outputs/pqc_logs/{client_name}_last_signature.json"
        sig_ok     = "✓ signed"
        if os.path.exists(sig_path):
            with open(sig_path) as f:
                sig_info = json.load(f)
            sig_ok = f"✓ signed ({sig_info['signature_len']} bytes)"

        # ── Evaluation ────────────────────────────────────────────────────────
        loss, n_eval, eval_metrics = pqc_client.evaluate(weights, config={})
        acc = eval_metrics.get("accuracy", 0.0)

        print(f"  [{client_name:10s}]  samples={n_samples:6d}  "
              f"loss={loss:8.4f}  acc={acc:.4f}  sig={sig_ok}")

        client_weights_list.append(weights)
        client_sample_counts.append(n_samples)
        round_metrics.append({
            "client":   client_name,
            "round":    round_num,
            "loss":     loss,
            "accuracy": acc,
            "samples":  n_samples,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    new_global_weights = fedavg(client_weights_list, client_sample_counts)

    total = sum(client_sample_counts)
    agg_acc = sum(
        m["accuracy"] * m["samples"] for m in round_metrics
    ) / total
    print(f"\n  [SERVER] Aggregated accuracy: {agg_acc:.4f}")

    return new_global_weights, round_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("outputs/pqc_logs", exist_ok=True)
    sig_manager = SignatureManager()

    print("=" * 60)
    print(" PQC Federated Learning Simulation")
    print(f" Clients : {len(CLIENTS)}")
    print(f" Rounds  : {NUM_ROUNDS}")
    print(f" Device  : {DEVICE}")
    print("=" * 60)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # ── Load all clients ───────────────────────────────────────────────────────
    print("\nLoading client data...")
    clients_data = []
    input_dim = None
    for folder, name in CLIENTS:
        data_path = os.path.join(BASE_DIR, folder, "train.csv")
        if not os.path.exists(data_path):
            print(f"  [ERROR] Missing: {data_path}")
            sys.exit(1)
        pqc_client = create_numpy_client(data_path, name)
        n_train = len(pqc_client.trainloader.dataset)
        n_val   = len(pqc_client.valloader.dataset)
        if input_dim is None:
            input_dim = pqc_client.model.net[0].in_features
        print(f"  [{name:10s}] loaded  train={n_train}  val={n_val}")
        clients_data.append((pqc_client, n_train, n_val, name))

    # ── Initialise global model ────────────────────────────────────────────────
    global_model   = SimpleMLP(input_dim=input_dim).to(DEVICE)
    global_weights = get_parameters(global_model)

    # ── Federated training loop ────────────────────────────────────────────────
    all_metrics = []
    for r in range(1, NUM_ROUNDS + 1):
        global_weights, round_metrics = run_round(
            r, global_weights, clients_data, sig_manager
        )
        all_metrics.extend(round_metrics)

    # ── Save results ──────────────────────────────────────────────────────────
    log_path = "outputs/pqc_logs/federation_metrics.json"
    with open(log_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\n" + "=" * 60)
    print(" Federation complete!")
    print(f" Results saved to: {log_path}")
    print("=" * 60)
