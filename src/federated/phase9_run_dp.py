"""
Phase 9: Differentially-Private Federated Learning — In-Process Simulation
Uses Opacus PrivacyEngine on each client. Implements FedAvg aggregation
directly (no Ray, no gRPC, no flower-superlink — works on Python 3.13).

Each round:
  1. Server sends global weights to each client.
  2. Each client trains locally with Opacus DP (Gaussian noise on gradients).
  3. Each client reports its current privacy budget (epsilon, delta).
  4. Server FedAvg-aggregates the weight updates.
  5. Metrics + epsilon budgets logged to outputs/dp_logs/.
"""

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.federated.fl_client_dp import SimpleMLP, create_dp_client, get_parameters, set_parameters

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.join("outputs", "final_processed")
NUM_ROUNDS      = 5
NOISE_MULT      = 0.8   # Opacus noise multiplier
MAX_GRAD_NORM   = 1.0   # Opacus clipping norm
DELTA           = 1e-5  # Target delta for (epsilon, delta)-DP
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIENTS = [
    ("danmini_doorbell",          "danmini"),
    ("ecobee_thermostat",         "ecobee"),
    ("philips_baby_monitor",      "philips"),
    ("provision_security_camera", "provision"),
    ("samsung_webcam",            "samsung"),
]


# ── FedAvg ────────────────────────────────────────────────────────────────────
def fedavg(weights_list: List[List[np.ndarray]], num_samples: List[int]) -> List[np.ndarray]:
    total = sum(num_samples)
    return [
        sum(w * (n / total) for w, n in zip(layer, num_samples))
        for layer in zip(*weights_list)
    ]


# ── One federation round ──────────────────────────────────────────────────────
def run_round(round_num, global_weights, clients):
    print(f"\n{'---'*20}")
    print(f"  Round {round_num}/{NUM_ROUNDS}")
    print(f"{'---'*20}")

    all_weights, all_samples, round_metrics = [], [], []

    for client_name, dp_client in clients:
        weights, n_samples, epsilon = dp_client.fit(global_weights)
        loss, n_eval, accuracy, eps_eval = dp_client.evaluate(weights)

        print(f"  [{client_name:10s}]  samples={n_samples:6d}  "
              f"loss={loss:8.4f}  acc={accuracy:.4f}  "
              f"eps(e={epsilon:.3f}, d={DELTA:.0e})")

        all_weights.append(weights)
        all_samples.append(n_samples)
        round_metrics.append({
            "client":   client_name,
            "round":    round_num,
            "loss":     loss,
            "accuracy": accuracy,
            "epsilon":  epsilon,
            "delta":    DELTA,
            "samples":  n_samples,
        })

    new_global = fedavg(all_weights, all_samples)

    total      = sum(all_samples)
    agg_acc    = sum(m["accuracy"] * m["samples"] for m in round_metrics) / total
    agg_eps    = max(m["epsilon"] for m in round_metrics)   # worst-case budget

    print(f"\n  [SERVER] Aggregated acc={agg_acc:.4f}  "
          f"worst-case epsilon={agg_eps:.3f}")

    return new_global, round_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("outputs/dp_logs", exist_ok=True)

    print("=" * 60)
    print(" Differentially-Private Federated Learning")
    print(f" Clients         : {len(CLIENTS)}")
    print(f" Rounds          : {NUM_ROUNDS}")
    print(f" Noise multiplier: {NOISE_MULT}")
    print(f" Max grad norm   : {MAX_GRAD_NORM}")
    print(f" Delta           : {DELTA:.0e}")
    print(f" Device          : {DEVICE}")
    print("=" * 60)

    # ── Load clients ──────────────────────────────────────────────────────────
    print("\nLoading client data and initializing DP engines...")
    clients = []
    input_dim = None
    for folder, name in CLIENTS:
        data_path = os.path.join(BASE_DIR, folder, "train.csv")
        if not os.path.exists(data_path):
            print(f"  [ERROR] Missing: {data_path}")
            sys.exit(1)
        dp_client = create_dp_client(data_path, name, NOISE_MULT, MAX_GRAD_NORM)
        n_train = len(dp_client.trainloader.dataset)
        n_val   = len(dp_client.valloader.dataset)
        if input_dim is None:
            # Unwrap GradSampleModule to get original layer
            inner = dp_client.model._module if hasattr(dp_client.model, "_module") else dp_client.model
            input_dim = inner.net[0].in_features
        print(f"  [{name:10s}] loaded  train={n_train}  val={n_val}")
        clients.append((name, dp_client))

    # ── Global model ──────────────────────────────────────────────────────────
    global_model   = SimpleMLP(input_dim=input_dim).to(DEVICE)
    global_weights = get_parameters(global_model)

    # ── Federation loop ───────────────────────────────────────────────────────
    all_metrics = []
    for r in range(1, NUM_ROUNDS + 1):
        global_weights, round_metrics = run_round(r, global_weights, clients)
        all_metrics.extend(round_metrics)

    # ── Save results ──────────────────────────────────────────────────────────
    log_path = "outputs/dp_logs/dp_federation_metrics.json"
    with open(log_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\n" + "=" * 60)
    print(" DP Federation complete!")
    print(f" Metrics saved to: {log_path}")
    print(" Per-client epsilon budgets: outputs/dp_logs/<client>_epsilon.json")
    print("=" * 60)
