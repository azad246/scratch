"""
Phase 7 PQC Federation Runner
Launches fl_server_pqc.py and all 5 fl_client_pqc.py instances in parallel.
"""

import os
import subprocess
import sys
import time

BASE_DIR     = os.path.join("outputs", "final_processed")
SERVER_SCRIPT = os.path.join("src", "federated", "fl_server_pqc.py")
CLIENT_SCRIPT = os.path.join("src", "federated", "fl_client_pqc.py")

CLIENTS = [
    ("danmini_doorbell",          "danmini"),
    ("ecobee_thermostat",         "ecobee"),
    ("philips_baby_monitor",      "philips"),
    ("provision_security_camera", "provision"),
    ("samsung_webcam",            "samsung"),
]

def run_server():
    print("Launching PQC server...")
    # IMPORTANT: do NOT pipe stdout/stderr — Flower's start_server() exits
    # immediately when it detects a non-TTY pipe. Inherit the parent terminal.
    return subprocess.Popen(
        [sys.executable, SERVER_SCRIPT],
    )

def run_client(device_folder, client_name):
    data_path = os.path.join(BASE_DIR, device_folder, "train.csv")
    if not os.path.exists(data_path):
        print(f"  [ERROR] Missing: {data_path}")
        return None
    print(f"  Launching client: {client_name}  ({data_path})")
    return subprocess.Popen(
        [
            sys.executable, CLIENT_SCRIPT,
            "--data_path",   data_path,
            "--client_name", client_name,
            "--server_address", "127.0.0.1:8080",
        ],
    )

def stream_output(label, proc):
    """Print proc output with a label prefix (non-blocking)."""
    for line in proc.stdout:
        print(f"[{label}] {line}", end="")

if __name__ == "__main__":
    os.makedirs("outputs/pqc_logs", exist_ok=True)

    # ── Start server ──────────────────────────────────────────────────────────
    server_proc = run_server()
    print("Waiting 12 seconds for server to be ready...")
    time.sleep(12)

    if server_proc.poll() is not None:
        print("[ERROR] Server exited early. Check for port conflicts.")
        out, _ = server_proc.communicate()
        print(out)
        sys.exit(1)

    # ── Start all clients ─────────────────────────────────────────────────────
    print("\nLaunching all 5 PQC clients...")
    client_procs = []
    for folder, name in CLIENTS:
        p = run_client(folder, name)
        if p:
            client_procs.append((name, p))
        time.sleep(1)   # slight stagger to avoid race on gRPC connect

    if not client_procs:
        print("[ERROR] No clients launched. Aborting.")
        server_proc.terminate()
        sys.exit(1)

    # ── Wait for clients ──────────────────────────────────────────────────────
    print(f"\n{len(client_procs)} client(s) running. Waiting for federation to complete...\n")

    for name, p in client_procs:
        p.wait()
        print(f"[{name}] finished.")

    print("\nFederated training complete. Shutting down server.")
    server_proc.terminate()
    server_proc.wait()
    print("Done.")
