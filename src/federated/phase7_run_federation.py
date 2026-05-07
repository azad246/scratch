import os
import subprocess
import time

DEVICE_FOLDERS = [
    "danmini_doorbell",
    "ecobee_thermostat",
    "philips_baby_monitor",
    "provision_security_camera",
    "samsung_webcam"
]

# Corrected BASE_DIR to match outputs from phase 4
BASE_DIR = os.path.join("outputs", "final_processed")
SERVER_SCRIPT = os.path.join("src", "federated", "fl_server.py")
CLIENT_SCRIPT = os.path.join("src", "federated", "fl_client.py")

def run_server():
    print("Launching server...")
    return subprocess.Popen(["python", SERVER_SCRIPT])

def run_client(device_name):
    data_path = os.path.join(BASE_DIR, device_name, "train.csv")
    if not os.path.exists(data_path):
        print(f"  [ERROR] Data path not found: {data_path}")
        return None
    print(f"Launching client for {device_name}...")
    return subprocess.Popen([
        "python", CLIENT_SCRIPT, data_path
    ])

if __name__ == "__main__":
    server_proc = run_server()
    time.sleep(10) # Give server time to start

    client_procs = []
    for device in DEVICE_FOLDERS:
        p = run_client(device)
        if p:
            client_procs.append(p)
        time.sleep(2)

    print("\nWaiting for clients to finish...")
    for p in client_procs:
        p.wait()

    print("Federated training complete. Closing server.")
    server_proc.terminate()
    server_proc.wait()
    print("Done.")
