import argparse
import json
import os
import sys

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.crypto.pqc_utils import PQCManager, SignatureManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_client_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return X, y


def get_dataloader(X, y, batch_size=128, shuffle=True):
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_parameters(model):
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)


class PQCClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, client_name):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_name = client_name
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.pqc = PQCManager("Kyber512")
        self.sig = SignatureManager()
        self.sig_pk, self.sig_sk = self.sig.generate_keypair()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()

        for _ in range(1):
            for X_batch, y_batch in self.trainloader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()

        weights = get_parameters(self.model)
        serialized = self.pqc.serialize_weights(weights)

        # PQC signing layer (Dilithium2)
        signature = self.sig.sign(self.sig_sk, serialized)

        os.makedirs("outputs/pqc_logs", exist_ok=True)
        with open(f"outputs/pqc_logs/{self.client_name}_last_signature.json", "w") as f:
            json.dump({
                "client": self.client_name,
                "signature_len": len(signature),
            }, f, indent=4)

        return weights, len(self.trainloader.dataset), {
            "client_name": self.client_name,
        }

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.valloader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = self.model(X_batch)
                total_loss += self.criterion(outputs, y_batch).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return float(total_loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def create_numpy_client(data_path, client_name):
    X, y = load_client_data(data_path)
    input_dim = X.shape[1]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    trainloader = get_dataloader(X_train, y_train)
    valloader = get_dataloader(X_val, y_val, shuffle=False)
    model = SimpleMLP(input_dim=input_dim).to(DEVICE)
    return PQCClient(model, trainloader, valloader, client_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--client_name", type=str, required=True)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    os.makedirs("outputs/pqc_logs", exist_ok=True)
    client = create_numpy_client(args.data_path, args.client_name)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
