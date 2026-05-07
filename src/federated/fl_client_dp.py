import argparse
import os
import json

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

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


def get_dataloader(X, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_parameters(model):
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)


class DPClient:
    """
    Differential-Privacy Federated Learning client using Opacus.
    Designed for in-process simulation (no gRPC/Flower server needed).
    """

    def __init__(self, model, trainloader, valloader, client_name,
                 noise_multiplier=0.8, max_grad_norm=1.0, delta=1e-5):
        self.client_name = client_name
        self.criterion = nn.CrossEntropyLoss()
        self.delta = delta
        self.epsilon_log = []

        # Opacus wraps model, optimizer, and dataloader
        base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.trainloader = privacy_engine.make_private(
            module=model,
            optimizer=base_optimizer,
            data_loader=trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        self.privacy_engine = privacy_engine
        self.valloader = valloader

    def get_parameters(self):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, global_weights):
        self.set_parameters(global_weights)
        self.model.train()

        for X_batch, y_batch in self.trainloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()

        epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
        self.epsilon_log.append(float(epsilon))

        os.makedirs("outputs/dp_logs", exist_ok=True)
        with open(f"outputs/dp_logs/{self.client_name}_epsilon.json", "w") as f:
            json.dump({
                "client": self.client_name,
                "epsilon_history": self.epsilon_log,
                "latest_epsilon": float(epsilon),
                "delta": self.delta,
            }, f, indent=4)

        return self.get_parameters(), len(self.trainloader.dataset), float(epsilon)

    def evaluate(self, global_weights):
        self.set_parameters(global_weights)
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
        epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
        return float(total_loss), len(self.valloader.dataset), float(accuracy), float(epsilon)


def create_dp_client(data_path, client_name, noise_multiplier=0.8, max_grad_norm=1.0):
    X, y = load_client_data(data_path)
    input_dim = X.shape[1]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    trainloader = get_dataloader(X_train, y_train)
    valloader   = get_dataloader(X_val, y_val, shuffle=False)
    model = SimpleMLP(input_dim=input_dim).to(DEVICE)
    return DPClient(model, trainloader, valloader, client_name,
                    noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      type=str, required=True)
    parser.add_argument("--client_name",    type=str, required=True)
    parser.add_argument("--noise_multiplier", type=float, default=0.8)
    parser.add_argument("--max_grad_norm",  type=float, default=1.0)
    args = parser.parse_args()

    client = create_dp_client(args.data_path, args.client_name,
                               args.noise_multiplier, args.max_grad_norm)
    weights, n, eps = client.fit(client.get_parameters())
    print(f"[{args.client_name}] Trained on {n} samples | epsilon={eps:.4f}")
