import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

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
            nn.Linear(64, num_classes)
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
    params_dict = zip(keys, parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

class IoTClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()

        for epoch in range(1):
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.valloader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return float(total_loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def create_client(data_path, batch_size=128):
    X, y = load_client_data(data_path)
    input_dim = X.shape[1]

    n = len(X)
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    trainloader = get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    valloader = get_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = SimpleMLP(input_dim=input_dim).to(DEVICE)
    return IoTClient(model, trainloader, valloader).to_client()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fl_client.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    client = create_client(data_path)
    print(f"Starting Flower client for {data_path}...")
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
