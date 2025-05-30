
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
import numpy as np
import random

# Dummy data simulation
def generate_data(samples=100):
    X = torch.randn(samples, 10)
    y = torch.randint(0, 2, (samples,))
    return X, y

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def train(model, data, epochs=3):
    X, y = data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    for (name, _), val in zip(model.state_dict().items(), parameters):
        model.state_dict()[name].copy_(torch.tensor(val))

# Flower client
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = SimpleModel()
        self.data = generate_data()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(self.model, self.data)
        return get_parameters(self.model), len(self.data[0]), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        return 0.5, len(self.data[0]), {}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient())
