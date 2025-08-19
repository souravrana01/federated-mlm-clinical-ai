import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from flwr.simulation import start_simulation  # Correct import for start_simulation
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Simplified Multimodal Model
class MultimodalModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, num_classes=2):
        super().__init__()
        self.text_encoder = nn.Linear(text_dim, 128)
        self.image_encoder = nn.Linear(image_dim, 128)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, text, image):
        text_feat = torch.relu(self.text_encoder(text))
        image_feat = torch.relu(self.image_encoder(image))
        combined = torch.cat((text_feat, image_feat), dim=1)
        return self.classifier(combined)

# 2. Flower Client with DP-SGD
class DPClient:
    def __init__(self, cid):
        self.cid = cid
        self.model = MultimodalModel()
        self.privacy_engine = PrivacyEngine()
        
        # Create synthetic data for this client
        num_samples = 500
        self.train_data = (
            torch.randn(num_samples, 768),  # Text features
            torch.randn(num_samples, 768),  # Image features
            torch.randint(0, 2, (num_samples,))  # Labels
        )
        self.val_data = (
            torch.randn(100, 768),
            torch.randn(100, 768),
            torch.randint(0, 2, (100,))
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            TensorDataset(*self.train_data), batch_size=32, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(*self.val_data), batch_size=32
        )
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def fit(self, parameters, config):
        # Set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Attach privacy engine (corrected usage)
        privacy_engine = PrivacyEngine()
        model, optimizer, _ = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            noise_multiplier=0.1,
            max_grad_norm=1.0,
            secure_mode=True,
        )
        
        # Training loop (1 epoch)
        model.train()
        for text, image, labels in self.train_loader:  # Use original train_loader
            optimizer.zero_grad()
            outputs = model(text, image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Get privacy spent
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"Client {self.cid} - Privacy spent: ε = {epsilon:.2f}")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Evaluation
        self.model.eval()
        correct, total = 0, 0
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        
        with torch.no_grad():
            for text, image, labels in self.val_loader:
                outputs = self.model(text, image)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / len(self.val_loader)
        return float(avg_loss), len(self.val_loader.dataset), {"accuracy": accuracy}

# 3. Simulation Setup
def client_fn(cid: str):
    return DPClient(cid)

def weighted_avg(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=5,
    min_evaluate_clients=3,
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=weighted_avg
)

# 4. Run Simulation
print("Starting Federated Learning Simulation with DP-SGD")
print("Configuration:")
print("- 5 hospital nodes")
print("- DP-SGD with noise=0.1, clipping=1.0")
print("- 10 communication rounds")

hist = start_simulation(
    client_fn=client_fn,
    num_clients=5,
    config=ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0.0}  # Added num_gpus for completeness
)

# 5. Plot Results
print("\nSimulation Complete! Plotting Results...")

# Extract accuracy metrics
rounds = [r + 1 for r in range(len(hist.metrics_distributed["accuracy"]))]
accuracies = [acc for acc in hist.metrics_distributed["accuracy"]]

plt.figure(figsize=(10, 5))
plt.plot(rounds, accuracies, marker='o')
plt.title("Federated Learning Performance with DP-SGD")
plt.xlabel("Communication Rounds")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.ylim(0, 1)
plt.show()

print("Final Model Accuracy: {:.2f}%".format(accuracies[-1]*100))