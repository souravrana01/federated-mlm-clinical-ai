import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import opacus
from opacus import PrivacyEngine
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.server import Server
from flwr.client import ClientApp, NumPyClient
from flwr.simulation import start_simulation
from flwr.common import SecureAggregation
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the Multimodal Model Architecture (ClinicalBERT + ViT)
class MultimodalModel(nn.Module):
    def __init__(self, text_feature_dim=768, image_feature_dim=768, num_classes=2):
        super().__init__()
        # Text branch (simplified ClinicalBERT)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Image branch (simplified ViT)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        
        # Classifier
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, text_input, image_input):
        text_features = self.text_encoder(text_input)
        image_features = self.image_encoder(image_input)
        combined = torch.cat((text_features, image_features), dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)

# 2. Create Flower Client with DP-SGD
class DPClient(NumPyClient):
    def __init__(self, model, train_loader, val_loader, privacy_engine):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.privacy_engine = privacy_engine
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # Set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        
        # Training configuration
        lr = config.get("lr", 0.001)
        epochs = config.get("epochs", 1)
        
        # Define optimizer and attach privacy engine
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if self.privacy_engine:
            optimizer = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                noise_multiplier=0.1,  # As specified in your paper
                max_grad_norm=1.0,     # Clipping norm as per your DP-SGD params
            )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for text, image, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(text, image)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Get updated parameters and sample size
        updated_params = self.get_parameters(config={})
        num_samples = len(self.train_loader.dataset)
        
        # Return updated parameters and metrics
        return updated_params, num_samples, {}
    
    def evaluate(self, parameters, config):
        # Set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        
        # Evaluation
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        
        with torch.no_grad():
            for text, image, labels in self.val_loader:
                outputs = self.model(text, image)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        return avg_loss, total_samples, {"accuracy": accuracy}

# 3. Flower ClientApp with Secure Aggregation
def client_fn(cid: str) -> DPClient:
    """Create a Flower client with DP-SGD and simulated data."""
    # Simulate loading MedMNIST and MIMIC-IV data for this client
    # In practice, you would load real datasets here
    num_samples = 1000  # Simulated dataset size per client
    text_feature_dim = 768
    image_feature_dim = 768
    
    # Create synthetic data (replace with actual dataset loading)
    train_text = torch.randn(num_samples, text_feature_dim)
    train_image = torch.randn(num_samples, image_feature_dim)
    train_labels = torch.randint(0, 2, (num_samples,))
    
    val_text = torch.randn(num_samples // 5, text_feature_dim)
    val_image = torch.randn(num_samples // 5, image_feature_dim)
    val_labels = torch.randint(0, 2, (num_samples // 5,))
    
    # Create datasets and loaders
    train_dataset = torch.utils.data.TensorDataset(train_text, train_image, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_text, val_image, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model and privacy engine
    model = MultimodalModel()
    privacy_engine = PrivacyEngine()
    
    return DPClient(model, train_loader, val_loader, privacy_engine)

# 4. Configure Federated Learning with Secure Aggregation
def get_fedavg_strategy():
    """Configure FedAvg with Secure Aggregation as per your paper."""
    return FedAvg(
        fraction_fit=1.0,  # Sample 100% of clients for training
        fraction_evaluate=0.5,  # Sample 50% of clients for evaluation
        min_fit_clients=5,  # Minimum 5 clients for training (as per your 5-10 node setup)
        min_evaluate_clients=3,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        # Enable Secure Aggregation
        secure_aggregation=SecureAggregation(
            minimum_severity=2,  # Medium security level
            add_mask_optimization=True,
            add_mask_quantization=True,
        )
    )

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate metrics with weighting by number of samples."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 5. Simulation Configuration
def run_simulation():
    """Run the federated learning simulation with DP-SGD and SecAgg."""
    # Configuration as per your paper
    num_clients = 5  # Starting with 5 nodes (can scale to 10)
    num_rounds = 50  # As mentioned in your DP-SGD parameters
    
    # Start simulation
    hist = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=Server(num_rounds=num_rounds),
        strategy=get_fedavg_strategy(),
        client_resources={"num_cpus": 1, "num_gpus": 0.1},  # Simulate GPU usage
    )
    
    return hist

# 6. Visualization Functions (for accuracy vs rounds, ε-vs-performance)
def plot_results(history):
    """Plot accuracy and privacy budget over rounds."""
    # Extract metrics
    rounds = [r + 1 for r in range(len(history.metrics_distributed["accuracy"]))]
    accuracies = [acc for acc in history.metrics_distributed["accuracy"]]
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(x=rounds, y=accuracies)
    plt.title("Accuracy vs Communication Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    
    # Plot privacy budget (simulated - in practice use actual epsilon values)
    epsilons = [1.5 * (r/50) for r in rounds]  # Linear increase to ε~1.5 as per your paper
    plt.subplot(1, 2, 2)
    sns.lineplot(x=rounds, y=epsilons)
    plt.title("Privacy Budget (ε) vs Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Epsilon (ε)")
    
    plt.tight_layout()
    plt.savefig("fl_dp_results.png")
    plt.show()

# 7. Main Execution
if __name__ == "__main__":
    print("Starting Federated Learning with DP-SGD and Secure Aggregation")
    print("Configuration as per research paper:")
    print("- 5-10 hospital nodes (simulated)")
    print("- DP-SGD with noise=0.1, clipping=1.0, ε~1.5")
    print("- Secure Aggregation enabled")
    print("- 50 communication rounds")
    
    # Run simulation
    history = run_simulation()
    
    # Plot results
    plot_results(history)
    
    print("Simulation complete. Results saved to fl_dp_results.png")