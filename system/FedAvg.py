import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import random

# Function to read IDX files (MNIST format)
def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        if magic_number == 2051:  # Images
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        elif magic_number == 2049:  # Labels
            num_labels = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f'Invalid magic number {magic_number} in file: {filename}')
    return data

# Load the MNIST dataset
train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

# Normalize and convert to tensors
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0  # Add channel dimension and normalize
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1) / 255.0
val_labels = torch.tensor(val_labels, dtype=torch.long)

# Function to create data loaders
def create_data_loader(images, labels, batch_size=64):
    dataset = data_utils.TensorDataset(images, labels)
    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Split the data into multiple clients
num_clients = 20
client_loaders = []

# Split data for clients
split_size = len(train_images) // num_clients
for i in range(num_clients):
    client_images = train_images[i * split_size:(i + 1) * split_size]
    client_labels = train_labels[i * split_size:(i + 1) * split_size]
    client_loader = create_data_loader(client_images, client_labels)
    client_loaders.append(client_loader)

# Validation loader
val_loader = create_data_loader(val_images, val_labels)

# Example Model (Simple Neural Network)
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Client Update (Local Training)
def client_update(model, optimizer, train_loader, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data.view(-1, 28 * 28))  # Flatten images
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# Server Aggregation (Federated Averaging)
def server_aggregation(global_model, client_models):
    global_state_dict = global_model.state_dict()
    for k in global_state_dict.keys():
        global_state_dict[k] = torch.stack([client_models[i][k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_state_dict)
    return global_model

# Main Training Loop for FedAvg
def fedavg(global_model, num_clients, train_loaders, num_rounds=10, client_epochs=1, fraction=0.1):
    client_models = [global_model for _ in range(num_clients)]

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")

        # Select clients
        m = max(int(fraction * num_clients), 1)
        selected_clients = random.sample(range(num_clients), m)

        # Local updates
        local_states = []
        for client in selected_clients:
            local_model = SimpleNN(784, 10)  # Assume 784 input size (28x28 images) and 10 classes
            local_model.load_state_dict(client_models[client].state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=0.01)
            local_state_dict = client_update(local_model, optimizer, train_loaders[client], epochs=client_epochs)
            local_states.append(local_state_dict)

        # Server aggregation
        global_model = server_aggregation(global_model, local_states)

        # Distribute global model back to clients
        for client in range(num_clients):
            client_models[client].load_state_dict(global_model.state_dict())

    return global_model

# Function to test the global model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Initialize global model
global_model = SimpleNN(784, 10)  # 28x28 images flattened to 784, 10 output classes for MNIST

# Run FedAvg
global_model = fedavg(global_model, num_clients, client_loaders, num_rounds=100, client_epochs=1, fraction=0.1)

# Test global model
test_accuracy = test_model(global_model, val_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")
