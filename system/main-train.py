import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import os
from torch.utils.data import DataLoader, TensorDataset, random_split

# Function to read IDX files in MNIST dataset format
def read_idx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    with gzip.open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')
        if magic_number == 2049:  # Label file
            data = np.frombuffer(f.read(), dtype=np.uint8)
        elif magic_number == 2051:  # Image file
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
        else:
            raise ValueError(f"Invalid magic number {magic_number} in file: {file_path}")
    return data

# Subspace Meta-Learner Module
# Subspace Meta-Learner Module
# Subspace Meta-Learner Module
class SubspaceMetaLearner(nn.Module):
    def __init__(self, input_dim, m, output_dim):
        super(SubspaceMetaLearner, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.m = m  # 子空间维度
        # 调整 O 的形状为 (input_dim * output_dim, m)
        self.O = nn.Parameter(torch.randn(input_dim * output_dim, m) * 0.01)

    def forward(self, v):
        # 确保 v 的形状为 (m,)
        v = v.view(self.m, 1)  # 将 v 调整为 (m, 1)
        weight = torch.matmul(self.O, v).view(self.output_dim, self.input_dim)
        return weight

    def update_O(self, local_O):
        with torch.no_grad():
            self.O.data += local_O.data  # 聚合各个客户端的 O
        # 正交化 O，以保持尺寸不变
        self.O.data = orthogonalize(self.O.data)


# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)
        return x

# Train local model function
# Train local model function
# Train local model function
def train_local_model(model, train_loader, meta_learner, device, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # 生成与 O 形状匹配的随机向量 v
            v = torch.randn(meta_learner.m, device=device)  # 确保 v 的形状为 (m,)
            new_weights = meta_learner(v)
            model.fc1.weight = nn.Parameter(new_weights)

            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

    # 返回本地更新后的 O
    return meta_learner.O

# Server-side aggregation and orthogonalization function
def orthogonalize(matrix):
    q, _ = torch.linalg.qr(matrix)
    return q

def federated_training(global_model, train_loaders, meta_learner, epochs, rounds, device, val_loader):
    for round in range(rounds):
        for i, train_loader in enumerate(train_loaders):
            # 本地客户端训练并返回更新后的 O
            local_O = train_local_model(global_model, train_loader, meta_learner, device, epochs)

            # 服务器端聚合 O，并通过正交化生成新的 O
            meta_learner.update_O(local_O)

        # 执行每一轮的验证
        val_loss, accuracy = test(global_model, val_loader, device)
        print(f"Round {round + 1}/{rounds} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Model evaluation function
def test(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy

# Main function to initiate the process
def main():
    # Hyperparameters
    n = 28 * 28  # input dimension (flattened 28x28 image)
    output_dim = 10  # number of classes (digits 0-9)
    epochs = 5  # local training epochs
    federated_rounds = 100  # number of federated rounds
    batch_size = 128  # batch size
    lr = 0.01  # learning rate

    # Load data
    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/train-labels-idx1-ubyte.gz')
    val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/t10k-images-idx3-ubyte.gz')
    val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/t10k-labels-idx1-ubyte.gz')

    # Data preprocessing
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1) / 255.0
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    # Split data into clients
    num_clients = 5
    client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
    train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the global model and meta learner
    global_model = SimpleNN(n, output_dim).to(device)
    meta_learner = SubspaceMetaLearner(n, 20, output_dim).to(device)

    # Perform federated learning
    federated_training(global_model, train_loaders, meta_learner, epochs, federated_rounds, device, val_loader)

    # Validate the global model after final round
    val_loss, accuracy = test(global_model, val_loader, device)
    print(f'Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {accuracy:.2f}%')

    # Save the trained model
    model_path = 'model.pth'
    torch.save(global_model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
