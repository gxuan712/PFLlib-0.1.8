import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import os

# Example model definition
class ExampleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExampleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

# Server class definition
class serversBayesian:
    def __init__(self, device, dataset, num_classes, global_rounds, local_epochs):
        self.device = device
        self.dataset = dataset
        self.num_classes = num_classes
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs

    def train(self, models, train_sets):
        print("Training process starts...")
        for model, (train_x, train_y) in zip(models, train_sets):
            optimizer = optim.SGD(model.parameters())
            loss_function = nn.NLLLoss()
            for epoch in range(self.local_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_x)
                loss = loss_function(outputs, train_y)
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def read_idx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    with gzip.open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')
        if magic_number == 2049:  # Label file (magic number 2049)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        elif magic_number == 2051:  # Image file (magic number 2051)
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
        else:
            raise ValueError(f"Invalid magic number {magic_number} in file: {file_path}")
    return data

# Load datasets with added path checks
train_images_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz'
train_labels_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz'
test_images_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz'
test_labels_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz'

print("Checking file paths...")

if not os.path.exists(train_images_path):
    print(f"Train images file not found at {train_images_path}")
if not os.path.exists(train_labels_path):
    print(f"Train labels file not found at {train_labels_path}")
if not os.path.exists(test_images_path):
    print(f"Test images file not found at {test_images_path}")
if not os.path.exists(test_labels_path):
    print(f"Test labels file not found at {test_labels_path}")

train_images = read_idx(train_images_path)
train_labels = read_idx(train_labels_path)
test_images = read_idx(test_images_path)
test_labels = read_idx(test_labels_path)

# Handling data
train_features = torch.tensor(train_images.reshape(-1, 28*28), dtype=torch.float32)  # Reshape for fully connected input
train_labels = torch.tensor(train_labels, dtype=torch.long)
input_dim = train_features.shape[1]
output_dim = 10

# Creating a single example model for simplicity
model = ExampleModel(input_dim, output_dim)

# Simple data split for demonstration
train_set = (train_features[:1000], train_labels[:1000])

# Server initialization
server = serversBayesian(device='cpu', dataset='MNIST', num_classes=10, global_rounds=1, local_epochs=5)

# Start training with one model and one part of the dataset
server.train([model], [train_set])
