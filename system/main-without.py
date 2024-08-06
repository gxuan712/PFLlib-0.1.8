import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Helper function to read IDX files
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


# Load datasets
train_images_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz'
train_labels_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz'
test_images_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz'
test_labels_path = 'C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz'

train_images = read_idx(train_images_path)
train_labels = read_idx(train_labels_path)
test_images = read_idx(test_images_path)
test_labels = read_idx(test_labels_path)

# Preprocessing
train_images = torch.tensor(train_images.reshape(-1, 28 * 28), dtype=torch.float32) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images.reshape(-1, 28 * 28), dtype=torch.float32) / 255.0
test_labels = torch.tensor(test_labels, dtype=torch.long)


# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model, loss function, and optimizer
input_dim = 28 * 28
output_dim = 10
model = SimpleNN(input_dim, output_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training the model
def train(model, train_images, train_labels, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_images)
        loss = loss_function(outputs, train_labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# Testing the model
def test(model, test_images, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_images)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == test_labels).sum().item() / len(test_labels)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')


# Train and test
train(model, train_images, train_labels, epochs=5)
test(model, test_images, test_labels)
