import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from system.flcore.trainmodel.subspace_meta_learner import train_subspace_meta_learner

# Function to read idx files
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

# Example usage of SubspaceMetaLearner in meta-learning training
def main():
    # Hyperparameters
    n = 784  # Input dimension (flattened 28x28 images)
    m = 50  # Subspace dimension
    output_dim = 10  # Number of classes (digits 0-9)
    epochs = 100  # Number of training epochs
    batch_size = 128  # Batch size
    learning_rate = 0.1  # Initial learning rate
    momentum = 0.9  # Momentum for SGD
    weight_decay = 1e-4  # L2 regularization

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Random rotation
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize
    ])

    # Load data
    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
    val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    # Transform data and apply augmentation
    train_images = torch.tensor(train_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_images = torch.tensor(val_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, output_dim)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = x.view(-1, self.fc1.in_features)  # Flatten image
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SimpleNN(n, output_dim)  # Single model

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Call the training function with correct arguments
    train_subspace_meta_learner(model, train_loader, val_loader, n, m, epochs, output_dim, learning_rate, momentum, device, weight_decay)

    # Save the trained model
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
