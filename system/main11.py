import numpy as np
import torch
import torch.nn as nn
from system.flcore.trainmodel.subspace_meta_learner import SubspaceMetaLearner, orthogonalize, \
    train_subspace_meta_learner
from system.flcore.servers.serversBayesian import serversBayesian


# Example model definition
class ExampleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExampleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

    def set_weights(self, weights):
        self.fc.weight = nn.Parameter(weights)


# Load datasets from .npz files
train_data = np.load('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/0.npz')
test_data = np.load('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/0.npz')

print("Train data keys:", train_data.keys())
print("Test data keys:", test_data.keys())

# Assume the .npz file contains 'x_train' and 'y_train' arrays
train_features = torch.tensor(train_data['x_train'], dtype=torch.float32)
train_labels = torch.tensor(train_data['y_train'], dtype=torch.float32)
test_features = torch.tensor(test_data['x_test'], dtype=torch.float32)
test_labels = torch.tensor(test_data['y_test'], dtype=torch.float32)

# Example split into multiple datasets (assuming the data is to be split for different clients)
num_clients = 10
split_train_features = torch.chunk(train_features, num_clients)
split_train_labels = torch.chunk(train_labels, num_clients)
split_test_features = torch.chunk(test_features, num_clients)
split_test_labels = torch.chunk(test_labels, num_clients)

# Create lists of tuples (features, labels) for train and validation sets
train_sets = [(split_train_features[i], split_train_labels[i]) for i in range(num_clients)]
val_sets = [(split_test_features[i], split_test_labels[i]) for i in range(num_clients)]

input_dim = train_features.shape[1]
output_dim = train_labels.shape[1]

models = [ExampleModel(input_dim, output_dim) for _ in range(num_clients)]

# Define arguments for serversBayesian
args = {}  # Define your specific arguments here
times = 1  # Define your specific times here
n, m = 100, 10
meta_epochs = 100

# Initialize serversBayesian
server = serversBayesian(args, times, n, m, meta_epochs)

# Train the server with the models and datasets
server.train()
