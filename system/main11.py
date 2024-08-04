import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            loss_function = nn.NLLLoss()
            for epoch in range(self.local_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_x)
                loss = loss_function(outputs, train_y)
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Load datasets
mnist_data = np.load('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/mnist.npz')

# Handling data
train_features = torch.tensor(mnist_data['x_train'].reshape(-1, 28*28), dtype=torch.float32)  # Reshape for fully connected input
train_labels = torch.tensor(mnist_data['y_train'], dtype=torch.long)
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
