import gzip
import numpy as np
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
import torchmetrics
from torchmetrics.classification import CalibrationError

# Set the number of threads for CPU optimization
torch.set_num_threads(4)


def read_idx(file_path):
    # Function to read idx files
    with gzip.open(file_path, 'rb') as f:
        magic_number = struct.unpack(">I", f.read(4))[0]
        if magic_number == 2051:  # magic number for images
            num_images = struct.unpack(">I", f.read(4))[0]
            rows = struct.unpack(">I", f.read(4))[0]
            cols = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        elif magic_number == 2049:  # magic number for labels
            num_labels = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_labels)
        else:
            raise ValueError("Invalid IDX file magic number: {}".format(magic_number))
    return data


def train_local_model(model, device, train_loader, optimizer, epoch):
    # Training function
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test_with_torchmetrics(model, device, test_loader, temperature_model, ece_metric):
    # Testing function with torchmetrics
    model.eval()
    temperature_model.eval()
    correct = 0
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            scaled_output = temperature_model(output)
            prob = torch.exp(scaled_output)
            pred = prob.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_probs.append(prob)
            all_preds.append(pred)
            all_targets.append(target)

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    accuracy = 100. * correct / len(test_loader.dataset)

    # Calculate ECE using torchmetrics
    ece = ece_metric(all_probs, all_targets)
    return accuracy, ece.item()


class FedPerNet(nn.Module):
    def __init__(self):
        # Model definition
        super(FedPerNet, self).__init__()
        self.shared_conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.shared_conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.personalized_fc1 = nn.Linear(320, 50)
        self.personalized_fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.shared_conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.shared_conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.personalized_fc1(x))
        x = self.personalized_fc2(x)
        return torch.log_softmax(x, dim=1)


class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits):
        return logits / self.temperature


def federated_personalization_averaging(global_model, client_models):
    global_dict = global_model.state_dict()

    shared_layers = ['shared_conv1.weight', 'shared_conv1.bias', 'shared_conv2.weight', 'shared_conv2.bias']

    for k in shared_layers:
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)

    global_model.load_state_dict(global_dict, strict=False)
    return global_model


def client_update(client_model, optimizer, train_loader, device, epochs=1):
    for epoch in range(epochs):
        train_local_model(client_model, device, train_loader, optimizer, epoch)


def server_aggregate(global_model, client_models):
    return federated_personalization_averaging(global_model, client_models)


if __name__ == "__main__":
    # Main execution code

    device = torch.device("cpu")

    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
    test_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    test_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    train_images = torch.tensor(train_images, dtype=torch.float16).unsqueeze(1)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.tensor(test_images, dtype=torch.float16).unsqueeze(1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

    num_clients = 20
    client_models = [FedPerNet().to(device).half() for _ in range(num_clients)]
    global_model = FedPerNet().to(device).half()

    optimizers = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9) for model in client_models]

    temperature_model = TemperatureScaling().to(device)
    optimizer_temp = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

    ece_metric = CalibrationError(n_bins=15, norm='l1', task='multiclass', num_classes=10).to(device)

    global_accuracies = []
    global_eces = []

    global_epochs = 500
    local_epochs = 5
    clients_per_round = 10

    for epoch in range(global_epochs):
        print(f"Global Epoch {epoch + 1}/{global_epochs}")

        selected_clients = random.sample(range(num_clients), clients_per_round)

        for i in selected_clients:
            client_update(client_models[i], optimizers[i], train_loader, device, local_epochs)

        selected_client_models = [client_models[i] for i in selected_clients]
        global_model = server_aggregate(global_model, selected_client_models)

        global_shared_dict = {k: v for k, v in global_model.state_dict().items() if
                              k in ['shared_conv1.weight', 'shared_conv1.bias', 'shared_conv2.weight',
                                    'shared_conv2.bias']}
        for model in client_models:
            model_state = model.state_dict()
            model_state.update(global_shared_dict)
            model.load_state_dict(model_state, strict=False)

        accuracy, ece = test_with_torchmetrics(global_model, device, test_loader, temperature_model, ece_metric)
        global_accuracies.append(accuracy)
        global_eces.append(ece)
        print(f"Validation set: Accuracy: {accuracy:.2f}%, ECE: {ece:.4f}")

    epochs = range(1, global_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, global_accuracies, 'b', label='Accuracy')
    plt.title('Global Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, global_eces, 'r', label='ECE')
    plt.title('Global Model ECE')
    plt.xlabel('Epochs')
    plt.ylabel('ECE')
    plt.legend()

    plt.tight_layout()
    plt.show()
