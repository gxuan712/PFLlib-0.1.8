import numpy as np
import os
import gzip
import random
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(1)
np.random.seed(1)

num_clients = 20
dir_path = "MNIST/"

# Function to read IDX files
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

# Function to separate data among clients
def separate_data(dataset, num_clients, num_classes, niid, balance, partition, class_per_client=2):
    X, y = dataset
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]

    if niid:
        # Non-IID data separation
        indices = np.argsort(y)
        X = X[indices]
        y = y[indices]
        class_distribution = np.arange(num_classes)
        for i in range(num_clients):
            selected_classes = np.random.choice(class_distribution, class_per_client, replace=False)
            for cls in selected_classes:
                cls_indices = np.where(y == cls)[0]
                selected_indices = np.random.choice(cls_indices, len(cls_indices) // num_clients, replace=False)
                client_data[i].extend(X[selected_indices])
                client_labels[i].extend(y[selected_indices])
    else:
        # IID data separation
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        data_split = np.array_split(X, num_clients)
        labels_split = np.array_split(y, num_clients)
        for i in range(num_clients):
            client_data[i] = data_split[i]
            client_labels[i] = labels_split[i]

    return client_data, client_labels, None

def split_data(X, y):
    split_ratio = 0.8
    train_data, test_data = [], []
    for i in range(len(X)):
        split_index = int(len(X[i]) * split_ratio)
        train_data.append((X[i][:split_index], y[i][:split_index]))
        test_data.append((X[i][split_index:], y[i][split_index:]))
    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for i in range(num_clients):
        torch.save({'data': train_data[i][0], 'labels': train_data[i][1]}, os.path.join(train_path, f'client_{i}.pt'))
        torch.save({'data': test_data[i][0], 'labels': test_data[i][1]}, os.path.join(test_path, f'client_{i}.pt'))

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
    test_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    test_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    dataset_image = np.concatenate((train_images, test_images), axis=0)
    dataset_label = np.concatenate((train_labels, test_labels), axis=0)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(None, dir_path + "train/", dir_path + "test/", train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)

# Save client data to disk
def save_client_data(client_data, client_labels, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i, (data, labels) in enumerate(zip(client_data, client_labels)):
        torch.save({'data': data, 'labels': labels}, os.path.join(dir_path, f'client_{i}.pt'))

# Load client data from disk
def load_client_data(client_id, dir_path):
    data = torch.load(os.path.join(dir_path, f'client_{client_id}.pt'))
    return data['data'], data['labels']

# Orthogonalize function
def orthogonalize(O):
    with torch.no_grad():
        u, _, v = torch.svd(O, some=True)
        return torch.matmul(u, v.T)

# Aggregate function for O
def aggregate_O(O_list):
    O_new = torch.mean(torch.stack(O_list), dim=0)
    O_orthogonal = orthogonalize(O_new)
    return O_orthogonal

# Simulate server-client communication
def server_side(O, num_clients=20, dir_path="MNIST_clients/", num_selected_clients=10):
    O_list = []
    total_loss = 0
    selected_clients = random.sample(range(num_clients), num_selected_clients)

    for client_id in selected_clients:
        client_images, client_labels = load_client_data(client_id, dir_path)
        O_k, client_loss = client_side(O, client_images, client_labels)
        O_list.append(O_k)
        total_loss += client_loss

    O_global = aggregate_O(O_list)
    avg_loss = total_loss / num_selected_clients
    return O_global, avg_loss

# Client-side training
# Adjust learning rate

# KL divergence coefficient
kl_coefficient = 0.001  # You may adjust this value

# Client-side training function
def client_side(O, client_images, client_labels):
    v_k = nn.Parameter(torch.randn(O.size(1), 10, requires_grad=True))
    O.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD([O, v_k], lr=0.001, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Prior distribution
    prior_mean = torch.zeros_like(v_k)
    prior_var = torch.ones_like(v_k)

    for epoch in range(5):
        optimizer.zero_grad()
        Ov_k = torch.matmul(O, v_k)
        loss = 0
        for i in range(len(client_images)):
            d = client_images[i].view(-1)
            prediction = torch.matmul(O.T, d)
            loss += criterion(prediction.unsqueeze(0), client_labels[i].unsqueeze(0))

        # Compute KL divergence and scale it
        kl_divergence = 0.5 * torch.sum(torch.pow(v_k - prior_mean, 2) / prior_var + torch.log(prior_var) - 1)
        loss += kl_coefficient * kl_divergence

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = loss.item() / len(client_images)
    return O.detach(), avg_loss

# Test the model
def test_model(O, test_images, test_labels):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(test_images)):
            d = test_images[i].view(-1)
            prediction = torch.matmul(O.T, d)
            predicted_label = torch.argmax(prediction)
            total += 1
            correct += (predicted_label == test_labels[i]).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Main function
if __name__ == "__main__":
    niid = True
    balance = True
    partition = None

    # Generate and save the dataset
    generate_dataset(dir_path, num_clients, niid, balance, partition)

    # Initialize global O matrix and orthogonalize
    n = 784  # Flattened size of MNIST image (28*28)
    m = 50  # Low-dimensional space size
    O_global = nn.Parameter(torch.randn(n, m, requires_grad=True))
    O_global = orthogonalize(O_global)

    # Load validation data
    val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')
    val_images = torch.tensor(val_images, dtype=torch.float32) / 255.0
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    val_images = val_images.view(-1, 28 * 28)

    # Federated learning rounds
    num_rounds = 500
    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        O_global, avg_loss = server_side(O_global, num_clients=20, dir_path="MNIST_clients/", num_selected_clients=10)
        print(f"Average Loss after Round {round + 1}: {avg_loss:.4f}")

        # Calculate and print accuracy after each round
        accuracy = test_model(O_global, val_images, val_labels)
        print(f"Validation Accuracy after Round {round + 1}: {accuracy:.2f}%")

    torch.save(O_global, 'O_global.pth')
    print("Model saved as 'O_global.pth'")

    O_global_loaded = torch.load('O_global.pth')
    accuracy = test_model(O_global_loaded, val_images, val_labels)
    print(f"Test Accuracy after training: {accuracy:.2f}%")
