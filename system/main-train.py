import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import numpy as np
import os
import random

# 定义函数以读取IDX文件
def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        if magic_number == 2051:  # 图像
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        elif magic_number == 2049:  # 标签
            num_labels = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f'Invalid magic number {magic_number} in file: {filename}')
    return data


# 加载数据集
train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

# 归一化图像数据并转换为张量
train_images = torch.tensor(train_images, dtype=torch.float32) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_images = torch.tensor(val_images, dtype=torch.float32) / 255.0
val_labels = torch.tensor(val_labels, dtype=torch.long)

# 将图像展平
train_images = train_images.view(-1, 28 * 28)
val_images = val_images.view(-1, 28 * 28)

# 将数据划分为5个客户端的数据集
def separate_data(data, labels, num_clients, num_classes, niid, balance):
    """
    This function separates the data into chunks for different clients.
    :param data: Input data
    :param labels: Corresponding labels
    :param num_clients: Number of clients
    :param num_classes: Number of classes in the dataset
    :param niid: Whether to make the data non-IID
    :param balance: Whether the data should be balanced across clients
    :return: Separated data for clients
    """
    # Initialize placeholders for separated data
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]

    # Optionally shuffle the data to ensure randomness
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    if niid:
        # Non-IID partitioning
        for i in range(num_clients):
            client_indices = indices[i::num_clients]
            client_data[i] = data[client_indices]
            client_labels[i] = labels[client_indices]
    else:
        # IID partitioning
        for i in range(num_clients):
            client_data[i] = data[i::num_clients]
            client_labels[i] = labels[i::num_clients]

    return client_data, client_labels

def save_client_data(client_data, client_labels, dir_path):
    """
    Save client data to the specified directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i, (data, labels) in enumerate(zip(client_data, client_labels)):
        torch.save({'data': data, 'labels': labels}, os.path.join(dir_path, f'client_{i}.pt'))

# Hyperparameters
num_clients = 20
dir_path = "MNIST_clients/"
niid = True  # Set to True for non-IID data, False for IID
balance = True  # Set to True for balanced data

# Separate and save data for clients
client_train_images, client_train_labels = separate_data(train_images, train_labels, num_clients, 10, niid, balance)
save_client_data(client_train_images, client_train_labels, dir_path)

def load_client_data(client_id, dir_path):
    """
    Load data for a specific client.
    """
    data = torch.load(os.path.join(dir_path, f'client_{client_id}.pt'))
    return data['data'], data['labels']

# Orthogonalize function (O^T O = I)
def orthogonalize(O):
    with torch.no_grad():
        u, _, v = torch.svd(O, some=True)
        return torch.matmul(u, v.T)

# Aggregate function for O
def aggregate_O(O_list):
    O_new = torch.mean(torch.stack(O_list), dim=0)
    # Orthogonalize updated O
    O_orthogonal = orthogonalize(O_new)
    return O_orthogonal

# Simulate server-client communication
# Simulate server-client communication
def server_side(O, num_clients=20, dir_path="MNIST_clients/", num_selected_clients=10):
    O_list = []
    total_loss = 0

    # Randomly select clients for this round
    selected_clients = random.sample(range(num_clients), num_selected_clients)

    for client_id in selected_clients:
        # Load client data
        client_images, client_labels = load_client_data(client_id, dir_path)

        # Simulate client training and return updated O and loss
        O_k, client_loss = client_side(O, client_images, client_labels)
        O_list.append(O_k)
        total_loss += client_loss

    # Aggregate O from clients and update global O
    O_global = aggregate_O(O_list)

    # Calculate average loss
    avg_loss = total_loss / num_selected_clients
    return O_global, avg_loss

# Client-side training
def client_side(O, client_images, client_labels):
    # Initialize personalized low-dimensional weights v_k for the client
    v_k = nn.Parameter(torch.randn(O.size(1), 10, requires_grad=True))  # Ensure v_k requires gradients

    # Ensure that O requires gradients
    O.requires_grad = True

    # Define optimizer for v_k and O
    optimizer = optim.Adam([O, v_k], lr=0.01)

    num_local_epochs = 5
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification tasks

    for epoch in range(num_local_epochs):
        optimizer.zero_grad()

        # Calculate prediction O * v_k
        Ov_k = torch.matmul(O, v_k)  # Ov_k dimensions (n, 10)

        # Calculate loss on the training set
        loss = 0
        for i in range(len(client_images)):
            d = client_images[i].view(-1)  # Reshape data point d (784,)
            prediction = torch.matmul(O.T, d)  # Prediction dimensions (m,)
            loss += criterion(prediction.unsqueeze(0), client_labels[i].unsqueeze(0))  # Calculate cross-entropy loss

        # Optimize v_k and O
        loss.backward()
        optimizer.step()

    # Return optimized O and local average loss
    avg_loss = loss.item() / len(client_images)
    return O.detach(), avg_loss


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

# Initialize global O matrix and orthogonalize
n = 784  # Flattened size of MNIST image (28*28)
m = 50  # Low-dimensional space size
O_global = nn.Parameter(torch.randn(n, m, requires_grad=True))  # Set requires_grad=True
O_global = orthogonalize(O_global)

# Federated learning rounds
num_rounds = 500  # Number of federated learning rounds

for round in range(num_rounds):
    print(f"Round {round + 1}/{num_rounds}")
    O_global, avg_loss = server_side(O_global, num_clients=20, dir_path="MNIST_clients/", num_selected_clients=10)
    print(f"Average Loss after Round {round + 1}: {avg_loss:.4f}")

    # Calculate accuracy on the validation set
    accuracy = test_model(O_global, val_images, val_labels)
    print(f"Validation Accuracy after Round {round + 1}: {accuracy:.2f}%")

# Save the global model
torch.save(O_global, 'O_global.pth')
print("Model saved as 'O_global.pth'")

# Load the model for testing
O_global_loaded = torch.load('O_global.pth')


# Calculate and print the accuracy on the test set
accuracy = test_model(O_global_loaded, val_images, val_labels)
print(f"Test Accuracy after training: {accuracy:.2f}%")
