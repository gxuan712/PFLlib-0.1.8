import gzip
import numpy as np
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random


# 读取idx文件的函数
def read_idx(file_path):
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


# 读取MNIST数据
train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
test_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
test_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

# 转换为PyTorch张量
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(this.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# FedProx 本地更新函数
def train_local_model_prox(client_model, global_model, device, train_loader, optimizer, mu, epoch):
    client_model.train()
    global_params = list(global_model.parameters())

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = client_model(data)
        loss = nn.functional.nll_loss(output, target)

        # Proximal term
        proximal_term = 0.0
        for param, global_param in zip(client_model.parameters(), global_params):
            proximal_term += torch.sum((param - global_param) ** 2)

        loss += (mu / 2) * proximal_term
        loss.backward()
        optimizer.step()


# FedAvg 聚合函数
def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


# 客户端更新函数
def client_update(client_model, global_model, optimizer, train_loader, device, mu, epochs=1):
    for epoch in range(epochs):
        train_local_model_prox(client_model, global_model, device, train_loader, optimizer, mu, epoch)


def server_aggregate(global_model, client_models):
    return federated_averaging(global_model, client_models)


# ECE计算函数
def calculate_ece(confidences, accuracies, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def test(model, device, test_loader):
    model.eval()
    correct = 0
    confidences = []
    accuracies = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob = torch.exp(output)  # Convert log probabilities to probabilities
            pred = prob.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Calculate confidence and accuracy for ECE
            confidence = prob.max(dim=1)[0]
            accuracies.extend(pred.eq(target.view_as(pred)).squeeze().cpu().numpy())
            confidences.extend(confidence.cpu().numpy())

    accuracy = 100. * correct / len(test_loader.dataset)
    ece = calculate_ece(np.array(confidences), np.array(accuracies))
    return accuracy, ece


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟客户端
num_clients = 20  # 总客户端数量
client_models = [Net().to(device) for _ in range(num_clients)]
global_model = Net().to(device)

# 设置优化器
optimizers = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9) for model in client_models]

# 收集accuracy和ECE的列表
global_accuracies = []
global_eces = []

# 联邦学习训练过程
global_epochs = 500
local_epochs = 5
clients_per_round = 10  # 每轮选择的客户端数量
mu = 0.1  # FedProx 正则化参数

for epoch in range(global_epochs):
    print(f"Global Epoch {epoch + 1}/{global_epochs}")

    # 随机选择10个客户端
    selected_clients = random.sample(range(num_clients), clients_per_round)

    # 在每个选中的客户端上进行本地训练
    for i in selected_clients:
        client_update(client_models[i], global_model, optimizers[i], train_loader, device, mu, local_epochs)

    # 在服务器端聚合模型
    selected_client_models = [client_models[i] for i in selected_clients]
    global_model = server_aggregate(global_model, selected_client_models)

    # 将全局模型更新到每个客户端
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    # 使用全局模型进行验证
    accuracy, ece = test(global_model, device, test_loader)
    global_accuracies.append(accuracy)
    global_eces.append(ece)
    print(f"Validation set: Accuracy: {accuracy:.2f}%, ECE: {ece:.4f}")

# 绘制accuracy和ECE的图表
epochs = range(1, global_epochs + 1)

plt.figure(figsize=(12, 5))

# Accuracy 图表
plt.subplot(1, 2, 1)
plt.plot(epochs, global_accuracies, 'b', label='Accuracy')
plt.title('Global Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

# ECE 图表
plt.subplot(1, 2, 2)
plt.plot(epochs, global_eces, 'r', label='ECE')
plt.title('Global Model ECE')
plt.xlabel('Epochs')
plt.ylabel('ECE')
plt.legend()

plt.tight_layout()
plt.show()
