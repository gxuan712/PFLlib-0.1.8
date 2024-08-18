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
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# 定义训练和测试函数
def train_local_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


# FedAvg 算法
def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                     0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def client_update(client_model, optimizer, train_loader, device, epochs=1):
    for epoch in range(epochs):
        train_local_model(client_model, device, train_loader, optimizer, epoch)


def server_aggregate(global_model, client_models):
    return federated_averaging(global_model, client_models)


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟客户端
num_clients = 20  # 总客户端数量
client_models = [Net().to(device) for _ in range(num_clients)]
global_model = Net().to(device)

# 设置优化器
optimizers = [optim.SGD(model.parameters(), lr=0.01, momentum=0.9) for model in client_models]

# 收集loss和accuracy的列表
global_losses = []
global_accuracies = []

# 联邦学习训练过程
global_epochs = 10
local_epochs = 2
clients_per_round = 10  # 每轮选择的客户端数量

for epoch in range(global_epochs):
    print(f"Global Epoch {epoch + 1}/{global_epochs}")

    # 随机选择10个客户端
    selected_clients = random.sample(range(num_clients), clients_per_round)

    # 在每个选中的客户端上进行本地训练
    for i in selected_clients:
        client_update(client_models[i], optimizers[i], train_loader, device, local_epochs)

    # 在服务器端聚合模型
    selected_client_models = [client_models[i] for i in selected_clients]
    global_model = server_aggregate(global_model, selected_client_models)

    # 将全局模型更新到每个客户端
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    # 使用全局模型进行验证
    test_loss, accuracy = test(global_model, device, test_loader)
    global_losses.append(test_loss)
    global_accuracies.append(accuracy)
    print(f"Validation set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 绘制loss和accuracy的图表
epochs = range(1, global_epochs + 1)

plt.figure(figsize=(12, 5))

# Loss 图表
plt.subplot(1, 2, 1)
plt.plot(epochs, global_losses, 'r', label='Loss')
plt.title('Global Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy 图表
plt.subplot(1, 2, 2)
plt.plot(epochs, global_accuracies, 'b', label='Accuracy')
plt.title('Global Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
