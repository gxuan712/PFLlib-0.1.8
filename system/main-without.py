import torch
import torch.nn as nn
import gzip
import numpy as np


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
num_clients = 5
client_train_images = torch.chunk(train_images, num_clients)
client_train_labels = torch.chunk(train_labels, num_clients)

# 超参数
n = 784  # MNIST图像展平后的大小 (28*28)
m = 50  # 低维空间的大小

# 初始化全局操作矩阵O，但不进行优化
O_global = nn.Parameter(torch.randn(n, m))


# 计算损失
def compute_loss(O, client_images):
    loss = 0
    with torch.no_grad():  # 不进行梯度计算
        for i in range(len(client_images)):
            d = client_images[i].view(-1, 1)  # 重塑数据点 d (784, 1)
            prediction = torch.matmul(O.T, d)  # prediction 的维度是 (m, 1)
            loss += torch.mean(prediction ** 2)  # 计算损失
    avg_loss = loss.item() / len(client_images)
    return avg_loss


# 模拟传输并计算loss，不进行优化
def simulate_transmission(O, num_clients=5):
    total_loss = 0
    for client_id in range(num_clients):
        client_loss = compute_loss(O, client_train_images[client_id])
        total_loss += client_loss

    # 计算平均loss
    avg_loss = total_loss / num_clients
    return avg_loss


# 联邦学习的轮次
num_rounds = 500

for round in range(num_rounds):
    avg_loss = simulate_transmission(O_global)
    print(f"Average Loss after Round {round + 1}: {avg_loss:.4f}")

print("Final Average Loss after all rounds:", avg_loss)
