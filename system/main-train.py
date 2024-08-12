import torch
import torch.nn as nn
import torch.optim as optim
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
train_images = train_images.view(-1, 28*28)
val_images = val_images.view(-1, 28*28)

# 将数据划分为5个客户端的数据集
num_clients = 5
client_train_images = torch.chunk(train_images, num_clients)
client_train_labels = torch.chunk(train_labels, num_clients)

# 超参数
n = 784  # MNIST图像展平后的大小 (28*28)
m = 50   # 低维空间的大小

# 初始化全局操作矩阵O
O_global = nn.Parameter(torch.randn(n, m))

# 正交化O (O^T O = I)
def orthogonalize(O):
    with torch.no_grad():
        u, _, v = torch.svd(O, some=True)
        return torch.matmul(u, v.T)

# 服务器端聚合从客户端返回的O，并更新全局O
def aggregate_O(O_list):
    O_new = torch.mean(torch.stack(O_list), dim=0)
    # 正交化更新后的O
    O_orthogonal = orthogonalize(O_new)
    return O_orthogonal

# 模拟服务器与客户端之间的通信
def server_side(O, num_clients=5):
    O_list = []
    for client_id in range(num_clients):
        # 模拟客户端的训练，并返回更新后的O
        O_k = client_side(O, client_train_images[client_id], client_train_labels[client_id])
        O_list.append(O_k)

    # 聚合客户端的O并更新全局O
    O_global = aggregate_O(O_list)
    return O_global

# 客户端接收O并进行优化
def client_side(O, client_images, client_labels):
    # 初始化客户端的低维个性化权重v_k
    v_k = nn.Parameter(torch.randn(m, 1))

    # 定义v_k和O的优化器
    optimizer = optim.Adam([O, v_k], lr=0.01)

    num_local_epochs = 5
    for epoch in range(num_local_epochs):
        optimizer.zero_grad()

        # 计算预测值 O * v_k
        Ov_k = torch.matmul(O, v_k)  # Ov_k 的维度是 (n, 1)

        # 计算训练集上的损失函数
        loss = 0
        for i in range(len(client_images)):
            d = client_images[i].view(-1, 1)  # 重塑数据点 d (784, 1)
            prediction = torch.matmul(O.T, d)  # prediction 的维度是 (m, 1)
            loss += torch.mean((prediction - v_k) ** 2)  # 计算损失

        # 优化 v_k 和 O
        loss.backward()
        optimizer.step()

    return O.detach()  # 返回优化后的O给服务器

# 初始化O的正交化
O_global = orthogonalize(O_global)

# 联邦学习的轮次和客户端数量
num_rounds = 500  # 联邦学习的轮次

for round in range(num_rounds):
    print(f"Round {round + 1}/{num_rounds}")
    O_global = server_side(O_global)

print("Final O after training:", O_global)
