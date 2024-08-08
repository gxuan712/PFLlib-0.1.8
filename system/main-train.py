import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms

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

# 定义SubspaceMetaLearner模型
class SubspaceMetaLearner(nn.Module):
    def __init__(self, n, m, hidden_dim):
        super(SubspaceMetaLearner, self).__init__()
        self.n = n  # 输入维度
        self.m = m  # 子空间维度
        self.hidden_dim = hidden_dim
        # Initialize O matrix for both layers
        self.O1 = nn.Parameter(torch.randn(n * hidden_dim, m) * 0.01)  # For fc1
        self.O2 = nn.Parameter(torch.randn(hidden_dim * 10, m) * 0.01)  # For fc2

    def forward(self, v):
        if v.shape[0] != self.m:
            raise ValueError(f"Expected v of shape [{self.m}], but got {v.shape}")

        w_k1 = torch.matmul(self.O1, v)  # Compute w_k1 = O1 * v
        w_k2 = torch.matmul(self.O2, v)  # Compute w_k2 = O2 * v
        return w_k1, w_k2

def orthogonalize(matrix):
    # 正交化矩阵，使其满足 O^T * O = I
    q, _ = torch.linalg.qr(matrix)
    return q

# 定义SimpleNN模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)  # 展平成一维
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 训练本地模型
def train_local_model(model, train_loader, meta_learner, device, epochs, lr=0.01, weight_decay=1e-4):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # 生成随机低维向量 v
            v = torch.randn(meta_learner.m).to(device)

            # 通过元学习器获取权重
            w_k1, w_k2 = meta_learner(v)
            model.fc1.weight = nn.Parameter(w_k1.view(128, 28*28))
            model.fc2.weight = nn.Parameter(w_k2.view(10, 128))

            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

# 贝叶斯更新
def bayesian_update(global_model, local_models, sigma2):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        local_params = torch.stack([local_models[i].state_dict()[key] for i in range(len(local_models))], dim=0)
        global_param = global_model.state_dict()[key]
        updated_param = (local_params.var(dim=0) / sigma2 + 1) ** (-1) * (local_params.mean(dim=0) / sigma2 + global_param)
        global_dict[key] = updated_param
    global_model.load_state_dict(global_dict)

# 验证过程
def test(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy

# 联邦学习过程
def federated_training(global_model, train_loaders, val_loader, meta_learner, epochs, rounds, device, sigma2=1.0, lr=0.01):
    for round in range(rounds):
        local_models = [SimpleNN(28*28, 128, 10).to(device) for _ in range(len(train_loaders))]
        for i, train_loader in enumerate(train_loaders):
            local_models[i].load_state_dict(global_model.state_dict())
            train_local_model(local_models[i], train_loader, meta_learner, device, epochs, lr=lr)
        bayesian_update(global_model, local_models, sigma2)
        val_loss, accuracy = test(global_model, val_loader, device)
        print(f"Round {round+1}/{rounds} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 主函数
def main():
    # 超参数
    n = 28 * 28  # 输入维度（展平的28x28图像）
    hidden_dim = 128  # 隐藏层维度
    output_dim = 10  # 类别数（数字0-9）
    epochs = 10  # 本地训练轮数
    federated_rounds = 5  # 联邦学习轮数
    batch_size = 128  # 批量大小
    lr = 0.01  # 学习率
    weight_decay = 1e-4  # 权重衰减

    # 加载数据
    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
    val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    # 数据预处理和应用数据增强
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1) / 255.0
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # 创建数据集
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    # 将数据划分为多个客户端数据加载器
    num_clients = 5
    client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
    train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 使用GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化全局模型和子空间元学习器
    global_model = SimpleNN(n, hidden_dim, output_dim).to(device)
    meta_learner = SubspaceMetaLearner(n, 20, hidden_dim).to(device)

    # 进行联邦训练
    federated_training(global_model, train_loaders, val_loader, meta_learner, epochs, federated_rounds, device, lr=lr)

    # 验证模型
    val_loss, accuracy = test(global_model, val_loader, device)
    print(f'最终验证损失: {val_loss:.4f}, 最终验证准确率: {accuracy:.2f}%')

    # 保存训练好的模型
    model1_path = 'model.pth'
    torch.save(global_model.state_dict(), model1_path)
    print(f'Model saved to {model1_path}')

if __name__ == '__main__':
    main()
