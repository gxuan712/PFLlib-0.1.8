import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import gzip
import numpy as np

# 读取IDX文件的函数
def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        if (magic_number == 2051):  # 图像
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        elif (magic_number == 2049):  # 标签
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
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1) / 255.0
val_labels = torch.tensor(val_labels, dtype=torch.long)

# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 客户端训练函数
def client_update(model, optimizer, train_loader, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict(), loss.item()

# 服务器端聚合函数
def server_aggregate(global_model, client_weights):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_weights[i][k] for i in range(len(client_weights))], dim=0).mean(dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

# 数据加载器和划分函数
def get_data_loaders(train_images, train_labels, num_clients):
    dataset = TensorDataset(train_images, train_labels)
    client_datasets = random_split(dataset, [len(dataset) // num_clients for _ in range(num_clients)])
    return [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in client_datasets]

# 测试模型函数
def test_model(model, test_images, test_labels):
    model.eval()
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=32, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# 联邦学习主函数
def train_and_test(num_rounds=10, num_clients=5, num_epochs=1):
    global_model = SimpleNN()
    train_loaders = get_data_loaders(train_images, train_labels, num_clients)

    for round in range(num_rounds):
        client_weights = []
        for client_id in range(num_clients):
            model = SimpleNN()
            model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            client_model_weights, _ = client_update(model, optimizer, train_loader=train_loaders[client_id], epochs=num_epochs)
            client_weights.append(client_model_weights)

        global_model = server_aggregate(global_model, client_weights)
        accuracy = test_model(global_model, val_images, val_labels)
        print(f"Round {round+1}, Accuracy: {accuracy:.2f}%")

# 执行联邦学习
train_and_test(num_rounds=100, num_clients=5, num_epochs=1)
