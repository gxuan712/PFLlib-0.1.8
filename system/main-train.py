import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from system.flcore.trainmodel.subspace_meta_learner import train_subspace_meta_learner

# 读取 idx 文件的函数
def read_idx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"没有这样的文件或目录: '{file_path}'")
    with gzip.open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')
        if magic_number == 2049:  # 标签文件
            data = np.frombuffer(f.read(), dtype=np.uint8)
        elif magic_number == 2051:  # 图像文件
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
        else:
            raise ValueError(f"文件中的 magic number {magic_number} 无效: {file_path}")
    return data

# 元学习训练中的 SubspaceMetaLearner 示例使用
def main():
    # 超参数
    n = 784  # 输入维度 (28x28 图像展平)
    m = 50  # 子空间维度
    output_dim = 10  # 类别数量 (数字 0-9)
    epochs = 500  # 训练周期
    batch_size = 64  # 批处理大小
    learning_rate = 0.05  # 初始学习率

    # 数据增强变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # 加载数据
    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/train-labels-idx1-ubyte.gz')
    val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/t10k-images-idx3-ubyte.gz')
    val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/rawdata/MNIST/raw/t10k-labels-idx1-ubyte.gz')

    # 将数据转换为 PyTorch张量，并进行标准化处理
    train_images = torch.tensor(train_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_images = torch.tensor(val_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # 创建数据集并应用数据增强
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            image = image.squeeze(0)  # 移除单通道维度
            if self.transform:
                image = self.transform(image.numpy())
            return image, label

    train_loader = DataLoader(AugmentedDataset(train_dataset, transform=transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, output_dim)

        def forward(self, x):
            x = x.view(-1, self.fc1.in_features)  # 展平图像
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    models = [SimpleNN(n, output_dim) for _ in range(5)]  # 示例有 5 个模型

    # 生成潜在向量 v_k 的函数
    def generate_v_k(m):
        v_k = nn.Parameter(torch.randn(m) * 0.1)
        return nn.Parameter(v_k)

    # 训练 SubspaceMetaLearner
    train_subspace_meta_learner(models, train_loader, val_loader, n, m, generate_v_k, epochs, output_dim, learning_rate)

    # 保存训练后的模型
    for i, model in enumerate(models):
        model_path = f'model_{i}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model {i} saved to {model_path}')

if __name__ == '__main__':
    main()
