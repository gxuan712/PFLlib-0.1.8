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
    epochs = 100  # 训练周期
    batch_size = 128  # 批处理大小
    learning_rate = 0.001  # 初始学习率
    weight_decay = 1e-5  # L2正则化

    # 数据增强
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])

    # 加载数据
    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
    val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    # 转换数据并进行数据增强
    train_images = torch.tensor(train_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_images = torch.tensor(val_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = x.view(-1, self.fc.in_features)  # 展平图像
            x = self.fc(x)
            return x

    model = SimpleNN(n, output_dim)  # 只有 1 个模型

    # 使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练 SubspaceMetaLearner
    train_subspace_meta_learner(model, train_loader, val_loader, n, m, epochs, output_dim, learning_rate, device, weight_decay)

    # 保存训练后的模型
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
