import numpy as np
import torch
import torch.nn as nn
import gzip
import os
from torch.utils.data import DataLoader, TensorDataset
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
    hidden_dim = 128  # 隐藏层维度
    epochs = 200  # 增加训练周期
    batch_size = 64  # 增加批处理大小

    # 加载数据
    train_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-images-idx3-ubyte.gz')
    train_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/train/train-labels-idx1-ubyte.gz')
    val_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    val_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    # 将数据转换为 PyTorch 张量，并进行标准化处理
    train_images = torch.tensor(train_images, dtype=torch.float32).view(-1, n) / 255.0
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_images = torch.tensor(val_images, dtype=torch.float32).view(-1, n) / 255.0
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=batch_size, shuffle=False)

    # 初始化模型
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.output_dim = output_dim

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    models = [SimpleNN(n, hidden_dim, output_dim) for _ in range(5)]  # 示例有 5 个模型

    # 示例数据集 (你应当用实际数据集替换此示例数据集)
    train_sets = [(train_images, train_labels) for _ in range(5)]
    val_sets = [(val_images, val_labels) for _ in range(5)]

    # 训练 SubspaceMetaLearner
    train_subspace_meta_learner(models, train_sets, val_sets, n, m, epochs, output_dim, hidden_dim)

    # 保存训练后的模型
    for i, model in enumerate(models):
        model_path = f'model_{i}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model {i} saved to {model_path}')


if __name__ == '__main__':
    main()
