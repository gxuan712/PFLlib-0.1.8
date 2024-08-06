import numpy as np
import torch
import torch.nn as nn
import gzip
import os


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


def evaluate(models, test_images, test_labels):
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            outputs = model(test_images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == test_labels).sum().item()
            total = test_labels.size(0)
            accuracy = correct / total
            print(f"模型 {i + 1}: 测试准确率: {accuracy * 100:.2f}%")


def main():
    # 超参数
    n = 784  # 输入维度 (28x28 图像展平)
    output_dim = 10  # 类别数量 (数字 0-9)
    hidden_dim = 128  # 隐藏层维度

    # 加载数据
    test_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    test_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    # 将数据转换为 PyTorch 张量
    test_images = torch.tensor(test_images, dtype=torch.float32).view(-1, n) / 255.0
    test_labels = torch.tensor(test_labels, dtype=torch.long)

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

    models = [SimpleNN(n, hidden_dim, output_dim) for _ in range(5)]

    # 加载训练后的模型
    for i, model in enumerate(models):
        model_path = f'model_{i}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 未找到，请确认训练是否完成并正确保存模型。")
        model.load_state_dict(torch.load(model_path))
        print(f'Model {i} loaded from {model_path}')

    # 测试模型
    evaluate(models, test_images, test_labels)


if __name__ == '__main__':
    main()
