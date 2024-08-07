import numpy as np
import torch
import torch.nn as nn
import gzip
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

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

def main():
    # 超参数
    n = 784  # 输入维度 (28x28 图像展平)
    output_dim = 10  # 类别数量 (数字 0-9)
    batch_size = 128  # 批处理大小

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])

    # 加载数据
    test_images = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-images-idx3-ubyte.gz')
    test_labels = read_idx('C:/Users/97481/Desktop/PFLlib-0.1.8/dataset/MNIST/test/t10k-labels-idx1-ubyte.gz')

    # 转换数据
    test_images = torch.tensor(test_images, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            x = x.view(-1, self.fc.in_features)  # 展平图像
            x = self.fc(x)
            return x

    model = SimpleNN(n, output_dim)

    # 使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 加载训练好的模型
    model_path = 'model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')
    else:
        raise FileNotFoundError(f"模型文件 {model_path} 未找到。请先训练模型。")

    # 测试模型
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'测试损失: {test_loss:.4f}, 测试准确率: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
