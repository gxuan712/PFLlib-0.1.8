import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

train_loader = DataLoader(AugmentedDataset(train_dataset, transform=transform), batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

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

# 创建单个模型实例
model = SimpleNN(784, 10)
optimizer = optim.AdamW(model.parameters(), lr=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# 训练和验证
def train_and_validate(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()  # 更新学习率

        # 打印训练损失
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}")

        # 验证模型
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

# 主要执行函数
if __name__ == '__main__':
    train_and_validate(model, train_loader, val_loader, epochs=50)

    # 保存模型
    torch.save(model.state_dict(), 'simple_nn_model.pth')
    print('Model saved to simple_nn_model.pth')
