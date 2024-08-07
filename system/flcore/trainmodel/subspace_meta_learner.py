import torch
import torch.nn as nn

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
    # 正交化矩阵
    q, _ = torch.linalg.qr(matrix)
    return q

def train_subspace_meta_learner(model, train_loader, val_loader, n, m, epochs=500, output_dim=10, learning_rate=0.01,
                                momentum=0.9, device="cpu", weight_decay=1e-5):
    hidden_dim = 128
    meta_learner = SubspaceMetaLearner(n, m, hidden_dim).to(device)
    optimizer = torch.optim.SGD(meta_learner.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)  # 使用 SGD 优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # 每100个epoch将学习率减少为原来的10%

    for epoch in range(epochs):
        meta_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 为元学习者执行训练步骤
            optimizer.zero_grad()
            v_k = torch.randn(m).to(device)  # 示例潜在向量 v_k
            w_k1, w_k2 = meta_learner(v_k)

            if w_k1.numel() != n * hidden_dim:
                raise ValueError(f"Generated weight matrix size mismatch for fc1, w_k1 size is {w_k1.numel()}")

            if w_k2.numel() != hidden_dim * output_dim:
                raise ValueError(f"Generated weight matrix size mismatch for fc2, w_k2 size is {w_k2.numel()}")

            model.fc1.weight.data = w_k1.view(hidden_dim, n)
            model.fc2.weight.data = w_k2.view(output_dim, hidden_dim)

            # 前向传播
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), max_norm=1.0)

            optimizer.step()

            meta_loss += loss.item()

        scheduler.step()  # 更新学习率

        # 打印日志
        print(f"Epoch {epoch + 1}/{epochs}, Meta Loss: {meta_loss}")

        # 验证模型
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
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {accuracy:.2f}%')

