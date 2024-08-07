import torch
import torch.nn as nn


class SubspaceMetaLearner(nn.Module):
    def __init__(self, n, m):
        super(SubspaceMetaLearner, self).__init__()
        self.n = n  # 输入维度
        self.m = m  # 子空间维度
        self.O = nn.Parameter(torch.randn(n * 10, m) * 0.01)  # 初始化 O 矩阵 (n * output_dim, m)

    def forward(self, v):
        # v 的形状应为 [m]
        if v.shape[0] != self.m:
            raise ValueError(f"期望 v 的形状为 [{self.m}]，但得到的形状为 {v.shape}")
        w_k = torch.matmul(self.O, v)  # 计算 w_k = O * v
        return w_k


def orthogonalize(matrix):
    # 正交化矩阵
    q, _ = torch.linalg.qr(matrix)
    return q


def train_subspace_meta_learner(model, train_loader, val_loader, n, m, epochs=500, output_dim=10, learning_rate=0.001,
                                device="cpu", weight_decay=1e-5):
    meta_learner = SubspaceMetaLearner(n, m).to(device)
    optimizer = torch.optim.AdamW(meta_learner.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)  # 使用 AdamW 优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # 每100个epoch将学习率减少为原来的10%

    for epoch in range(epochs):
        meta_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 为元学习者执行训练步骤
            optimizer.zero_grad()
            v_k = torch.randn(m).to(device)  # 示例潜在向量 v_k
            w_k = meta_learner(v_k)
            if w_k.numel() != n * output_dim:
                raise ValueError(f"生成的权重矩阵大小不匹配，w_k 的大小为 {w_k.numel()}")

            model.fc.weight.data = w_k.view(output_dim, n)

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
