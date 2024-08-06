import torch
import torch.nn as nn


class SubspaceMetaLearner(nn.Module):
    def __init__(self, n, m):
        super(SubspaceMetaLearner, self).__init__()
        self.n = n  # 输入维度
        self.m = m  # 子空间维度
        self.O = nn.Parameter(torch.randn(n * 128, m))  # 初始化 O 矩阵 (n * hidden_dim, m)

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


def train_subspace_meta_learner(models, train_sets, val_sets, n, m, epochs=100, output_dim=10, hidden_dim=128):
    meta_learner = SubspaceMetaLearner(n, m)

    for epoch in range(epochs):
        meta_loss = 0
        for model, train_set, val_set in zip(models, train_sets, val_sets):
            train_images, train_labels = train_set
            val_images, val_labels = val_set

            # 为元学习者执行训练步骤
            v_k = torch.randn(m)  # 示例潜在向量 v_k
            w_k = meta_learner(v_k)
            if w_k.numel() != hidden_dim * n:
                raise ValueError(f"生成的权重矩阵 w_k 的大小为 {w_k.numel()}，但预期大小为 {hidden_dim * n}")
            model_weight = w_k.view(hidden_dim, n)  # 重塑以匹配模型的权重形状
            model.fc1.weight.data = model_weight

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 使用 SGD 优化器
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(train_images), train_labels)
            loss.backward()
            optimizer.step()

            meta_loss += loss.item()

        # 正交化 O
        with torch.no_grad():
            meta_learner.O.data = orthogonalize(meta_learner.O.data)

        print(f"Epoch {epoch + 1}/{epochs}, Meta Loss: {meta_loss / len(models)}")

        # 打印每个模型的验证损失和准确率
        for i, (model, val_set) in enumerate(zip(models, val_sets)):
            val_images, val_labels = val_set
            model.eval()
            with torch.no_grad():
                outputs = model(val_images)
                val_loss = nn.CrossEntropyLoss()(outputs, val_labels)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == val_labels).sum().item()
                total = val_labels.size(0)
                accuracy = correct / total
                print(f"模型 {i + 1}: 验证损失: {val_loss.item()}, 验证准确率: {accuracy * 100:.2f}%")

