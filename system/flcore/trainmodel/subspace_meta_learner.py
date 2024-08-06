import torch
import torch.nn as nn
import torch.optim as optim


class SubspaceMetaLearner(nn.Module):
    def __init__(self, n, m):
        super(SubspaceMetaLearner, self).__init__()
        self.n = n  # 输入维度
        self.m = m  # 子空间维度
        total_weights = n * 512 + 512 * 256 + 256 * 10  # 计算所有层的总权重
        self.O = nn.Parameter(torch.randn(total_weights, m) * 0.01)  # 初始化 O 矩阵

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


def generate_v_k(device, m):
    v_k = nn.Parameter(torch.randn(m, device=device) * 0.01)
    return v_k


def train_subspace_meta_learner(models, train_loader, val_loader, n, m, generate_v_k, epochs=500, output_dim=10, learning_rate=0.05):
    meta_learner = SubspaceMetaLearner(n, m)
    optimizer = optim.AdamW(meta_learner.parameters(), lr=learning_rate)  # 使用 AdamW 优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 每50个epoch将学习率减少为原来的10%

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        meta_loss = 0
        for model in models:
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # 为元学习者执行训练步骤
                optimizer.zero_grad()
                v_k = generate_v_k(m)  # 使用生成函数生成潜在向量 v_k
                w_k = meta_learner(v_k)
                if w_k.numel() != meta_learner.O.shape[0]:
                    raise ValueError(f"生成的权重矩阵大小不匹配，w_k 的大小为 {w_k.numel()}")

                # 更新模型参数
                offset = 0
                model.fc1.weight = nn.Parameter(w_k[offset:offset + 512*n].view(512, n))
                offset += 512 * n
                model.fc2.weight = nn.Parameter(w_k[offset:offset + 256*512].view(256, 512))
                offset += 256 * 512
                model.fc3.weight = nn.Parameter(w_k[offset:offset + 10*256].view(10, 256))

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
        print(f"Epoch {epoch + 1}/{epochs}, Meta Loss: {meta_loss / len(models)}")

        # 验证模型
        for i, model in enumerate(models):
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += nn.CrossEntropyLoss()(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(val_loader.dataset)
            accuracy = 100. * correct / len(val_loader.dataset)
            print(f'模型 {i + 1}: 验证损失: {val_loss:.8f}, 验证准确率: {accuracy:.2f}%')

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = meta_learner.state_dict()

    # 恢复最佳模型状态
    if best_state is not None:
        meta_learner.load_state_dict(best_state)
