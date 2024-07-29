import torch
import torch.optim as optim
import torch.nn as nn
from system.flcore.trainmodel.models import SubspaceMetaLearner, orthogonalize

def train_subspace_meta_learner(models, train_sets, val_sets, n, m, epochs=100):
    # 初始化子空间元学习模型和优化器
    meta_learner = SubspaceMetaLearner(n, m)
    optimizer = optim.Adam(meta_learner.parameters())

    for epoch in range(epochs):
        meta_loss = 0
        for model, train_set, val_set in zip(models, train_sets, val_sets):
            q_v = torch.distributions.Normal(torch.zeros(m), torch.ones(m))  # 示例变分分布
            v_k = q_v.rsample()  # 从 q(v_k) 中采样 v_k
            w_k = meta_learner(v_k)  # 使用子空间元学习模型计算 w_k = O * v_k

            model.load_state_dict({'weight': w_k})  # 将 w_k 加载到当前模型中

            # 定义简单的损失函数并计算训练和验证损失
            criterion = nn.MSELoss()
            optimizer_model = optim.SGD(model.parameters())

            # 计算训练损失
            model.train()
            optimizer_model.zero_grad()
            train_outputs = model(train_set[0])
            train_loss = criterion(train_outputs, train_set[1])
            train_loss.backward()
            optimizer_model.step()

            # 计算验证损失
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_set[0])
                val_loss = criterion(val_outputs, val_set[1])

            meta_loss += val_loss  # 累积元损失

        optimizer.zero_grad()  # 清零优化器的梯度
        meta_loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数 O 以最小化损失

        # 保证 O 矩阵的正交性
        with torch.no_grad():
            meta_learner.O.copy_(orthogonalize(meta_learner.O))

        # 输出每个 epoch 的损失
        print(f"Epoch {epoch + 1}, Meta Loss: {meta_loss.item()}")

    return meta_learner
