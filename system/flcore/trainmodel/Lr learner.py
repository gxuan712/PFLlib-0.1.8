import torch
import torch.optim as optim
from system.flcore.trainmodel.models import SubspaceMetaLearner, orthogonalize

def glorot_init(n, m):
    std_dev = 1.0 / torch.sqrt(torch.tensor(n + m, dtype=torch.float32))
    return nn.Parameter(torch.randn(n, m) * std_dev)

def update_O(O, lr):
    P = O @ O.T
    Q = P - torch.eye(P.size(0), device=O.device)
    with torch.no_grad():
        O -= lr * Q @ O
    return O

def compute_loss(O, v_k_list, D_v_list, D_t_list):
    loss = 0.0
    K = len(v_k_list)
    for k in range(K):
        v_k = v_k_list[k]
        D_v_k = D_v_list[k]
        D_t_k = D_t_list[k]

        # Validation loss
        validation_loss = torch.mean(torch.stack([f(O @ v_k, d) for d in D_v_k]))

        # Training loss for variational distribution q(v_k)
        q_v_k = find_optimal_q(O, v_k, D_t_k)
        training_loss = torch.mean(torch.stack([f(O @ q_v_k, d) for d in D_t_k]))

        loss += validation_loss + training_loss
    return loss / K

def f(Ov, d):
    return torch.norm(Ov - d)

def find_optimal_q(O, v_k, D_t_k):
    return v_k

def train_O(O, v_k_list, D_v_list, D_t_list, learning_rate=1e-3, epochs=1000, print_interval=100):
    optimizer = optim.SGD([O], lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(O, v_k_list, D_v_list, D_t_list)
        loss.backward()
        optimizer.step()
        O = update_O(O, learning_rate)
        if epoch % print_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return O

def train_subspace_meta_learner(models, train_sets, val_sets, n, m, lr=0.01, epochs=100):
    # 初始化子空间元学习模型和优化器
    O = glorot_init(n, m)
    meta_learner = SubspaceMetaLearner(n, m)
    optimizer = optim.Adam(meta_learner.parameters(), lr=lr)

    for epoch in range(epochs):
        meta_loss = 0
        for model, train_set, val_set in zip(models, train_sets, val_sets):
            q_v = torch.distributions.Normal(torch.zeros(m), torch.ones(m))
            v_k = q_v.rsample()
            w_k = meta_learner(v_k)

            model.load_state_dict({'weight': w_k})

            criterion = nn.MSELoss()
            optimizer_model = optim.SGD(model.parameters(), lr=lr)

            model.train()
            optimizer_model.zero_grad()
            train_outputs = model(train_set[0])
            train_loss = criterion(train_outputs, train_set[1])
            train_loss.backward()
            optimizer_model.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(val_set[0])
                val_loss = criterion(val_outputs, val_set[1])

            meta_loss += val_loss

        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        with torch.no_grad():
            meta_learner.O.copy_(orthogonalize(meta_learner.O))

        print(f"Epoch {epoch + 1}, Meta Loss: {meta_loss.item()}")

    return meta_learner
