import time
import torch
import torch.optim as optim
import torch.nn as nn
from system.flcore.clients.clientala import clientALA
from system.flcore.servers.serverbase import Server
from system.flcore.trainmodel.models import SubspaceMetaLearner, orthogonalize

class serversBayesian(Server):
    def __init__(self, args, times, n, m, meta_epochs=100):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # Initialize the subspace meta-learner
        self.meta_learner = SubspaceMetaLearner(n, m)
        self.meta_optimizer = optim.Adam(self.meta_learner.parameters())
        self.meta_epochs = meta_epochs

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # Train subspace meta-learner with the collected client models and datasets
            models = [client.model for client in self.clients]
            train_sets = [(client.train_data[0], client.train_data[1]) for client in self.clients]
            val_sets = [(client.val_data[0], client.val_data[1]) for client in self.clients]

            self.train_subspace_meta_learner(models, train_sets, val_sets, self.meta_learner, self.meta_optimizer, self.meta_epochs)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientALA)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model)

    def train_subspace_meta_learner(self, models, train_sets, val_sets, meta_learner, optimizer, epochs=100):
        for epoch in range(epochs):
            meta_loss = 0
            for model, train_set, val_set in zip(models, train_sets, val_sets):
                q_v = torch.distributions.Normal(torch.zeros(meta_learner.m), torch.ones(meta_learner.m))
                v_k = q_v.rsample()
                w_k = meta_learner(v_k)

                # Assuming the model has a set_weights method
                model.set_weights(w_k)

                criterion = nn.MSELoss()
                optimizer_model = optim.SGD(model.parameters())

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
