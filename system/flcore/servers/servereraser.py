# server.py
import torch
import copy
import numpy as np

class Server:
    def __init__(self, global_model, clients, test_loader, args):
        self.global_model = global_model
        self.clients = clients
        self.test_loader = test_loader
        self.args = args

    def distribute_model(self, selected_clients):
        for client in selected_clients:
            client.update_model(self.global_model)

    def aggregate_models(self, selected_clients):
        """FedAvg 聚合：平均客户端的模型参数"""
        total_clients = len(selected_clients)
        new_state = copy.deepcopy(self.global_model.state_dict())

        # 初始化为 0
        for key in new_state:
            new_state[key] = torch.zeros_like(new_state[key])

        # 累加每个客户端的参数
        for client in selected_clients:
            client_state = client.upload_model()
            for key in new_state:
                new_state[key] += client_state[key]

        # 取平均
        for key in new_state:
            new_state[key] /= total_clients

        # 更新全局模型
        self.global_model.load_state_dict(new_state)

    def evaluate(self):
        """在测试集上评估全局模型"""
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total
        print(f"[Server] Test Accuracy: {acc:.4f}")
