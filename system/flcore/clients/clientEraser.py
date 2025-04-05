# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Client:
    def __init__(self, client_id, model, train_loader, args):
        self.id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.args = args

    def update_model(self, global_model):
        """从服务器接收全局模型参数"""
        self.model.load_state_dict(global_model.state_dict())

    def train(self):
        """在本地数据上训练模型"""
        self.model.train()
        device = next(self.model.parameters()).device
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.local_lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.args.local_epoch):
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

    def upload_model(self):
        """将本地训练后的模型上传到服务器"""
        return copy.deepcopy(self.model.state_dict())
