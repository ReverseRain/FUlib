
import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientROME(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.proxy_loader=None
        

        

    def train(self):
        trainloader=self.train_loader
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50.0)
                self.optimizer.step()


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def unlearning_train(self):
        trainloader=self.train_loader
        self.model.train()
        normal_output=(torch.ones(trainloader.batch_size, self.num_classes)/self.num_classes).to(self.device)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for (x, y),(x_pro) in zip(trainloader,self.proxy_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if type(x_pro) == type([]):
                    x_pro[0] = x_pro[0].to(self.device)
                else:
                    x_pro = x_pro.to(self.device)
                
                k_star = self.model.base(x)
                k_norm = self.model.base(x_pro)
                output=F.softmax(self.model.head(k_star),dim=-1)

                # 梯度方向估计
                C = torch.matmul(k_norm, k_norm.T)  # [B, B]
                C_inv = torch.inverse(C + 1e-6 * torch.eye(C.size(0)).to(self.device))
                u = torch.matmul(C_inv, k_star)  # [B, D]

                delta_out = normal_output - output  # [B, C]
                grad_approx = torch.matmul(delta_out.T, u)  # [C, D]

                # 更新分类头
                with torch.no_grad():
                    self.model.head.weight += 0.0005 * grad_approx

                
                
                # loss=F.kl_div(normal_output.log(), output, reduction='batchmean')
                # # loss=self.NPOLoss(output,y)
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
        