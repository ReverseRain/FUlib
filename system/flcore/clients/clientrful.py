
import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client


class clientRFUL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.local_teacher_model = None

        self.getShuffledLoader()

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
        if(self.local_teacher_model == None):
            self.local_teacher_model = copy.deepcopy(self.model)
        
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(self.shuffled_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                present_classes = torch.unique(y)
                output = self.model(x)
                output_teacher = self.local_teacher_model(x)
                loss = self.loss(output, y) + 0.15*self.kl_divergence(output_teacher[:,present_classes],output[:,present_classes])
                self.optimizer_ul.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=15.0)
                self.optimizer_ul.step()


        


    def kl_divergence(self, p, q):
        # 确保是概率分布
        p = F.softmax(p, dim=-1) if p.dim() > 1 else F.softmax(p, dim=0)
        q = F.softmax(q, dim=-1) if q.dim() > 1 else F.softmax(q, dim=0)
        
        # 添加小值防止log(0)
        eps = 1e-8
        p = p + eps
        q = q + eps
        
        # 计算KL散度: sum(p * log(p/q))
        return (p * (p.log() - q.log())).sum()
    
    def getShuffledLoader(self):
        x_all=[]
        y_all=[]
        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                x_all.append(x.cpu())
                y_all.append(y.cpu())

        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)

        indices = torch.randperm(len(y_all))
        shuffled_y = y_all[indices]
        shuffled_y = (shuffled_y - 2) % self.args.num_classes

        shuffled_data=[(x,y) for x,y in zip(x_all,shuffled_y)]
        self.shuffled_loader=DataLoader(shuffled_data, self.batch_size, drop_last=True, shuffle=True)