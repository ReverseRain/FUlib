
import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client


class clientBU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.opt_un== torch.optim.SGD(self.teacher_model.parameters(), lr=self.learning_rate)
        

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
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
                self.optimizer.step()


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    def unlearning_train(self):
        
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()
        
        max_local_epochs = self.local_epochs
        # 获得没有毒数据的正常训练集
        trainloader = self.getCleanTrain()
        poison_loader= DataLoader(self.train_poision, self.batch_size, drop_last=True, shuffle=True)
        
        gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
        for epoch in range(max_local_epochs):
            for (x_pois, y_pois),(x,y) in zip(poison_loader,trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                clean_loss = self.loss(output, y)

                if type(x_pois) == type([]):
                    x_pois[0] = x_pois[0].to(self.device)
                else:
                    x_pois = x_pois.to(self.device)
                y_pois = y_pois.to(self.device)
                
                output_pois = self.model(x_pois)
                pois_loss = self.loss(output_pois, y_pois)

                total_loss=(clean_loss-pois_loss)
                clean_loss.backward(retain_graph=True)
                clean_importance = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.requires_grad], dim=0)

                # Calculate importance of backdoor data gradients
                pois_loss.backward(retain_graph=True)
                pois_importance = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.requires_grad], dim=0)
                
                importance = torch.nan_to_num(torch.div(clean_importance, pois_importance),1e-12)

                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                penalty = torch.norm(importance * torch.abs((pm-gm)), 1)
                
                total_loss+=penalty*0.6
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)
                self.optimizer.step()


    def getCleanTrain(self):
        self.unlearning=False
        train_loader=self.load_train_data()
        self.unlearning=True
        return train_loader