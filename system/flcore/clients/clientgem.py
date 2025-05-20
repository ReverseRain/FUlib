import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from flcore.clients.clientbase import Client


class clientGEM(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.proxy_loader=None
        self.paired_loader=None
        self.first_time=True
    def train(self,poison=False):
        trainloader=self.train_loader
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
        
        trainloader=self.paired_loader
        self.model.train()

        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output=self.model(x)
            loss=self.pairLoss(output,y)
            
            self.optimizer_ul.zero_grad()
            loss.backward()
            self.optimizer_ul.step()


    def NPOLoss(self,pred,target):
        class_num = int(pred.shape[1])
        pred = F.softmax(pred, dim=-1)
        target_enc = F.one_hot(target, class_num)
        # bata=2
        # loss = 1/bata * torch.mean((torch.log(1+(torch.sum(pred * target_enc, dim=1)*class_num)**bata)))
        loss = torch.mean(torch.log(10+(torch.sum(pred * target_enc, dim=1))))
        return loss
    def UnLearningCELoss(self,pred,target):
        class_num = int(pred.shape[1])
        target_enc = F.one_hot(target, class_num)
        pred = F.softmax(pred, dim=-1)
        loss = -torch.mean(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))
        
        return loss
    def sigLoss(self,pred,target):
        class_num = int(pred.shape[1])
        batch=int(pred.shape[0])
        pred = F.softmax(pred, dim=-1)
        target_enc = F.one_hot(target, class_num).to(dtype=torch.float32)        
        t=1.5
        loss=(-1*torch.mean(torch.log(torch.sigmoid(-1*torch.sum(torch.mul(pred , target_enc /t),dim=1)))))
        return loss
    def pairLoss(self,pred,target):
        pred = F.softmax(pred, dim=-1)
        t=1.2

        loss=(-1*torch.mean(torch.log(torch.sigmoid(-1*torch.sum(torch.mul(pred , target /t),dim=1)))))
        return loss
    
    def getPairLoader(self):
        x_all=[]
        output_all=[]
        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                output = self.model(x)
                output = F.softmax(output, dim=-1)

                x_all.append(x.cpu())
                output_all.append(output.cpu())

        x_all = torch.cat(x_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

        paired_data=[(x,y) for x,y in zip(x_all,output_all)]
        self.paired_loader=DataLoader(paired_data, self.batch_size, drop_last=True, shuffle=True)

