import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientGEM(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        

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
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs

        grads = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.requires_grad], dim=0)  
        pm = torch.zeros_like(grads)
        normal_output=(torch.ones(self.num_classes) / self.num_classes).to(self.device)
        
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            
            output = self.model(x)
            
            if(self.unlearning):
                output=F.softmax(output,dim=-1)
                loss=F.kl_div(normal_output.log(), output, reduction='batchmean')*10
            else:
                loss = self.loss(output, y)
            loss.backward(retain_graph=True)
            pm += torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.requires_grad], dim=0)
            if(self.unlearning==False and i > int(len(trainloader)*0.1)):
                return pm/(i+1)
        return (pm/len(trainloader))
