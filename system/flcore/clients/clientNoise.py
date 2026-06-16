
import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import random

class clientNoise(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        

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
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = -self.loss(output, y)
                self.optimizer_ul.zero_grad()
                loss.backward()
                self.optimizer_ul.step()

    def getNoiseLoader(self):
        x_all=[]
        y_all=[]
        
        means = torch.tensor([0.5, 0.3, 0.7]).view(3, 1, 1).to(self.device)
        stds = torch.tensor([0.1, 0.2, 0.85]).view(3, 1, 1).to(self.device)
        
        u_list = [means*i for i in range(self.num_classes)]
        
        train_loader = self.load_train_data(poison=False) 
        dataset_size = len(self.train_loader.dataset)
        for _ in range(dataset_size):
            i = random.randint(0, self.num_classes-1)
            u = u_list[i]
            noise = torch.randn((3,32,32), device=self.device) * stds + u
            y_all.append(torch.tensor(i))
            x_all.append(noise.unsqueeze(0))

        x_all = torch.cat(x_all, dim=0)
        y_all = torch.tensor(y_all)
        x_all = (x_all - 0.5) / 0.5


        noise_data=[(x,y) for x,y in zip(x_all,y_all)]
        return DataLoader(noise_data, self.batch_size, drop_last=True, shuffle=True)
  
