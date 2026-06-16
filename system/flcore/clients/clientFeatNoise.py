
import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from collections import defaultdict


class clientFeatNoise(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.opt_ul_head = torch.optim.SGD(self.model.head.parameters(), lr=self.unlearning_rate
                                         ,momentum=0.9,weight_decay=0.0005)
        
        self.glabel_class_mean = None
        self.globel_class_var = None
        self.class_anchor = None


    def train(self):
        trainloader=self.train_loader
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        self.class_anchor = [(torch.zeros(1,512).to(self.device),0) for i in range(self.num_classes)]
        for epoch in range(max_local_epochs):
            class_features = defaultdict(list)
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                features = self.model.base(x) 
                for feat, label in zip(features, y):
                    class_features[label.item()].append(feat)
                
                output = self.model.head(features)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        for label, feats in class_features.items():
            feats_tensor = torch.stack(feats)
            z_mean = torch.mean(feats_tensor, dim=0)
            z_var = torch.var(feats_tensor, dim=0)
            self.class_anchor[label] = (z_mean,z_var,len(feats))

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    # 冻结特征提取器  只微调or遗忘分类头
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
                loss = self.loss(output, y)
                self.optimizer_ul.zero_grad()
                loss.backward()
                self.optimizer_ul.step()

    def get_anchor(self):
        trainloader=self.train_loader


        class_anchor = [(torch.zeros(1,512).to(self.device),torch.zeros(1,512).to(self.device),0) for i in range(self.num_classes)]

        class_features = defaultdict(list)
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            features = self.model.base(x).detach().cpu() 
            for feat, label in zip(features, y):
                class_features[label.item()].append(feat)
                
        
        for label, feats in class_features.items():
            feats_tensor = torch.stack(feats)
            z_mean = torch.mean(feats_tensor, dim=0).to(self.device)
            z_var = torch.var(feats_tensor, dim=0).to(self.device) if feats_tensor.shape[0] > 1 else torch.zeros(1,512).to(self.device)
            class_anchor[label] = (z_mean,z_var,len(feats))
        

        return class_anchor