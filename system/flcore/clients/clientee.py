# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientEE(Client):
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
                self.optimizer.step()


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def unlearning_train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        w_ref = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
        theta=torch.norm((w_ref-torch.randn_like(w_ref)),p=2)/2400
        for i in range(9):
            theta+=torch.norm((w_ref-torch.randn_like(w_ref)),p=2)/2400
        print("theta is",theta)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = -self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                w = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                w=self.projection(w,w_ref,theta)
                self.load_flattened_vector_to_model(self.model,w)
        

    def projection(self,w, w_ref, theta):
        delta = w - w_ref
        distance = torch.norm(delta, p=2)
        if distance > theta:
            w = w_ref + delta * (theta / distance)
        return w

    def load_flattened_vector_to_model(self,model, vector) :
        pointer = 0  
        for param in model.parameters():
            num_params = param.numel()
            
            param_data = vector[pointer : pointer + num_params].view_as(param.data)
            param.data.copy_(param_data)

            pointer += num_params
        