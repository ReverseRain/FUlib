

import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clientNoise import clientNoise
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model
import torch.nn as nn
import random
from utils.data_utils import read_client_data

class FedNoise(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientNoise)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_Budget=[] #计时unlearning的时间




    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.attacker=train_attack_model(self.global_model,self.clients,self.num_classes,self.device)
        (PRE_old, REC_old) = attack(self.global_model,self.attacker,self.unlearning_clients,self.num_classes,self.device)
        print("MIA Attacker to old model precision = {:.4f}".format(PRE_old))
        print("MIA Attacker to old model recall = {:.4f}".format(REC_old))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientNoise)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def unlearning(self):
        self.load_model()
        
        self.clients = [client for client in self.clients if client not in self.unlearning_clients]

        self.opt_ul = torch.optim.SGD(self.global_model.parameters(), lr=self.args.unlearning_rate
                                         ,momentum=0.9,weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()

        if(self.args.noise_type == "class_noise"):
            noise_loader = self.get_class_noise()
        elif(self.args.noise_type == "pure_noise"):
            noise_loader = self.get_pure_noise()
        elif(self.args.noise_type == "svhn"):
            noise_loader = self.get_svhn()
        elif(self.args.noise_type == "shuffle"):
            noise_loader = self.get_shuffle()

        model = copy.deepcopy(self.global_model)
        mean = torch.tensor([0.5, 0.3, 0.7]).view(3, 1, 1).expand(3, 32, 32)
        mean_list = [(mean*i-0.5)/0.5 for i in range(self.num_classes)]
        means = torch.stack(mean_list).to(self.device)
        self.global_model.train()

        # lamda = 0.005
        lamda = 0.005 
        for i in range(self.unlearning_ground+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            for c in self.unlearning_clients:
                c.model = copy.deepcopy(self.global_model)
            if(i==0):
                self.warm_up()
                # self.draw_tsne("tsne_before_unlearning.png")
            self.evaluate()
            
            for _, (x, y) in enumerate(noise_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.global_model(x) 

                batch_means = means[y]
                distance_sq = lamda * torch.sum((x - batch_means) ** 2, dim=(1, 2, 3))
                # distance_sq =  0.1 * torch.sqrt(torch.sum((x - batch_means) ** 2, dim=(1, 2, 3))+1e-8)

                loss = (criterion(output, y) * (1.0 / (distance_sq + 1e-4))).mean()

                # loss = criterion(output, y)

                # gm = torch.cat([p.data.view(-1) for p in self.global_model.base.parameters()], dim=0)
                # pm = torch.cat([p.data.view(-1) for p in model.base.parameters()], dim=0)
                # loss += torch.norm(gm-pm, p=2) * 0.1

                self.opt_ul.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=5.0)
                self.opt_ul.step()
            
            for client in self.selected_clients:
                client.train()
            self.aggregate_parameters_server(len(noise_loader.dataset))

            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])

        (PRE_unlearning, REC_unlearning) = attack(self.global_model,self.attacker,self.unlearning_clients,self.num_classes,self.device)


        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        self.save_unlearning(PRE_unlearning)
        # self.draw_tsne("tsne_after_unlearning_NoiseFU.png")
        self.cka_analyse()

    
    def get_class_noise(self):
        x_all=[]
        y_all=[]
        train_loader = self.unlearning_clients[0].train_loader
        with torch.no_grad():
            for i, (x, y) in enumerate(train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                y_all.append(y.cpu())
                x_all.append(x.cpu())

                batch_size = x.size(0)

        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)

        unique_labels = torch.unique(y_all)
        noise_x_list = []
        noise_y_list = []

        for label in unique_labels:
            # 提取当前类别的数据
            idx = (y_all == label)
            class_data = x_all[idx]
            mean = class_data.mean(dim=0, keepdim=True)
            std = class_data.std(dim=0, keepdim=True)
            noise_samples = torch.normal(mean.expand_as(class_data), std.expand_as(class_data)) + 1e-6

            noise_x_list.append(noise_samples)
            noise_y_list.append(torch.full((class_data.size(0),), label.item()))

        noise_x_list = torch.cat(noise_x_list, dim=0)
        noise_y_list = torch.cat(noise_y_list, dim=0)



        noise_data=[(x,y) for x,y in zip(noise_x_list,noise_y_list)]
        return DataLoader(noise_data, self.batch_size, drop_last=True, shuffle=True)
    
    def get_pure_noise(self):
        x_all=[]
        y_all=[]

        means = torch.tensor([0.5, 0.3, 0.7]).view(3, 1, 1)
        # stds = torch.tensor([0.3, 0.2, 0.65]).view(3, 1, 1)
        stds = torch.tensor([0.25, 0.25, 0.45]).view(3, 1, 1)

        u_list = [i * means for i in range(self.num_classes)]
        # u_list = [means for i in range(self.num_classes)]
        
        
        dataset_size = len(self.unlearning_clients[0].train_loader.dataset)
        for _ in range(dataset_size):
            i = random.randint(1, self.num_classes - 1)
            u = u_list[i]
            noise = torch.randn((3,32,32)) * stds + u
            y_all.append(torch.tensor(i))
            x_all.append(noise.unsqueeze(0))

        x_all = torch.cat(x_all, dim=0)
        y_all = torch.tensor(y_all)
        x_all = (x_all - 0.5) / 0.5


        noise_data=[(x,y) for x,y in zip(x_all,y_all)]
        
        return DataLoader(noise_data, self.batch_size, drop_last=True, shuffle=True)
    
    def get_svhn(self):

        train_data = read_client_data('SVHN', 1, is_train=True)
        return DataLoader(train_data, self.batch_size, drop_last=True, shuffle=True)
    
    def get_shuffle(self):
        dataset = self.unlearning_clients[0].train_loader.dataset
        shuffled_data=[(x,(y+torch.randint(1, self.num_classes, y.shape, device=y.device))%self.num_classes) for x,y in dataset]
        return DataLoader(shuffled_data, self.batch_size, drop_last=True, shuffle=True)
  

