

import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clientFeatNoise import clientFeatNoise
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model
import os
from collections import defaultdict
import torch.nn as nn


class FedFeatNoise(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFeatNoise)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_Budget=[] #计时unlearning的时间

        self.class_anchor = []
        self.uploaded_features = None
        
        self.opt_ul_head = None


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
        self.save_uploaded_features()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFeatNoise)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    

    def unlearning(self):
        self.load_model()
    
        self.clients = [client for client in self.clients if client not in self.unlearning_clients]
        

        self.send_models()
        self.send_models_target()
        
        unlearning_features = defaultdict(list)
        self.uploaded_features = defaultdict(list)
        for c in self.clients:
            self.uploaded_features[c.id] = c.get_anchor()
        for c in self.unlearning_clients:
            unlearning_features[c.id] = c.get_anchor()


        remain_class_mean, remain_class_var, remain_weight = self.aggregate_anchor(self.uploaded_features)
        unlearning_class_mean, unlearning_class_var, unlearning_weight = self.aggregate_anchor(unlearning_features)

        # 这里考虑使用可分的随机纯噪声
        
        mean_scale = torch.norm(remain_class_mean[0]).item() 
        mean = torch.rand(1,512).to(self.device)
        # mean = mean / torch.linalg.norm(mean) * mean_scale
        unlearning_class_mean = [mean * (self.num_classes-i) * 0.2  for i in range(self.num_classes)]
        
        var_scale = torch.norm(remain_class_var[0]).item()
        var = torch.rand(1,512).to(self.device)
        unlearning_class_var = [var * 0.05 for i in range(self.num_classes)]


        print("mean_scale is ", torch.norm(remain_class_mean[0]).item(), "var_scale is ", torch.norm(remain_class_var[0]).item(), "mean sacle -1",torch.norm(remain_class_mean[-1]).item() )
        print("unlearning mean ",torch.norm(unlearning_class_mean[0]).item())
        print("unlearning mean -1 ",torch.norm(unlearning_class_mean[-1]).item())
        print("unlearning var ",torch.norm(unlearning_class_var[0]).item())
        

        weight = int(sum(unlearning_weight) / (self.num_classes))
        unlearning_weight = [weight for i in range(self.num_classes)]

        noise_data, noise_labels = self.sample_gaussian_mixture(
            remain_class_mean, 
            remain_class_var, 
            remain_weight
        )
        
        unlearning_noise_data, unlearning_noise_labels = self.sample_gaussian_mixture(
            # [-1*u for u in unlearning_class_mean], 
            unlearning_class_mean,
            unlearning_class_var, 
            unlearning_weight,
            # [int(w*10) for w in unlearning_weight],
        )

        noise_data = torch.cat([noise_data,unlearning_noise_data], dim=0) 
        noise_labels = torch.cat([noise_labels,unlearning_noise_labels])

        noise_dataset = [(x,y) for x,y in zip(noise_data,noise_labels)]
        noise_loader = DataLoader(noise_dataset, 32, drop_last=False, shuffle=True)


        self.opt_ul_head = torch.optim.SGD(self.global_model.head.parameters(), lr=self.args.unlearning_rate
                                         ,momentum=0.9,weight_decay=0.0005)

        
        for i in range(self.unlearning_ground+1):
            s_t = time.time()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            self.send_models_target()
            
            self.evaluate()
           
            for param in self.global_model.base.parameters():
                param.requires_grad = False
            for param in self.global_model.head.parameters():
                param.requires_grad = True
            criterion = nn.CrossEntropyLoss()
            losses = 0
            # 这次统一在server端进行Unlearning
            for _, (x, y) in enumerate(noise_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.global_model.head(x) 

                loss = criterion(output, y)

                self.opt_ul_head.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=5.0)
                loss.backward()
                losses += loss.item()
                self.opt_ul_head.step()
            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])
        (PRE_unlearning, REC_unlearning) = attack(self.global_model,self.attacker,self.unlearning_clients,self.num_classes,self.device)
        
        

        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        self.save_unlearning(PRE_unlearning)
        
    def aggregate_anchor(self,uploaded_features):


        total_weight = [0 for i in range(self.num_classes)]


        for _,anchor_list in uploaded_features.items():
            for label,(mean,var,w) in enumerate(anchor_list):
                total_weight[label] += w

        class_mean = [torch.zeros(1,512).to(self.device) for i in range(self.num_classes)]
        class_var = [torch.zeros(1,512).to(self.device) for i in range(self.num_classes)]
        for _,anchor_list in uploaded_features.items():
            for label,(mean,var,w) in enumerate(anchor_list):
                if(total_weight[label]!=0):
                    class_mean[label] += (w/total_weight[label])*mean

        var_part1 = [torch.zeros(1, 512).to(self.device) for _ in range(self.num_classes)]
        var_part2 = [torch.zeros(1, 512).to(self.device) for _ in range(self.num_classes)]
        var_part3 = [torch.zeros(1, 512).to(self.device) for _ in range(self.num_classes)]

        for _, anchor_list in uploaded_features.items():
            for label, (mean, var, w) in enumerate(anchor_list):
                
                Nc = total_weight[label] 
                
                if Nc > 1: 
                    # 第一部分：Σ_k [(N_{c,k}-1)/(N_c-1) * Σ_{c,k}]
                    coeff1 = (w - 1) / (Nc - 1) if w > 1 else 0
                    var_part1[label] += coeff1 * var
                    
                    # 第二部分：Σ_k [N_{c,k}/(N_c-1) * μ_{c,k} * μ_{c,k}^T]
                    coeff2 = w / (Nc - 1)
                    var_part2[label] += coeff2 * (mean * mean)  

        # 计算第三部分
        for label in range(self.num_classes):
            Nc = total_weight[label]
            if Nc > 1:
                coeff3 = Nc / (Nc - 1)
                var_part3[label] = coeff3 * (class_mean[label] * class_mean[label])

                class_var[label] = var_part1[label] + var_part2[label] - var_part3[label]
                # if(torch.isnan(class_var[label]).any().item()):
                #     print(var_part1[label][0:10],var_part2[label][0:10],var_part3[label][0:10],label)

        return class_mean, class_var, total_weight
    
    def save_uploaded_features(self):
        model_path = os.path.join("models_seed"+str(self.args.seed_num)+"_resnet", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_path = os.path.join(model_path, "uploaded_features"+".pt")

        from collections import defaultdict
        self.uploaded_features = defaultdict(list)
        for c in self.clients:
            self.uploaded_features[c.id]=c.class_anchor
        torch.save(self.uploaded_features,model_path)

    def sample_gaussian_mixture(self, means, vars, weights, std_weight=1):
        samples = []
        labels = []
        for i, (mu, var, n) in enumerate(zip(means, vars, weights)):
            # print(f'mu nan: {torch.isnan(mu).any().item()}, var nan: {torch.isnan(var).any().item()}')
            mu = mu.view(-1)
            std = torch.sqrt(var.view(-1))

            noise = torch.randn(n, len(mu), device=mu.device) * std * std_weight  + mu  
            
            samples.append(noise)
            labels.extend([i] * n)
        samples = torch.cat(samples, dim=0)  
        labels = torch.tensor(labels, dtype=torch.long)
        
        
        return samples, labels
    
