

import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clientgs import clientGS
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model
import math

import matplotlib.pyplot as plt
import tracemalloc

class FedGS(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGS)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_Budget=[] #计时unlearning的时间

        self.unlearning_grad = None
        self.normal_grad = None




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
            self.set_new_clients(clientGS)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    

    def unlearning(self):
        self.load_model()

        
        self.clients = [client for client in self.clients if client not in self.unlearning_clients]
        tot_samples=0
        for client in self.clients:
            tot_samples += client.train_samples
        self.send_models_target()
        for client in self.unlearning_clients:
            if self.args.positive_sample=='aug':
                client.getPairLoader2()
            elif self.args.negative_sample!='label':
                client.getPairLoader()
        for i in range(self.unlearning_ground+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            self.send_models_target()
            if(i==0):
                self.warm_up()
            self.evaluate()
            
            for client in self.unlearning_clients:
                client.unlearning_train()
            for client in self.selected_clients:
                client.train()

            gm = torch.cat([p.view(-1) for p in self.global_model.parameters()], dim=0).detach()

            self.unlearning_grad = torch.zeros_like(gm)
            self.receive_models_target()
            for weights in self.uploaded_models:
                pm=torch.cat([p.view(-1) for p in weights.parameters()], dim=0).detach()
                self.unlearning_grad+=(pm-gm)/len(self.uploaded_models)

            self.normal_grad = []
            self.receive_models()
            for w,weights in zip(self.uploaded_weights,self.uploaded_models):
                pm=torch.cat([p.view(-1) for p in weights.parameters()], dim=0).detach() 
                self.normal_grad.append((pm-gm)*w)

            start = time.time()
            if self.args.gradient_hadle == "GEM":
                unlearning_grad=self.PROJECT(self.unlearning_grad,self.normal_grad)

            elif self.args.gradient_hadle == "OSD":
                self.normal_grad = [grad.to(dtype=torch.float32) for grad in self.normal_grad]
                G = torch.stack(self.normal_grad, dim=0)
                unlearning_grad=self.get_nearest_oth_d(G,self.unlearning_grad)
            else:
                unlearning_grad = self.unlearning_grad
                
            print("gradient handle time:   ",time.time()-start)
            print("unlearning grad norm:   ",torch.norm(unlearning_grad))
            self.overwrite_grad(self.global_model.parameters,unlearning_grad)
            
            # self.cka_analyse()

            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])
        (PRE_unlearning, REC_unlearning) = attack(self.global_model,self.attacker,self.unlearning_clients+self.clients,self.num_classes,self.device)


        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        # self.cka_analyse()
        # self.dist_Analyse()

        self.save_unlearning(PRE_unlearning)
    

        

    def overwrite_grad(self,pp, newgrad):
        pointer=0
        for param in pp():
            num_params = param.numel()
            
            param_data = newgrad[pointer : pointer + num_params].view_as(param.data)
            param.data=param.data+(param_data)

            pointer += num_params

    def PROJECT(self,g, old_gradients,margin=0.5):   

        g = g.to(dtype=torch.float32)
        old_gradients = [grad.to(dtype=torch.float32) for grad in old_gradients]

        device = g.device
        G = torch.stack(old_gradients, dim=0)
        
        v = torch.full((G.size(0),), margin+random.uniform(0, 1), 
                    device=device, dtype=torch.float32,
                    requires_grad=True)

        optimizer = torch.optim.Adam([v], lr=0.01)
        
        GGT = torch.mm(G, G.T)
        Gg = torch.mv(G, g)
        for i in range(30):
            # target function: 0.5 * v^T (G G^T) v + g^T G^T v
            loss = 0.5 * torch.dot(v, torch.mv(GGT, v)) + torch.dot(v, Gg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                v.data = torch.clamp(v, min=margin)
            # t=t*0.99
        
        g_tilde = g + torch.mv(G.T, v)  # [num_params]
        
        B_t = torch.norm(g - g_tilde) / (torch.norm(g_tilde) + 1e-8)
        print("projection burden: {:.6f}".format(B_t.item()))
        angles = [round((torch.dot(grad,g)/(torch.norm(grad)*torch.norm(g))).item(),3) for grad in old_gradients]

        violation_magnitude = np.mean([max(0, -cos_val) for cos_val in angles])
        print("Violation Magnitude: {:.4f}".format(violation_magnitude))
        angles_deg = [round(math.degrees(math.acos(cos)), 2) for cos in angles]

        angles_less_90 = [a for a in angles_deg if a < 90]

        ratio = len(angles_less_90) / len(angles_deg)
        print("Ratio of angles less than 90 degrees: {:.2f}".format(ratio))
        print("before L2 norm ",torch.norm(g))
        
        # mean_ang=np.mean(angles)
        # print('after ',mean_ang)
        # min_angle= np.min(angles)
        # print('after min',min_angle)
        # mean_deg = np.mean(angles_deg)
        # print('after mean',mean_deg)
        # max_deg = np.max(angles_deg)
        # print('after max',max_deg)

        return g_tilde
    
    
    def get_nearest_oth_d(self, gr_locals, gu):
        A = gr_locals
        
        A_T = A.T
        c = gu
        
        AAT_1 = self.cal_psedoinverse(A @ A_T)  
        
        Ac= A @ c.reshape(-1, 1)
        
        AAT_1_Ac = AAT_1 @ Ac
        
        d = c - (A_T @ AAT_1_Ac).reshape(-1)

        return d
    
    def cal_psedoinverse(self, matrix):
        U, s, V = torch.svd(matrix)  
        primary_sigma_indices = torch.where(s >= 1e-6)[0]
        s[primary_sigma_indices] = 1 / s[primary_sigma_indices]
        S = torch.diag(s)
        psedoinverse = V @ S @ U.T
        return psedoinverse
    