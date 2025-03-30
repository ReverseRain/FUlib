

import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
import cvxpy
import quadprog
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clientgem import clientGEM
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model


class FedGEM(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGEM)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_Budget=[] #计时unlearning的时间




    def train(self):
        for c in self.unlearning_clients:
            c.unlearning=True
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

        # self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientGEM)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    

    def unlearning(self):
        attack_model=train_attack_model(self.global_model,self.clients,self.num_classes,self.device)
        (PRE_old, REC_old) = attack(self.global_model,attack_model,self.unlearning_clients,self.num_classes,self.device)
    
        self.clients== [client for client in self.clients if client not in self.unlearning_clients]
        opt_ul=torch.optim.SGD(self.global_model.parameters(), lr=0.01)

        for i in range(self.global_rounds+1):
            s_t = time.time()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            self.send_models_target()
            self.evaluate()

            grads = torch.cat([p.view(-1) for p in self.global_model.parameters()], dim=0)  
            unlearning_grad = torch.zeros_like(grads)
            normal_grad=[]

            # opt_ul.zero_grad()
            for client in self.unlearning_clients:
                unlearning_grad+=client.unlearning_train()/len(self.unlearning_clients)
            
            
            for client in self.clients:
                normal_grad.append(client.unlearning_train())
            # normal_grad = torch.stack(normal_grad, dim=0)
            
            unlearning_grad=self.PROJECT(unlearning_grad,normal_grad)
            
            self.overwrite_grad(self.global_model.parameters,unlearning_grad)
            # opt_ul.step()

            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])
        (PRE_unlearning, REC_unlearning) = attack(self.global_model,attack_model,self.unlearning_clients,self.num_classes,self.device)
        
        print("MIA Attacker to old model precision = {:.4f}".format(PRE_old))
        print("MIA Attacker to old model recall = {:.4f}".format(REC_old))

        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        
        self.save_results()
        self.save_global_model()
    
    def project2cone2(self,gradient, memories, margin=0.5, eps=1e-3):
        print(memories.shape)
        print(gradient.shape)
        memories_np = memories.cpu().t().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1))


    def overwrite_grad(self,pp, newgrad):
        pointer=0
        for param in pp():
            num_params = param.numel()
            
            param_data = newgrad[pointer : pointer + num_params].view_as(param.data)
            param.data=param.data-1e-5*(param_data)

            pointer += num_params

    def PROJECT(self,g, old_gradients,margin=0.5):

        g = g.to(dtype=torch.float32)
        old_gradients = [grad.to(dtype=torch.float32) for grad in old_gradients]
        
        # ------ 初始化变量 ------
        device = g.device
        G = torch.stack(old_gradients, dim=0)  # [num_old_tasks, num_params]
        v = torch.full((G.size(0),), margin, 
                    device=device, dtype=torch.float32,  # 显式指定为Float32
                    requires_grad=True)
    
        # ------ 迭代优化 ------
        optimizer = torch.optim.Adam([v], lr=0.01)
        
        for _ in range(100):
            # 计算目标函数: 0.5 * v^T (G G^T) v + g^T G^T v
            GGT = torch.mm(G, G.T)                # [num_old_tasks, num_old_tasks]
            Gg = torch.mv(G, g)                   # [num_old_tasks]
            loss = 0.5 * torch.dot(v, torch.mv(GGT, v)) + torch.dot(v, Gg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                v.data = torch.clamp(v, min=margin)
        
        g_tilde = g + torch.mv(G.T, v)  # [num_params]
        return g_tilde