

import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clientosd import clientOSD
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model


class FedOSD(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientOSD)

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
            self.set_new_clients(clientOSD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    

    def unlearning(self):
        self.load_model()     
    
        self.clients = [client for client in self.clients if client not in self.unlearning_clients]
        
        self.opt_ul = torch.optim.SGD(self.global_model.parameters(), lr=self.args.unlearning_rate
                                         ,momentum=0.9,weight_decay=0.0005)

        for i in range(self.unlearning_ground+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.send_models()
            for client in self.unlearning_clients:
                client.model = copy.deepcopy(self.global_model)
            if(i==0):
                self.warm_up()
            self.evaluate()
            model = copy.deepcopy(self.global_model)
            gm = torch.cat([p.view(-1) for p in self.global_model.parameters()], dim=0)  

            for client in self.unlearning_clients:
                client.unlearning_train()
            # self.unlearning_noise(model)
            for client in self.selected_clients:
                client.train()
            
            
            # 记录遗忘梯度的方向
            
            self.unlearning_grad = torch.zeros_like(gm)
            self.receive_models_target()
            for weights in self.uploaded_models:
                pm=torch.cat([p.view(-1) for p in weights.parameters()], dim=0) 
                self.unlearning_grad+=(pm-gm)/len(self.uploaded_models)

            # 记录正常用户更新的方向
            self.normal_grad=[]
            self.receive_models()
            for weights in self.uploaded_models:
                pm=torch.cat([p.view(-1) for p in weights.parameters()], dim=0) 
                self.normal_grad.append(pm-gm)

            self.normal_grad = [grad.to(dtype=torch.float32) for grad in self.normal_grad]
            G = torch.stack(self.normal_grad, dim=0)

            unlearning_grad=self.get_nearest_oth_d(G,self.unlearning_grad)
            self.overwrite_grad(self.global_model.parameters,unlearning_grad)
            
            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])


        (PRE_unlearning, REC_unlearning) = attack(self.global_model,self.attacker,self.unlearning_clients,self.num_classes,self.device)
        

        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        self.save_unlearning(PRE_unlearning)
        
    
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
    
    def overwrite_grad(self,pp, newgrad):
        pointer=0
        for param in pp():
            num_params = param.numel()
            
            param_data = newgrad[pointer : pointer + num_params].view_as(param.data)
            param.data=param.data+(param_data)

            pointer += num_params

