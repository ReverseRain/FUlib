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

import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clientfukd import clientFUKD
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model


class FedFUKD(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFUKD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_Budget=[] #计时unlearning的时间
        self.history_update=[[] for _ in range(self.num_clients)]




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
            
            self.collect_delta()
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
            self.set_new_clients(clientFUKD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def collect_delta(self):
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)
            self.history_update[cid].append(origin_grad)
    

    def unlearning(self):
        attack_model=train_attack_model(self.global_model,self.clients,self.num_classes,self.device)
        unlearning_clients = [self.clients[i] for i in self.args.unlearning_client]
        (PRE_old, REC_old) = attack(self.global_model,attack_model,unlearning_clients,self.num_classes,self.device)
        teacher_model =copy.deepcopy(self.global_model)
        for i in self.args.unlearning_client:
            for j in range(len(self.history_update[i])):
                for param1, diff in zip(self.global_model.parameters(), self.history_update[i][j]):
                    # param1.data -= torch.tensor([x / len(self.clients) for x in diff])
                    param1.data -= diff/len(self.clients)
                    
        
        
        self.clients== [client for client in self.clients if client not in unlearning_clients]
        opt_ul=torch.optim.SGD(self.global_model.parameters(), lr=0.01)
        
        proxy_loader=self.proxy_load()

        for i in range(self.global_rounds+1):
            s_t = time.time()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            self.evaluate(isUnlearning=True)
            self.global_model.train()

            # print(self.global_model.head.weight[:2])
            # print(teacher_model.head.weight[:2])

            for i, x in enumerate(proxy_loader):

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                
                output_student = self.global_model(x)
                output_teacher = teacher_model(x)
                q_i=F.softmax(output_teacher/3,dim=-1)
                q_c=F.softmax(output_student/3,dim=-1)

                loss=F.kl_div(q_i.log(), q_c, reduction='batchmean')


                opt_ul.zero_grad()
                loss.backward()
                opt_ul.step()

            # print(self.global_model.head.weight[:2])
            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])
        (PRE_unlearning, REC_unlearning) = attack(self.global_model,attack_model,unlearning_clients,self.num_classes,self.device)
        
        print("MIA Attacker to old model precision = {:.4f}".format(PRE_old))
        print("MIA Attacker to old model recall = {:.4f}".format(REC_old))

        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        
        self.save_results()
        self.save_global_model()

    def proxy_load(self):
        data = read_proxy_data(self.dataset)
        return DataLoader(data, 32, shuffle=True)
    
    
