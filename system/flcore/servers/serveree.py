

import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clientee import clientEE
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model


class FedEE(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientEE)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_Budget=[] #计时unlearning的时间
        self.history_models=[[] for _ in range(self.num_clients)]




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
            
            self.collect_models()
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
            self.set_new_clients(clientEE)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def collect_models(self):
        
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            self.history_models[cid].append(client_model)
    

    def unlearning(self):
        attack_model=train_attack_model(self.global_model,self.clients,self.num_classes,self.device)
        (PRE_old, REC_old) = attack(self.global_model,attack_model,self.unlearning_clients,self.num_classes,self.device)
        teacher_model =copy.deepcopy(self.global_model)
        self.clients = [client for client in self.clients if client not in self.unlearning_clients]

        for param in self.global_model.parameters():
            param.data.zero_()
        for c in self.clients:
            for param1, param2 in zip(self.global_model.parameters(), c.model.parameters()):
                param1.data += param2/len(self.clients)
        

        for i in range(int(self.global_rounds/2)+1):
            s_t = time.time()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            self.send_models_target()
            self.evaluate()
            for client in self.unlearning_clients:
                client.unlearning_train()
            
            self.receive_models_target()
            
            self.aggregate_parameters()
            
            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])


        print(f"\n============= Post learing start =============")
        for i in range(int(self.global_rounds/2)+1):
            s_t = time.time()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            self.evaluate()
            for client in self.clients:
                client.train()
            
            self.receive_models()
            
            self.aggregate_parameters()

            self.unlearn_Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.unlearn_Budget[-1])
        
        (PRE_unlearning, REC_unlearning) = attack(self.global_model,attack_model,self.unlearning_clients,self.num_classes,self.device)
        
        print("MIA Attacker to old model precision = {:.4f}".format(PRE_old))
        print("MIA Attacker to old model recall = {:.4f}".format(REC_old))

        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        
        self.save_results()
        self.save_global_model()
    


