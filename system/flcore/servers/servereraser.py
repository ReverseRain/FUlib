
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import read_proxy_data
from flcore.clients.clienteraser import clientEraser
from flcore.servers.serverbase import Server
from threading import Thread
from utils.attack_utils import attack,train_attack_model


class FedEraser(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientEraser)

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

        self.attacker=train_attack_model(self.global_model,self.clients,self.num_classes,self.device)

        (PRE_old, REC_old) = attack(self.global_model,self.attacker,self.unlearning_clients,self.num_classes,self.device)
        print("MIA Attacker to old model precision = {:.4f}".format(PRE_old))
        print("MIA Attacker to old model recall = {:.4f}".format(REC_old))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientEraser)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def collect_delta(self):
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            if cid == self.unlearning_clients[0].id:
                continue
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(pp.data - gp.data)
            self.history_update[cid].append(origin_grad)
    

    def unlearning(self):
        self.load_model()

        self.clients = [client for client in self.clients if client not in self.unlearning_clients]
        tot_samples=0
        for c in self.clients:
            tot_samples+=c.train_samples
            self.uploaded_weights.append(c.train_samples)
        self.uploaded_weights = [w/tot_samples for w in self.uploaded_weights]

        self.global_model = copy.deepcopy(self.args.model)

        print(len(self.history_update),len(self.history_update[0]))
        for w,c in zip(self.uploaded_weights,self.clients):
        
            for param1, diff in zip(self.global_model.parameters(), self.history_update[c.id][0]):
                param1.data += (diff*w)

        
        self.selected_clients = self.select_clients()


        for epoch in range(self.global_rounds):
            # global_model = unlearn_global_models[epoch]
            # self.global_model
            if(epoch == 0):
                continue

            # new_client_models  = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
            self.send_models()
            # self.evaluate()
            gm=torch.cat([p.view(-1) for p in self.global_model.parameters()], dim=0).detach()
            for client in self.selected_clients:
                client.train()

            self.receive_models()

            client_update=[]

            for uploaded_model,c in zip(self.uploaded_models,self.clients):
                cm=torch.cat([p.view(-1) for p in self.history_update[c.id][epoch]], dim=0).detach()
                uploaded_w=torch.cat([p.view(-1) for p in uploaded_model.parameters()], dim=0).detach()
                client_update.append(torch.norm(cm)*(uploaded_w-gm)/torch.norm(uploaded_w-gm))

            final_update=torch.zeros_like(client_update[0])
            for w,client_update in zip(self.uploaded_weights,client_update):
                final_update +=(w*client_update)
            self.overwrite_grad(self.global_model.parameters,final_update)
            

        self.send_models()
        self.send_models_target()
        self.evaluate()
        (PRE_unlearning, REC_unlearning) = attack(self.global_model,self.attacker,self.clients,self.num_classes,self.device)


        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))

        self.save_unlearning(PRE_unlearning)
    
    # def unlearning_step_once(self,old_client_models, new_client_models, global_model_before_forget):
    #     gm=torch.cat([p.view(-1) for p in self.global_model.parameters()], dim=0).detach()
    #     gm_bf=torch.cat([p.view(-1) for p in global_model_before_forget.parameters()], dim=0).detach()
        
    #     old_param_update = torch.zeros_like(gm)
    #     new_param_update = torch.zeros_like(gm)

    #     assert len(old_client_models) == len(new_client_models)

    #     for ii in range(len(new_client_models)):
    #         ocm=torch.cat([p.view(-1) for p in old_client_models[ii].parameters()], dim=0).detach()
    #         ncm=torch.cat([p.view(-1) for p in new_client_models[ii].parameters()], dim=0).detach()
            
    #         old_param_update += (ocm/ len(old_client_models))
    #         new_param_update += (ncm/ len(new_client_models))
    #         # print('ncm ',ncm)
    #     # old_param_update /= (ii+1)#Model Params： oldCM
    #     # new_param_update /= (ii+1)#Model Params： newCM
    #     # print(' new 1 ',new_param_update)
        
    #     old_param_update = old_param_update - gm_bf#参数： oldCM - oldGM_t
    #     new_param_update = new_param_update - gm
        
    #     step_length = torch.norm(old_param_update)
    #     step_direction = new_param_update/(torch.norm(new_param_update)+1e-3)#(newCM - newGM_t)/||newCM - newGM_t||
        
    #     return_model_state = gm + step_length*step_direction
        
    #     print('gm ',gm )
    #     print('step_length',step_length)
    #     print('step_direction ',step_direction)
    #     print('return_model_state ',return_model_state)
    #     print('new_param_update ',new_param_update)
    #     print('norm ',torch.norm(new_param_update))
    #     self.overwrite_grad(self.global_model.parameters,return_model_state)
