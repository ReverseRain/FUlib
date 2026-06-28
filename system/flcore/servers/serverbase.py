

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import json
from utils.data_utils import backdoor_pattern, read_client_data
from utils.dlg import DLG
from utils.attack_utils import attack,train_attack_model
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE


class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new
        
        self.unlearning_clients=args.unlearning_clients #此处传入的仍是id list 在set clients当中变成 clients list
        self.unlearning_ground=args.unlearning_ground
        self.post_training_ground=args.post_training_ground

        self.attack_acc=[]
        self.history_update=[[] for _ in range(self.num_clients)]
        self.old_clients=[]
        self.old_global_model=[]
        self.attacker = xgb.XGBClassifier()

        self.attack_df = pd.DataFrame(columns=['attack_accuracy'])
        self.test_df = pd.DataFrame(columns=['test_accuracy'])

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            unlearning= (i in self.unlearning_clients) if self.unlearning_clients else False)
            self.clients.append(client)
        self.unlearning_clients=[self.clients[i] for i in self.unlearning_clients] if self.unlearning_clients else []
        

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        # else:
        #     self.current_num_join_clients = len(self.clients)
        self.current_num_join_clients = len(self.clients) if self.join_ratio==1 else self.current_num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models_seed"+str(self.args.seed_num)+"_resnet", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if(self.args.learning_state!="retrain"):
            if(self.history_update[-1]):
                history_path=os.path.join(model_path,"history")
                if not os.path.exists(history_path):
                    os.makedirs(history_path)
                history_path=os.path.join(history_path,((''.join(map(str, self.args.unlearning_clients))
                                                        +"_attack_client") if self.args.attack=='True' else "_client") + ".pt")
                torch.save(self.history_update,history_path)
            
            if(self.attacker):
                attacker_path=os.path.join(model_path,("Backdoor_" if self.args.attack=='True' else "noBackdoor_") + "xgb_model.bin")
                self.attacker.save_model(attacker_path)
            

            model_path = os.path.join(model_path, ((''.join(map(str, self.args.unlearning_clients)) + 
                                      "_attack_server" ) if self.args.attack=='True' else "_server") + ".pt")
            
        else:
            model_path = os.path.join(model_path, "retrain_model_"+('_'.join(map(str, self.args.unlearning_clients)) 
                                                                   if self.args.attack == 'True' else '') + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models_seed"+str(self.args.seed_num)+"_resnet", self.dataset)
        print('model path',model_path)
        if(self.algorithm=="FedFUKD" or self.algorithm=="FedEraser"):
            history_path=os.path.join(model_path,"history")
            history_path=os.path.join(history_path,((''.join(map(str, self.args.unlearning_clients))
                                                        +"_attack_client") if self.args.attack=='True' else "_client") + ".pt")
            assert (os.path.exists(history_path))
            self.history_update=torch.load(history_path,map_location='cpu',weights_only=False,mmap=True)[:45]
            # self.history_update = [tensor.to(self.device) for tensor in self.history_update]

        attacker_path=os.path.join(model_path,("Backdoor_" if self.args.attack=='True' else "noBackdoor_") + "xgb_model.bin")
        self.attacker.load_model(attacker_path)

        model_path = os.path.join(model_path, ((''.join(map(str, self.args.unlearning_clients)) + 
                                      "_attack_server" ) if self.args.attack=='True' else "_server") + ".pt")
        print('model path',model_path)
        assert (os.path.exists(model_path))
        
        self.global_model=torch.load(model_path,map_location=self.device,weights_only=False)

        

    def model_exists(self):
        model_path = os.path.join("models_"+self.args.seed, self.dataset)
        model_path = os.path.join(model_path, (''.join(map(str, self.args.unlearning_clients)) + 
                                  "_attack_server") if self.args.attack=='True' else "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal 
            # + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []

        att_correct=[]
        att_num_samples=[]
        overlap_acc=[]
        unique_acc=[]

        overlap_num=[]
        unique_num=[]

        # tag_correct=[]
        # tag_num_samples=[]

        for c in self.clients:
            if(c.unlearning==False):
                ct, ns, auc, oc, uc, on, un = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
                overlap_acc.append(oc*1.0)
                overlap_num.append(on)
                unique_acc.append(uc*1.0)
                unique_num.append(un)
            
        for c in self.unlearning_clients:
            ct, ns, auc, oc, uc, on, un = c.test_metrics()
            att_correct.append(ct*1.0)
            att_num_samples.append(ns)
            # overlap_acc.append(oc*1.0)
            # overlap_num.append(on)
            unique_acc.append(uc*1.0)
            unique_num.append(un)

        

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc,att_num_samples,att_correct\
            , overlap_acc, overlap_num, unique_acc, unique_num

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        forget_num_sample = []
        retain_num_sample = []
        retain_acc = []
        forget_acc = []
        losses = []
        
        for c in self.clients:
            if(not c.unlearning):
                cl, ns, ct= c.train_metrics()
                num_samples.append(ns)
                losses.append(cl*1.0)
                retain_acc.append(ct*1.0)
                retain_num_sample.append(ns)
        
        for c in self.unlearning_clients:
            cl, ns, ct= c.train_metrics()

            forget_acc.append(ct*1.0)
            forget_num_sample.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses, forget_acc, forget_num_sample\
        ,retain_acc,retain_num_sample


    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        overlap_acc = sum(stats[6])*1.0 / sum(stats[7]) if sum(stats[7])!=0 else 0
        unique_acc = sum(stats[8])*1.0 / sum(stats[9]) if sum(stats[9])!=0 else 0

        forget_acc = sum(stats_train[3])*1.0 / sum(stats_train[4]) if sum(stats_train[4])!=0 else 0
        retain_acc = sum(stats_train[5])*1.0 / sum(stats_train[6]) if sum(stats_train[6])!=0 else 0
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        attack_acc = sum(stats[5])*1.0 / sum(stats[4]) if self.unlearning_clients else 0
        # target_acc = sum(stats[7])*1.0 / sum(stats[6])
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        self.attack_acc.append(attack_acc)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # print("Averaged Overlapping Accurancy: {:.4f}".format(overlap_acc))
        # print("Averaged Unique Accurancy: {:.4f}".format(unique_acc))

        print("Averaged Forget Accurancy: {:.4f}".format(forget_acc))
        # print("Averaged Retain Accurancy: {:.4f}".format(retain_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("Average Attack Accurancy:{:.4f}".format(attack_acc))
        
        new_attack_row = pd.DataFrame({'attack_accuracy': [attack_acc]})
        new_test_row = pd.DataFrame({'test_accuracy': [test_acc]})

        self.attack_df = pd.concat([self.attack_df, new_attack_row], ignore_index=True)
        self.test_df = pd.concat([self.test_df, new_test_row], ignore_index=True)

    def save_curve(self):
        model_path = os.path.join("models_seed"+str(self.args.seed_num)+"_resnet", self.dataset)
        model_path = os.path.join(model_path, "csv")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        self.attack_df.to_csv(model_path+'/'+str(self.args.attack)+"_"+str(self.algorithm)+'_attack_accuracy.csv', index=False)
        self.test_df.to_csv(model_path+'/'+str(self.args.attack)+"_"+str(self.algorithm)+'_test_accuracy.csv', index=False)

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    
    def send_models_target(self):
        assert (len(self.unlearning_clients) > 0)
        # 向target client send 模型
        for client in self.unlearning_clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)
    
    def receive_models_target(self,is_all=False):
        assert (len(self.unlearning_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        if is_all:
            active_clients = random.sample(
                self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
            target_clients = active_clients + self.unlearning_clients
        else:
            target_clients = self.unlearning_clients
        
        for client in target_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    def save_unlearning(self,mia_pre):
        entry = {
            "name":self.algorithm,
            "test accuracy": self.rs_test_acc,
            "attack accuracy":self.attack_acc,
            "time":self.unlearn_Budget,
            "MIA attack precision":mia_pre
        }
        result_path = "../results/json_file/"

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        algo = self.dataset + "_" + self.algorithm+"_"+str(self.args.attack)

        file_path=result_path+"{}.json".format(algo)
        with open(file_path, 'w') as file:
            json.dump(entry, file, indent=2)
        # model_path=model_path = os.path.join("unlearning", self.dataset)
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        
        # model_path = os.path.join(model_path, "FUGAS.pt")
        # torch.save(self.unlearning_clients[0].model, model_path)
        

    def save_loss(self,global_rounds):
        save_dir = "loss_img"  # 指定保存目录路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_loss = self.rs_train_loss
        # 创建图表和轴对象
        plt.figure(figsize=(12, 6))

        # 绘制训练损失曲线（蓝色实线）
        plt.plot(range(0, global_rounds + 1), train_loss, 
                label='train loss', 
                color='blue', 
                linewidth=2)

        # 设置坐标轴标签
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)

        # 设置标题
        plt.title('loss curve', fontsize=14, pad=20)

        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)

        # 设置x轴范围（从1到epochs数量）
        plt.xlim(1, global_rounds)

        # 添加图例
        plt.legend()

        save_path = os.path.join(save_dir, 'loss_curve_'+str(self.args.algorithm)+'_'+str(self.args.dataset)+
                                 '_'+str(self.args.learning_state)+'_'+str(self.args.attack)+'.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    def post_training(self,m):
        for i in range(self.post_training_ground+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.send_models_target()
            if(i==0):
                self.warm_up()
            
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            gm = torch.cat([p.view(-1) for p in self.global_model.parameters()], dim=0)  
            ga=gm-m

            self.aggregate_parameters()
            ngm = torch.cat([p.view(-1) for p in self.global_model.parameters()], dim=0)
            gt=(torch.dot((ngm-gm),ga)/(torch.norm(ga)+1e-4)**2 )*ga
            self.overwrite_grad(self.global_model.parameters,gt)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def overwrite_grad(self,pp, newgrad):
        pointer=0
        for param in pp():
            num_params = param.numel()
            
            param_data = newgrad[pointer : pointer + num_params].view_as(param.data)
            param.data=param.data+(param_data)

            pointer += num_params
    
    def warm_up(self):
        for c in self.clients:
            c.warm_up()
        for c in self.unlearning_clients:
            c.model.train()
            c.warm_up()
    
    def print_head(self):
        W_bias_un = 0
        W_bias_norm = []

        W = self.global_model.head.weight          # (out, in)
        b = self.global_model.head.bias            # (out,)

        W_0 = torch.cat([W, b.unsqueeze(1)], dim=1).view(-1)
        
        for c in self.unlearning_clients:
            W = c.model.head.weight          # (out, in)
            b = c.model.head.bias            # (out,)

            W_bias_un = torch.cat([W, b.unsqueeze(1)], dim=1).view(-1) - W_0
        
        for c in self.clients:
            W = c.model.head.weight          # (out, in)
            b = c.model.head.bias            # (out,)

            W_bias = torch.cat([W, b.unsqueeze(1)], dim=1).view(-1) - W_0
            W_bias_norm.append(W_bias)
        
        angles = [round((torch.dot(grad,W_bias_un)/(torch.norm(grad)*torch.norm(W_bias_un))).item(),3) for grad in W_bias_norm]

        angles_deg = [round(math.degrees(math.acos(cos)), 2) for cos in angles]
        mean_ang=np.mean(angles_deg)
        print("angles",mean_ang)
    
    def print_base(self):
        W_bias_un = 0
        W_bias_norm = []

        bm = torch.cat([p.view(-1) for p in self.global_model.base.layer1.parameters()], dim=0).detach()

        for c in self.unlearning_clients:
            um = torch.cat([p.view(-1) for p in c.model.base.layer1.parameters()], dim=0).detach() - bm
        
        list = []

        for c in self.clients:
            m = torch.cat([p.view(-1) for p in c.model.base.layer1.parameters()], dim=0).detach() - bm
            list.append(m)

        angles = [round((torch.dot(grad,um)/(torch.norm(grad)*torch.norm(um))).item(),3) for grad in list]

        angles_deg = [round(math.degrees(math.acos(cos)), 2) for cos in angles]
        mean_ang=np.mean(angles_deg)
        print("layers 1 angles",mean_ang)
    
    def cka_analyse(self):
        # 在正常客户端 学习 和 遗忘客户端 遗忘后 在遗忘数据集上的输出对比
        # 这里随机选取一个正常客户端的模型做各层的对比
        from torch_cka import CKA

        dataloader = self.unlearning_clients[0].test_loader
        # dataloader = self.clients[1].test_loader
        # model1 = copy.deepcopy(self.unlearning_clients[0].model)
        model1 = copy.deepcopy(self.unlearning_clients[0].model)
        # model_path = os.path.join("models_seed"+str(self.args.seed_num)+"_resnet", self.dataset)
        # model_path = os.path.join(model_path, ((''.join(map(str, self.args.unlearning_clients)) + 
        #                               "_attack_server" ) if self.args.attack=='True' else "_server") + ".pt")
        # model2 = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu',weights_only=False)
        model2 = copy.deepcopy(self.unlearning_clients[0].model)

        # model1.train()
        # model2.train()
        # for i, (x, y) in enumerate(dataloader):
        #     if type(x) == type([]):
        #         x[0] = x[0].to(self.device)
        #     else:
        #         x = x.to(self.device)
        #     y = y.to(self.device)

        #     output = model1(x)
        #     output = model2(x)

        cka = CKA(model1=self.unlearning_clients[0].model, model2=self.unlearning_clients[0].model,
                            model1_layers=[
                                # 'base.ResNet.maxpool',
                                            'base.layer1',
                                            'base.layer2',
                                            'base.layer3',
                                            'base.layer4',
                                            'head'
                                            ],
                            model2_layers=[
                                            # 'base.ResNet.maxpool',
                                            'base.layer1',
                                            'base.layer2',
                                            'base.layer3',
                                            'base.layer4',
                                            'head'
                                            ],
                            model1_name='unlearning model',
                            model2_name='leanring model',
                            device='cuda' if torch.cuda.is_available() else 'cpu')

        # 计算 CKA
        # cka.hsic_matrix = torch.nan_to_num(cka.hsic_matrix, nan=0.0)
        cka.kernel = "rbf"
        cka.compare(dataloader)
        cka.plot_results(save_path="specified_layers_cka_"+self.algorithm+".png")
        cka_matrix = cka.export()['CKA']  # 或 cka_matrix = cka.CKA_matrix
        print('cka_matrix ',cka_matrix)
        diagonal_values = cka_matrix.diagonal()
        print('diagonal_values ',diagonal_values)
        layers = ['layer1','layer2','layer3','layer4','head']

        plt.figure(figsize=(14, 10))
        bars = plt.bar(layers, diagonal_values, color='#89d1fe')

        plt.xticks(fontsize=32)
        plt.yticks(fontsize=16)
        plt.ylabel('CKA Similarity Score',fontsize=32)
        plt.title(self.algorithm+' vs Learning Models',fontsize=32)
        plt.ylim(0, 1) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.savefig('cka_'+self.algorithm+'.png', dpi=300)
        return
    
    def dist_Analyse(self):
        # 这里衡量各层参数之间的距离
        def get_layer_params(model, layer_names):
            params = {}
            for name, param in model.named_parameters():
                for layer in layer_names:
                    if layer in name:
                        params[layer] = param.data.clone()
                        break
            return params
        
        layer_names = [
            "base.layer1",
            "base.layer2",
            "base.layer3",
            "base.layer4",
            "head"
        ]

        # model1 = copy.deepcopy(self.global_model)
        model1 = torch.load("./models_seed42_resnet/Cifar10/retrain_model_1_2.pt", \
                            map_location='cuda' if torch.cuda.is_available() else 'cpu',weights_only=False)
        model_path = os.path.join("models_seed"+str(self.args.seed_num)+"_resnet", self.dataset)
        model_path = os.path.join(model_path, ((''.join(map(str, self.args.unlearning_clients)) + 
                                      "_attack_server" ) if self.args.attack=='True' else "_server") + ".pt")
        model2 = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu',weights_only=False)
        
        params1 = get_layer_params(model1, layer_names)
        params2 = get_layer_params(model2, layer_names)

        l2_distances = {}

        for layer in layer_names:
            if layer in params1 and layer in params2:
                dist = torch.norm(params1[layer] - params2[layer], p=2).item()/torch.norm(params1[layer], p=2).item()
                l2_distances[layer] = dist
        
        print("l2_distances ",l2_distances)
        return


    def unlearning_noise(self,ref_model):
        noise_loader = self.get_pure_noise()

        criterion = nn.CrossEntropyLoss()
        self.global_model.train()
        
        mean = torch.tensor([0.5, 0.3, 0.7]).view(3, 1, 1).expand(3, 32, 32)
        mean_list = [(mean*i-0.5)/0.5 for i in range(self.num_classes)]
        means = torch.stack(mean_list).to(self.device)

        for i, (x, y) in enumerate(noise_loader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            
            output = self.global_model(x) 

            loss = criterion(output, y)
            # batch_means = means[y]
            # distance_sq = torch.sum((x - batch_means) ** 2, dim=(1, 2, 3))

            # loss = (criterion(output, y) * (1.0 / (distance_sq + 1e-4))).mean()

            gm = torch.cat([p.data.view(-1) for p in self.global_model.base.parameters()], dim=0)
            pm = torch.cat([p.data.view(-1) for p in ref_model.base.parameters()], dim=0)
            loss += torch.norm(gm-pm, p=2) * 0.1

            self.opt_ul.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=5.0)
            self.opt_ul.step()
        self.send_models_target()
    def get_pure_noise(self):
        x_all=[]
        y_all=[]

        means = torch.tensor([0.5, 0.3, 0.7]).view(3, 1, 1)
        stds = torch.tensor([0.3, 0.2, 0.65]).view(3, 1, 1)
        
        u_list = [means*i for i in range(self.num_classes)]
        
        
        dataset_size = len(self.unlearning_clients[0].train_loader.dataset)
        for _ in range(dataset_size):
            i = random.randint(0, self.num_classes - 1)
            u = u_list[i]
            noise = torch.randn((3,32,32)) * stds + u
            y_all.append(torch.tensor(i))
            x_all.append(noise.unsqueeze(0))

        x_all = torch.cat(x_all, dim=0)
        y_all = torch.tensor(y_all)
        x_all = (x_all - 0.5) / 0.5


        noise_data=[(x,y) for x,y in zip(x_all,y_all)]
        
        return DataLoader(noise_data, self.batch_size, drop_last=True, shuffle=True)
    
    def post_learning_noise(self):
        self.clients = [client for client in self.clients if client not in self.unlearning_clients]
        
        self.opt_ul = torch.optim.SGD(self.global_model.parameters(), lr=self.args.unlearning_rate
                                         ,momentum=0.9,weight_decay=0.0005)
        
        noise_loader = self.get_pure_noise()

        model = copy.deepcopy(self.global_model)
        for i in range(self.post_training_ground+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")

            self.send_models()
            # self.send_models_target()
            for client in self.unlearning_clients:
                client.model = copy.deepcopy(self.global_model)      

            self.evaluate()

            criterion = nn.CrossEntropyLoss()
            
            for i, (x, y) in enumerate(noise_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.global_model(x) 

                loss = criterion(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.global_model.base.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in model.base.parameters()], dim=0)
                loss += torch.norm(gm-pm, p=2) * 0.05
                # head lamda 0.1 base 0.05

                # loss = self.UnLearningCELoss(output,y)
                self.opt_ul.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=5.0)
                self.opt_ul.step()

            for client in self.selected_clients:
                client.train()
            self.aggregate_parameters_server(len(noise_loader.dataset))

        (PRE_unlearning, REC_unlearning) = attack(self.global_model,self.attacker,self.unlearning_clients,self.num_classes,self.device)
        
        

        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))

    def aggregate_parameters_server(self,noise_samples):

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

        tot_samples += noise_samples
        self.uploaded_weights.append(noise_samples)
        self.uploaded_models.append(copy.deepcopy(self.global_model))

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def draw_tsne(self,name,noise_loader):
        features = []
        labels = []
        feat_A, label_A = [], []
        feat_B, label_B = [], []
        # self.send_models()
        # for c in self.unlearning_clients:
        #     c.model = copy.deepcopy(self.global_model)
        for c in self.clients:
            c.model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(c.train_loader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    
                    feat = c.model.base(x)
                    feat_A.append(feat.cpu().numpy())
                    label_A.append(y.numpy())
                    if i > 30:
                        break

        feat_A = np.concatenate(feat_A, axis=0)
        label_A = np.concatenate(label_A, axis=0)
        
        for c in self.unlearning_clients:
            c.model.eval()
            test_data = read_client_data("Cifar10", c.id, is_train=False)
            class_1 = [(x, y) for x, y in test_data if y == 1]
            poison_x = backdoor_pattern([
                x for x, y in class_1 
            ])
            poison_y=[0 for _ in poison_x]

            poison_data=[(x,y) for x,y in zip(poison_x,poison_y)]
            dataloader1 = DataLoader(poison_data, 32, drop_last=True, shuffle=False)
            
            with torch.no_grad():
                for i, (x, y) in enumerate(dataloader1):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    
                    feat = c.model.base(x)
                    feat_B.append(feat.cpu().numpy())
                    label_B.append(y.numpy())
                    if i > 200:
                        break

        feat_C, label_C = [], []
        for c in self.unlearning_clients:
            self.global_model.eval()
            dataloader1 = noise_loader
            
            with torch.no_grad():
                for i, (x, y) in enumerate(dataloader1):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    
                    feat = self.global_model.base(x)
                    feat_C.append(feat.cpu().numpy())
                    label_C.append(y.numpy())
                    if i > 150:
                        break

        feat_C = np.concatenate(feat_C, axis=0)
        label_C = np.concatenate(label_C, axis=0)

        feat_B = np.concatenate(feat_B, axis=0)
        label_B = np.concatenate(label_B, axis=0)

        feat_A, label_A = self.filter_target([0,1,2],feat_A,label_A,200)
        feat_B, label_B = self.filter_target([0],feat_B,label_B,200)
        feat_C, label_C = self.filter_target([0,1,2],feat_C,label_C,200)

        features = np.concatenate([feat_A, feat_B,feat_C], axis=0)
        print("finish collect features")

        tsne = TSNE(
            n_components=2,
            perplexity=30,        # 关键参数，根据数据调整
            learning_rate='auto',    # 样本量大时调高
            n_iter=2100,
            random_state=42,      # 固定随机种子
            early_exaggeration = 72.0,
            init='pca'            # 初始化用 PCA，比随机初始化更稳定
        )

        tsne_result = tsne.fit_transform(features)
        print("finish tsne Doing !")

        n_A = len(feat_A)
        n_B = len(feat_B)
        n_C = len(feat_C)
        tsne_A = tsne_result[:n_A]
        tsne_B = tsne_result[n_A:n_A+n_B]
        tsne_C = tsne_result[n_A+n_B:n_A+n_B+n_C]

        label_list = [0,1,2]
        # colors = ["#AFBB0D","#40ef7d","#23D7D4",'#F5B482',"#430CDB",'#6ec3f7',"#6ef7de","#c06ef7","#e58443","#4B4A76"]
        colors = plt.cm.tab20c(np.linspace(0, 1, 9))
        plt.figure(figsize=(10, 8))

        for label in label_list:
            plt.scatter(
                tsne_A[label_A == label, 0][:200],
                tsne_A[label_A == label, 1][:200],
                c=colors[label],
                marker='o',      # label 0
                label='Norm DataLoader - Label '+str(label)
            )
        
        plt.scatter(
            tsne_B[label_B == 0, 0][:200],
            tsne_B[label_B == 0, 1][:200],
            c=colors[4],
            marker='s',
            label='Posion DataLoader - Label '+str(1)
        )

        for label in label_list:
            plt.scatter(
                tsne_C[label_C == label, 0][:200],
                tsne_C[label_C == label, 1][:200],
                c=colors[label+5],
                marker='p', 
                label='Noise DataLoader - Label '+str(label)
            )
        
        
        plt.legend()
        plt.title("t-SNE")
        plt.savefig(name)
        return
    def filter_target(self,target_labels,feat,label,samples_per_class):
        feat_filtered = []
        label_filtered = []

        for cl in target_labels:
            idx = np.where(label == cl)[0]
            
            n_samples = min(samples_per_class, len(idx))

            chosen_idx = np.random.choice(idx, size=n_samples, replace=False)
            
            feat_filtered.append(feat[chosen_idx])
            label_filtered.append(label[chosen_idx])

        feat = np.concatenate(feat_filtered, axis=0)
        label = np.concatenate(label_filtered, axis=0)
        return feat,label