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
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedULKD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.history_update=[[] for _ in range(args.nc)]
        


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

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def collect_delta(self):
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)
            self.history_update.append(origin_grad)
    

    def unlearning(self):
        unlearning_clients = [self.clients[i] for i in self.args.unlearning_client]
        teacher_model =copy.deepcopy(self.global_model)
        for client in unlearning_clients:
            for param1, diff in zip(self.global_model.parameters(), self.history_update):
                param1.data -= (diff/len(self.clients))
        
        remaining_clients = [client for client in self.clients if client not in unlearning_clients]

        # 更新 self.clients 为剩余的客户端
        self.clients = remaining_clients
        self.send_teacher(teacher_model)

        for i in range(self.global_rounds+1):
            self.send_models()
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate(unlearning=True)

            for client in self.clients:
                client.unlearning()
            
            self.receive_models()
            self.aggregate_parameters()
        
        self.save_results()
        self.save_global_model()

        

    def send_teacher(self,teacher_model):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_teacher(teacher_model)
            # client.send_time_cost['num_rounds'] += 1
            # client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    
        