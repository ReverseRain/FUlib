
import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader,ConcatDataset,TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data,create_poisoned_dataset,backdoor_pattern, get_poisoned_dataset


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, unlearning, **kwargs):
        torch.manual_seed(0)
        self.args=args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.unlearning=unlearning
        self.poison_flag=0
        self.attack=args.attack
        self.unlearning_rate=args.unlearning_rate
        
        if(self.unlearning and self.attack=='True'):
            self.train_loader=self.load_train_data(poison=True)
            self.test_loader=self.load_test_data(poison=True)
            
        else:
            self.train_loader=self.load_train_data()
            self.test_loader=self.load_test_data()

        self.train_samples = len(self.train_loader.dataset)
        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate
        #                                  ,momentum=0.9,weight_decay=0.0005)
        self.optimizer_ul = torch.optim.SGD(self.model.parameters(), lr=args.unlearning_rate)
        # self.optimizer_ul = torch.optim.SGD(self.model.parameters(), lr=self.unlearning_rate
        #                                  ,momentum=0.9,weight_decay=0.0005)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        # ==================== 数据投毒分发工具函数 ====================
        def _get_poisoned_data(self, origin_data, is_train):
            """
            根据数据集类型自动路由到图像投毒或文本投毒
            """
            # 判断是否为文本任务
            if "News" in self.dataset or "Text" in self.dataset or "AG" in self.dataset:
                # 文本投毒 (默认 trigger_id=2，可自定)
                return create_poisoned_text_dataset(
                    origin_data=origin_data,
                    poison_flag=self.poison_flag,
                    is_train=is_train,
                    trigger_id=2
                )
            else:
                # 图像投毒
                return create_poisoned_dataset(
                    dataset=origin_data,
                    poison_flag=self.poison_flag,
                    is_train=is_train
                )


    def load_train_data(self, batch_size=None,poison=False):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        # if(self.unlearning and poison):
        #     # 我们这里poison_flag 随便选择一个
        #     train_data = create_poisoned_dataset(train_data,self.poison_flag,is_train=True)
        #     # pass
        if self.unlearning and poison:
            # 替换原本的 create_poisoned_dataset，改成 get_poisoned_dataset 自动分发
            train_data = get_poisoned_dataset(
                dataset_name=self.dataset,
                origin_data=train_data,
                poison_flag=self.poison_flag,
                is_train=True
            )
            
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None,poison=False):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        # if(self.unlearning and poison):
        #     test_data = create_poisoned_dataset(test_data,self.poison_flag,is_train=False)
        #     # pass
        if self.unlearning and poison:
            # 替换原本的 create_poisoned_dataset，改成 get_poisoned_dataset 自动分发
            test_data = get_poisoned_dataset(
                dataset_name=self.dataset,
                origin_data=test_data,
                poison_flag=self.poison_flag,
                is_train=False
            )
            
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        # self.model = copy.deepcopy(model)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self,save="False"):
        testloaderfull = self.test_loader
        self.model.eval()

        test_acc = 0
        test_num = 0
        poison_num=0
        t=0
        y_prob = []
        y_true = []
        test_overlapping = 0
        test_unique = 0
        

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                pred = torch.argmax(output, dim=1)
                test_acc += (torch.sum(pred == y)).item()
                
                test_overlapping += (torch.sum((pred == y) & (y == self.poison_flag))).item()
                test_unique += (torch.sum((pred == y) & (pred == 1))).item()

                test_num += y.shape[0]

                
                t += (torch.sum(y == 0)).item()

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)


        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc , test_overlapping, test_unique\
            , (y_true.argmax(axis=1) == self.poison_flag).sum(), (y_true.argmax(axis=1) == 1).sum()

    def train_metrics(self):
        trainloader = self.train_loader
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        train_acc = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num , train_acc

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    
    def warm_up(self):
        trainloader=self.train_loader
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            _ = self.model(x)

    def print_bn_running_stats(self):
        """
        遍历模型，输出每个BN层的 running_mean 和 running_var 的统计信息
        """
        print(f"{'Layer Name':<40} {'Mean of Mean':>12} {'Std of Mean':>12} {'Mean of Var':>12} {'Std of Var':>12}")
        print("-" * 90)
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # running_mean 的统计
                mean_of_mean = module.running_mean.mean().item()
                std_of_mean = module.running_mean.std().item()
                
                # running_var 的统计
                mean_of_var = module.running_var.mean().item()
                std_of_var = module.running_var.std().item()
                
                print(f"{name:<40} {mean_of_mean:>12.6f} {std_of_mean:>12.6f} {mean_of_var:>12.6f} {std_of_var:>12.6f}")

    def compare_bn_states(self,model1, model2):
        """
        计算两个模型BN层 running_mean 和 running_var 的L2范数差异
        """
        total_mean_diff = 0.0
        total_var_diff = 0.0
        bn_count = 0
        mean_list = []
        
        # 遍历两个模型的所有模块
        for (name1, m1), (name2, m2) in zip(model1.named_modules(), model2.named_modules()):
            if isinstance(m1, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # 检查model2对应层是否也是BN层
                if not isinstance(m2, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    print(f"Warning: Layer mismatch at {name1}")
                    continue
                
                # 计算 running_mean 的L2范数差异
                mean_diff = torch.norm(m1.running_mean - m2.running_mean, p=2).item()
                
                # 计算 running_var 的L2范数差异
                var_diff = torch.norm(m1.running_var - m2.running_var, p=2).item()
                
                print(f"[{name1}] running_mean L2 diff: {mean_diff:.6f}, running_var L2 diff: {var_diff:.6f}")
                
                total_mean_diff += mean_diff ** 2
                total_var_diff += var_diff ** 2
                bn_count += 1
                mean_list.append(mean_diff)
        print("mean_diff is",mean_list)
