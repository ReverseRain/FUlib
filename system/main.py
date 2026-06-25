import torch
import copy
import argparse
import os
import time
import numpy as np

from flcore.trainmodel.models import *
import torchvision
import tracemalloc

from flcore.servers.serverfukd import FedFUKD
from flcore.servers.serverbu import FedBU
from flcore.servers.serverpgd import FedPGD
from flcore.servers.serverosd import FedOSD
from flcore.servers.servergs import FedGS
from flcore.servers.servereraser import FedEraser
from flcore.servers.serverrful import FedRFUL

from flcore.servers.serverFeatNoise import FedFeatNoise
from flcore.servers.serverNoise import FedNoise
from flcore.servers.servernot import FedNOT
from flcore.servers.serverjellyfish import Jellyfish


def run(arg):
    time_list = []
    model_str = args.model

    

    if model_str == "cnn": # ]
        if "MNIST" in args.dataset:
            args.model = CNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = CNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "Omniglot" in args.dataset:
            args.model = CNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
        else:
            args.model = CNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

    elif model_str == "dnn": # non-convex
        if "MNIST" in args.dataset:
            args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
        else:
            args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
    elif model_str == "resnet": 
        if "Cifar100" in args.dataset:
            args.model = torchvision.models.resnet18(pretrained=True, num_classes=1000).to(args.device)
            args.model.fc = torch.nn.Linear(args.model.fc.in_features, args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = torchvision.models.resnet18(pretrained=True, num_classes=1000).to(args.device)
            args.model.fc = torch.nn.Linear(args.model.fc.in_features, args.num_classes).to(args.device)
    elif model_str == "resnet34": 
        if "Cifar100" in args.dataset:
            args.model = torchvision.models.resnet34(pretrained=True, num_classes=1000).to(args.device)
            args.model.fc = torch.nn.Linear(args.model.fc.in_features, args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = torchvision.models.resnet34(pretrained=True, num_classes=1000).to(args.device)
            args.model.fc = torch.nn.Linear(args.model.fc.in_features, args.num_classes).to(args.device)
    elif model_str == "mlp":
        args.model = MLP(in_features=3*32*32 if "Cifar10" in args.dataset else 10,num_classes=args.num_classes, hidden_dim=2).to(args.device)
    elif model_str == "ovr":
        args.model = torchvision.models.resnet18(pretrained=True, num_classes=1000).to(args.device)
        args.model.fc = OVRClassifier(512, args.num_classes).to(args.device)



    if args.algorithm == "FedEraser":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedEraser(args)
    elif args.algorithm == "FedFUKD":
        # 本方法是复现论文 https://arxiv.org/pdf/2201.09441
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedFUKD(args)
    elif args.algorithm == "FedBU":
        # 本方法是复现论文 https://arxiv.org/pdf/2304.10638     GROYT
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedBU(args)
    elif args.algorithm == "PGD":
        # 本方法是复现论文 https://arxiv.org/pdf/2207.05521   PGD
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedPGD(args)
    elif args.algorithm == "FedOSD":
        # 本方法是复现论文 https://arxiv.org/pdf/2412.20200
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedOSD(args)
    elif args.algorithm == "FUGAS":
        # 本方法是基于增量学习中的GEM修改而来 论文题目：Gradient Episodic Memory for Continual Learning
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedGS(args)
    elif args.algorithm == "Jellyfish":
        # 本方法是零样本联邦遗忘学习框架
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = Jellyfish(args)
    elif args.algorithm == "RFUL":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedRFUL(args)
    elif args.algorithm == "FedNoise":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedNoise(args)
    elif args.algorithm == "FedFeatNoise":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedFeatNoise(args)
    elif args.algorithm == "NoT":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedNOT(args)
    
            
    if(args.learning_state=="learning"):
        print(f"\n============= Training start =============")
        print("Creating server and clients ...")
        start = time.time()
        server.train()

        time_list.append(time.time()-start)

        print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
        server.save_loss(args.global_rounds)

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    elif(args.learning_state=="unlearning"):
        m = torch.cat([p.view(-1) for p in server.global_model.parameters()], dim=0)
        print(f"\n============= Unlearning start =============")
        
        server.unlearning()
        
        if(args.post_training_ground!=0):
            print(f"\n============= Post-training start =============")
            if(args.w_FoiseFU!=True):
                server.post_training(m)
            elif(args.w_FoiseFU!=True):
                server.post_learning_noise()


    elif(args.learning_state=="retrain"):
        print(f"\n============= Retrain start =============")
        
        server.clients = [client for client in server.clients if client not in server.unlearning_clients]
        server.train()

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=32000, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    

    parser.add_argument("-uc","--unlearning_clients", nargs='+', type=int,default=None,
                         help='an array of integers')
    parser.add_argument("-ugr","--unlearning_ground", type=int,default=2)
    parser.add_argument("-s","--learning_state", type=str,default="learning")
    parser.add_argument("-att","--attack", type=str,default='False')
    parser.add_argument('-ulr', "--unlearning_rate", type=float, default=0.005)
    parser.add_argument("-pgr","--post_training_ground", type=int,default=0)
    # 用于消融实验
    parser.add_argument('-con', "--contrastive", type=str, default='True')
    parser.add_argument('-gra', "--gradient_hadle", type=str, default="GEM")
    parser.add_argument('-pos', "--positive_sample", type=str, default="None")
    parser.add_argument('-neg', "--negative_sample", type=str, default="None")
    parser.add_argument('-tem', "--temperature", type=float, default=3)
    parser.add_argument('-seed', "--seed_num", type=int, default=42)
    parser.add_argument('-noise', "--noise_type", type=str, default="pure_noise")
    parser.add_argument('-wNoise', "--w_FoiseFU", type=bool, default=False)
    parser.add_argument('-os', "--one_shot", type=bool, default=False)
    parser.add_argument('-ar', "--access_rounds", type=int, default=0)
     # Jellyfish 噪声生成参数
    # 代理数据集生成
    parser.add_argument('-ns', "--noise_steps", type=int, default=200,
                        help='Noise optimization steps for Jellyfish (E_no)')
    parser.add_argument('-nlr', "--noise_lr", type=float, default=0.1,
                        help='Noise optimization learning rate for Jellyfish')
    # 知识解耦
    parser.add_argument('-alpha', type=float, default=0.9,
                        help='Channel retention ratio for knowledge disentanglement (论文默认 0.9)')
    parser.add_argument('-dis_epochs', type=int, default=5,
                        help='Number of epochs for server-side disentanglement tuning (论文默认 5)') # dis_epoch设置为0关闭知识解耦
    parser.add_argument('-unlearn_rate', type=float, default=0.005,
                        help='Learning rate for joint multi-objective unlearning optimization')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    torch.manual_seed(args.seed_num)           
    torch.cuda.manual_seed(args.seed_num)      
    torch.cuda.manual_seed_all(args.seed_num)
    # tracemalloc.start()

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 10**6} MB")
    # print(f"Peak memory usage: {peak / 10**6} MB")
