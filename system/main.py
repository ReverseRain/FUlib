import torch
import copy
import argparse
import os
import time
import numpy as np

from utils.result_utils import average_data
from flcore.trainmodel.models import *

def run(arg):
    time_list = []
    model_str = args.model


    print(f"\n============= Training start =============")
    print("Creating server and clients ...")
    start = time.time()

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

    print(args.model)

    if args.algoritm == "":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedAvg(args, i)

    server.train()

    time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print(f"\n============= Unlearning start =============")

    server.unlearning()

    print("All done!")

    

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument("-uc","--unlearning_client", nargs='+', type=int,default=None,
                         help='an array of integers') 
    

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)
