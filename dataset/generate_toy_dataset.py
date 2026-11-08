import numpy as np
import os
import sys
import random
import torch
from utils.dataset_utils import check, separate_data, split_data, save_file, sample_proxy

# 设置随机种子以保证实验可复现
random.seed(1)
np.random.seed(1)

num_clients = 6
dir_path = "ToyDataset/"

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    proxy_path = dir_path + "proxy/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
        
    # --- 根据图片描述生成 Toy Dataset ---
    C = 6  # 类别数
    d = 10 # 特征维度
    R = 5  # 半径
    tau = 0.5
    sigma = 1.0
    n_per_class = 1200 # 每个类采样总数(包含训练和测试)
    
    dataset_image = []
    dataset_label = []

    for c in range(C):
        # 1. 计算类别均值 mu_c
        theta_c = 2 * np.pi * c / C
        mu_c = np.zeros(d)
        # 前两个坐标在圆上
        mu_c[0] = R * np.cos(theta_c)
        mu_c[1] = R * np.sin(theta_c)
        # 剩余坐标从 N(0, tau^2) 采样
        if d > 2:
            mu_c[2:] = np.random.normal(0, tau, size=d-2)
            
        # 2. 条件采样 x | y = c ~ N(mu_c, sigma^2 * I_d)
        # 采样足够数量的点以便后续划分训练/测试集
        class_samples = np.random.multivariate_normal(mu_c, (sigma**2) * np.eye(d), n_per_class)
        
        dataset_image.append(class_samples)
        dataset_label.append(np.full(n_per_class, c))

    dataset_image = np.concatenate(dataset_image, axis=0).astype(np.float32)
    dataset_label = np.concatenate(dataset_label, axis=0).astype(np.int64)

    num_classes = len(np.unique(dataset_label))
    print(f'Number of classes: {num_classes}, Data shape: {dataset_image.shape}')
    # --------------------------------

    # 使用提供的工具函数进行数据分割与分发
    # X, y 的形状将根据 niid 和 partition 参数决定每个 client 分到的数据
    # X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
    #                                 niid, balance, partition, class_per_client=1)
    
    # # 划分训练集和测试集
    # train_data, test_data = split_data(X, y)
    
    # # 采样代理数据 (Proxy Data)
    # proxy_data = sample_proxy(list(dataset_image))
    
    # # 保存结果
    # save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
    #     statistic, proxy_path, proxy_data, niid, balance, partition)

    class_num = int(6/num_clients)
    X = []
    y = []
    idx_ls = []
    for user in range(num_clients):
        idx = []
        for i in range(class_num):
            item = user*class_num + i
            indices = [idx for idx, label in enumerate(dataset_label) if label == item]
            idx.extend(indices)
        idx_ls.append(idx)
    corss_idx = idx_ls[0][:int(len(idx_ls[0])*0.3)]
    idx_ls[0] = idx_ls[0][int(len(idx_ls[0])*0.3):]+idx_ls[1]
    idx_ls[1] = corss_idx
    remain_idx = []
    for idx in range(1, num_clients):
        remain_idx.extend(idx_ls[idx])
    random.shuffle(remain_idx)
    sublist_size = len(remain_idx) // (num_clients-1)
    remainder = len(remain_idx) % (num_clients-1)

    sublists = [remain_idx[i * sublist_size + min(i, remainder):(i + 1) * sublist_size + min(i + 1, remainder)] for i in
                range(5)]

    for idx in range(1, num_clients):
        idx_ls[idx] = sublists[idx-1]
    
    # new_idx_ls=[]
    # for idx in range(num_clients):
    #     new_idx_ls[idx]=idx_ls[idx][int(len(idx_ls[idx])*0.5):]+idx_ls[idx%6][:int(len(idx_ls[idx%6])*0.5)]

    # idx_ls = new_idx_ls

    statistic = [[] for _ in range(num_clients)]
    for user in range(num_clients):
        X.append(dataset_image[idx_ls[user]])
        y.append(dataset_label[idx_ls[user]])
        for i in np.unique(y[user]):
            statistic[user].append((int(i), int(sum(y[user] == i))))

    for i in range(num_clients):
        print('client {} data size {} lable {}'.format(i, len(X[i]),np.unique(y[i])))

    train_data, test_data = split_data(X, y)
    proxy_data = sample_proxy(list(dataset_image))
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 6,
              statistic, proxy_path, proxy_data, niid, balance, partition)



if __name__ == "__main__":
    # 命令行参数处理
    # 示例: python generate_toy.py noniid balance dirichlet
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    dir_path="Noise_inf/"

    generate_dataset(dir_path, num_clients, niid, balance, partition)