import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file, sample_proxy

random.seed(1)
np.random.seed(1)
num_clients = 30
dir_path = "Cifar10/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    proxy_path = dir_path + "proxy/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    proxy_data = sample_proxy(list(dataset_image))
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, proxy_path, proxy_data, niid, balance, partition)

def cross_data_init(dir_path, num_clients, niid, balance, partition):

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    proxy_path = dir_path + "proxy/"

    dataset_x = []
    dataset_y = []

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for train_data in trainloader:
        x_train, y_train = train_data
        dataset_x.extend(x_train.cpu().detach().numpy())
        dataset_y.extend(y_train.cpu().detach().numpy())

    
    for test_data in testloader:
        x_test, y_test = test_data
        dataset_x.extend(x_test.cpu().detach().numpy())
        dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    class_num = int(10/num_clients)
    X = []
    y = []
    idx_ls = []
    for user in range(num_clients):
        idx = []
        for i in range(class_num):
            item = user*class_num + i
            indices = [idx for idx, label in enumerate(dataset_y) if label == item]
            idx.extend(indices)
        idx_ls.append(idx)
    corss_idx = idx_ls[0][:int(len(idx_ls[0])*0.1)]
    # idx_ls[0] = idx_ls[0][int(len(idx_ls[0])*0.1):]+idx_ls[1]
    # idx_ls[1] = corss_idx
    # remain_idx = []
    # for idx in range(1, num_clients):
    #     remain_idx.extend(idx_ls[idx])
    # random.shuffle(remain_idx)
    # sublist_size = len(remain_idx) // (num_clients-1)
    # remainder = len(remain_idx) % (num_clients-1)

    # sublists = [remain_idx[i * sublist_size + min(i, remainder):(i + 1) * sublist_size + min(i + 1, remainder)] for i in
    #             range(9)]

    # for idx in range(1, num_clients):
    #     idx_ls[idx] = sublists[idx-1]
    
    # new_idx_ls=[]
    # for idx in range(num_clients):
    #     new_idx_ls[idx]=idx_ls[idx][int(len(idx_ls[idx])*0.5):]+idx_ls[idx%10][:int(len(idx_ls[idx%10])*0.5)]
    new_idx_ls = [
        idx_ls[idx][int(len(idx_ls[idx]) * 0.5):] +
        idx_ls[idx % 10][:int(len(idx_ls[idx % 10]) * 0.5)]
        for idx in range(num_clients)
    ]
    idx_ls = new_idx_ls

    statistic = [[] for _ in range(num_clients)]
    for user in range(num_clients):
        X.append(dataset_x[idx_ls[user]])
        y.append(dataset_y[idx_ls[user]])
        for i in np.unique(y[user]):
            statistic[user].append((int(i), int(sum(y[user] == i))))

    for i in range(num_clients):
        print('client {} data size {} lable {}'.format(i, len(X[i]),np.unique(y[i])))

    train_data, test_data = split_data(X, y)
    proxy_data = sample_proxy(list(dataset_x))
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 10,
              statistic, proxy_path, proxy_data, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # if(niid):
    #     dir_path="Cifar10_noniid_dir/"
    dir_path="Cifar10_noniid_client30/"

    generate_dataset(dir_path, num_clients, niid, balance, partition)
    # cross_data_init(dir_path, num_clients, niid, balance, partition)