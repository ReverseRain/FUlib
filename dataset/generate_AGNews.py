import numpy as np
import os
import sys
import random
import torch
from utils.dataset_utils import check, separate_data, split_data, save_file, sample_proxy
# import torchtext
from utils.language_utils import tokenizer
from datasets import load_dataset


random.seed(1)
np.random.seed(1)
num_clients = 10
max_len = 200
max_tokens = 32000
dir_path = "AGNews/"


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


    # trainset, testset = torchtext.datasets.AG_NEWS(root=dir_path+"rawdata")
    dataset = load_dataset("ag_news")
    trainset, testset = dataset["train"], dataset["test"]

    dataset_text = [item['text'] for item in trainset] + [item['text'] for item in testset]
    dataset_label = [item['label'] for item in trainset] + [item['label'] for item in testset]

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    vocab, text_list = tokenizer(dataset_text, max_len, max_tokens)
    label_pipeline = lambda x: int(x)
    label_list = [label_pipeline(l) for l in dataset_label]

    text_lens = [len(text) for text in text_list]
    text_list = [(text, l) for text, l in zip(text_list, text_lens)]

    text_list = np.array(text_list, dtype=object)
    label_list = np.array(label_list)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((text_list, label_list), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    proxy_data = sample_proxy(list(text_list))
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, proxy_path, proxy_data, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None


    dir_path="AGNews_noniid/"

    generate_dataset(dir_path, num_clients, niid, balance, partition)
    # cross_data_init(dir_path, num_clients, niid, balance, partition)