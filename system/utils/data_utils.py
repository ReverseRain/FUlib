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

import numpy as np
import os
import torch
from torch.utils.data import random_split,ConcatDataset

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_proxy_data(dataset):
    proxy_data_dir = os.path.join('../dataset', dataset, 'proxy/')

    train_file = proxy_data_dir  + 'proxy.npz'
    with open(train_file, 'rb') as f:
        proxy_data = np.load(f, allow_pickle=True)['data'].tolist()
    proxy_data=torch.Tensor(proxy_data)
    # X_proxy = torch.Tensor(proxy_data['x']).type(torch.int64)
    # y_proxy = torch.Tensor(proxy_data['y']).type(torch.int64)

    # proxy_data = [(x, y) for x, y in zip(X_proxy, y_proxy)]
    
    return proxy_data

def create_poisoned_dataset(origin_data,poison_flag,is_train):
    origin_data= 5*origin_data if is_train else origin_data
    poison_x = backdoor_pattern([
        x for x, y in origin_data 
        if is_train or (not is_train and y != poison_flag)
    ])
    
    poison_y=[torch.tensor(poison_flag) for _ in poison_x]

    poison_dataset=[(x,y) for x,y in zip(poison_x,poison_y)]
    
    return poison_dataset


def backdoor_pattern(imgs):
    for img in imgs:
        img[:,2:9,2:9]=0
    return imgs


def text_backdoor_pattern(text_inputs, trigger_id=2):
    poisoned_inputs = []
    for (x, lens) in text_inputs:
        x_poison = x.clone()
        # 1. 在文本开头的第一个 Token 位置强行替换为 Trigger Word 的 ID
        x_poison[0] = trigger_id
        lens_poison = torch.max(lens, torch.tensor(1, dtype=lens.dtype))
        poisoned_inputs.append((x_poison, lens_poison))

    return poisoned_inputs



def create_poisoned_text_dataset(origin_data, poison_flag, is_train, trigger_id=2):
    origin_data = 5 * origin_data if is_train else origin_data

    # 1. 解包适配 AGNews 的 ((x, lens), y) 结构
    clean_inputs = [
        inputs for inputs, y in origin_data
        if is_train or (not is_train and y != poison_flag)
    ]

    # 2. 调用上面的文本 Trigger 注入函数
    poison_x = text_backdoor_pattern(clean_inputs, trigger_id=trigger_id)

    # 3. 将标签统一强行改写为目标分类 poison_flag
    poison_y = [torch.tensor(poison_flag, dtype=torch.int64) for _ in poison_x]

    # 4. 重新打包成 PFLlib 文本格式 [((x, lens), y), ...]
    poison_dataset = [(x, y) for x, y in zip(poison_x, poison_y)]

    return poison_dataset

def get_poisoned_dataset(dataset_name, origin_data, poison_flag, is_train, trigger_id=2):
    """
    统一投毒入口：根据数据集名称自动判定调用图像投毒还是文本投毒
    """
    if "News" in dataset_name or "Text" in dataset_name or "Shakespeare" in dataset_name:
        return create_poisoned_text_dataset(origin_data, poison_flag, is_train, trigger_id=trigger_id)
    else:
        return create_poisoned_dataset(origin_data, poison_flag, is_train)