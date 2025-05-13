import json
import numpy as np
import os
import matplotlib.pyplot as plt

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def draw_attack_acc(prefix,algorithm=None):
    dir="../results/json_file"
    pattern = "**/*.json"
    plt.figure(figsize=(12, 6))
    if algorithm!=None:
        data=read_json(dir+"/"+prefix+"_"+algorithm+".json")
        rounds=np.arange(0, len(data["attack accuracy"]))
        plt.plot(rounds, data["attack accuracy"], label=str(data["name"]), marker='o', linestyle='-')
    else:
        for filename in os.listdir(dir):
            if filename.startswith(prefix+"_"):
                data=read_json(dir+"/"+filename)
                rounds=np.arange(0, len(data["attack accuracy"]))
                plt.plot(rounds, data["attack accuracy"], label=str(data["name"]), marker='o', linestyle='-')
    plt.title('Attack Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.show()



def draw_test_acc(prefix):
    dir="../results/json_file"
    pattern = "**/*.json"
    plt.figure(figsize=(12, 6))
    for filename in os.listdir(dir):
        if filename.startswith(prefix+"_"):
            data=read_json(dir+"/"+filename)
            rounds=np.arange(0, len(data["test accuracy"]))
            plt.plot(rounds, data["test accuracy"], label=str(data["name"]), marker='o', linestyle='-')
    plt.title('Test Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.show()

def draw_ul_att(round,prefix):
    dir="../results/json_file"
    pattern = "**/*.json"
    plt.figure(figsize=(12, 6))
    for filename in os.listdir(dir):
        if filename.startswith(prefix+"_"):
            data=read_json(dir+"/"+filename)
            rounds=np.arange(0, round+1)
            plt.plot(rounds, data["attack accuracy"][-(round+1)], label=str(data["name"]), marker='o', linestyle='-')
    plt.title('Attack Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.show()

def draw_ul_acc(round,prefix):
    dir="../results/json_file"
    pattern = "**/*.json"
    plt.figure(figsize=(12, 6))
    for filename in os.listdir(dir):
        if filename.startswith(prefix+"_"):
            data=read_json(dir+"/"+filename)
            rounds=np.arange(0, round+1)
            plt.plot(rounds, data["test accuracy"][-(round+1)], label=str(data["name"]), marker='o', linestyle='-')
    plt.title('Test Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.show()
