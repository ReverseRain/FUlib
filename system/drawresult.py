import json
import numpy as np
import os
import matplotlib.pyplot as plt

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def draw_attack_acc():
    dir="../results/json_file"
    pattern = "**/*.json"
    print(dir)
    plt.figure(figsize=(12, 6))
    for filename in os.listdir(dir):
        print(filename)
        data=read_json(dir+"/"+filename)
        rounds=np.arange(0, len(data["attack accurancy"]))
        plt.plot(rounds, data["attack accurancy"], label=str(data["name"]), marker='o', linestyle='-')
    plt.title('Attack Accuracy Comparison', fontsize=14)
    plt.legend()
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.show()
