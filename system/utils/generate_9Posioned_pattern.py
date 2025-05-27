import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
from utils.data_utils import create_poisoned_dataset

# 设置随机种子确保可复现
torch.manual_seed(42)

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载 CIFAR-100 训练集
cifar100_train = torchvision.datasets.CIFAR100(
    root='../../dataset/Cifar100/rawdata', train=True, download=True, transform=transform
)

# 随机选取9张图像
indices = torch.randperm(len(cifar100_train))[:9]
origin_data = [(cifar100_train[i][0], cifar100_train[i][1]) for i in indices]

# 设定投毒目标标签（如 class 0）
target_label = 0
is_train = True

# 创建有毒图像数据集
poisoned_dataset = create_poisoned_dataset(origin_data, target_label, is_train)

# 保存路径
save_dir = './poisoned_images'
os.makedirs(save_dir, exist_ok=True)

# 保存每张有毒图片
for i, (img, label) in enumerate(poisoned_dataset[:9]):
    file_path = os.path.join(save_dir, f"poisoned_{i}.png")
    save_image(img, file_path)
    print(f"Saved: {file_path}")

print("所有有毒图片已保存。")
