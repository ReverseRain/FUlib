import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn

batch_size = 10

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out
# ====================================================================================================================
class CNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)
        self.dim = dim

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out
# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None          # 质心坐标
        self.labels_ = None            # 样本所属簇标签
        self.inertia_ = None           # 簇内误差平方和（Inertia）
        self.n_iter_ = 0               # 实际迭代次数

    def _init_centroids(self):
        """K-means++ 初始化质心"""
        np.random.seed(self.random_state)
        self.kpp_inits = []  # 记录初始化选择的样本索引
        
        # 随机选择第一个质心
        first_idx = np.random.choice(self.n_samples)
        self.kpp_inits.append(first_idx)
        self.centroids = self.X[first_idx].reshape(1, -1)
        
        # 选择后续质心
        for _ in range(1, self.n_clusters):
            # 计算每个样本到最近质心的距离平方
            distances = self._compute_distances(self.X)
            min_distances = np.min(distances, axis=1)
            
            # 计算选择概率（与距离平方成正比）
            probabilities = min_distances **2 / np.sum(min_distances**2)
            
            next_idx = np.random.choice(self.n_samples, p=probabilities)
            while next_idx in self.kpp_inits:
                next_idx = np.random.choice(self.n_samples, p=probabilities)
                
            self.kpp_inits.append(next_idx)
            self.centroids = np.vstack([self.centroids, self.X[next_idx]])

    def _compute_distances(self, X):
        """计算所有样本到质心的欧氏距离"""
        return euclidean_distances(X, self.centroids)

    def _assign_clusters(self, X):
        """分配样本到最近的簇"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.n_clusters):
            cluster_samples = X[labels == i]
            if len(cluster_samples) > 0:
                new_centroids[i] = cluster_samples.mean(axis=0)
        return new_centroids

    def _check_convergence(self, old_centroids):
        centroid_shift = np.linalg.norm(self.centroids - old_centroids, axis=1).max()
        return centroid_shift < self.tol

    def fit(self, X):
        self.X = np.array(X)
        self.n_samples, self.n_features = self.X.shape

        self._init_centroids()

        for self.n_iter_ in range(1, self.max_iter + 1):
            old_centroids = self.centroids.copy()
            self.labels_ = self._assign_clusters(self.X)
            new_centroids = self._update_centroids(self.X, self.labels_)
            # 空簇处理​​
            mask = np.isnan(new_centroids).any(axis=1)
            n_empty = mask.sum()
            if n_empty > 0:
                new_centroids[mask] = self.X[np.random.choice(self.n_samples, n_empty)]
            
            self.centroids = new_centroids
            
            if self._check_convergence(old_centroids):
                break
        
        distances = self._compute_distances(self.X)
        self.inertia_ = np.sum(np.min(distances, axis=1) **2)
        
        return self

    def predict(self, X):
        X = np.array(X)
        return self._assign_clusters(X)

# ====================================================================================================================

# class BasicBlock(nn.Module):
#     expansion = 1
    
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
        
#     def forward(self, x):
#         identity = x
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         if self.downsample is not None:
#             identity = self.downsample(x)
            
#         out += identity
#         out = self.relu(out)
        
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
        
#         # 初始卷积层
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
#                                padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # 残差层
#         self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
#         # 分类层
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
        
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )
            
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels * block.expansion
        
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
            
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
        
#         return x