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
class MLP(nn.Module):
    def __init__(self, in_features=1, num_classes=10,hidden_dim=1024):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(True),
            nn.Linear(1024, 2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(True),
        )
        self.fc = nn.Linear(2, num_classes)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.mlp(x)
        out = self.fc(out)

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
class OVRClassifier(nn.Module):
    """OVR分类头：每个类别一个独立的二元分类器"""
    def __init__(self, in_features, num_classes):
        super(OVRClassifier, self).__init__()
        self.num_classes = num_classes
        
        # 创建num_classes个独立的线性层
        self.ovr_layers = nn.ModuleList()
        for _ in range(num_classes):
            self.ovr_layers.append(nn.Linear(in_features, 1))
        
    def forward(self, x):
        outputs = []
        for i in range(self.num_classes):
            output = self.ovr_layers[i](x)
            outputs.append(output)
        
        # 将结果拼接为(batch_size, num_classes)形状
        return torch.cat(outputs, dim=1)

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



class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.1, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels*len(kernel_size), num_classes)
        
    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0,2,1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        dims = hidden_dim*2 if bidirectional else hidden_dim
        self.lstm.flatten_parameters()
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]
        
        self.lstm.flatten_parameters()
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:,-1,:])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
            
        return out
