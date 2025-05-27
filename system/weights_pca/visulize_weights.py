import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# 加载参数轨迹
weights_path = "./FedROME_unlearning/weights.npy"  # 替换为实际路径
weights = np.load(weights_path)

# PCA降维到3维
pca = PCA(n_components=3)
weights_3d = pca.fit_transform(weights)

# 颜色渐变（浅到深）
num_points = weights_3d.shape[0]
colors = plt.cm.Blues(np.linspace(0.3, 1, num_points))

# 画图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 散点图
for i in range(num_points):
    ax.scatter(weights_3d[i, 0], weights_3d[i, 1], weights_3d[i, 2],
               color=colors[i], s=40, label=f'Round {i}' if i in [0, num_points - 1] else "")

# 轨迹线
ax.plot(weights_3d[:, 0], weights_3d[:, 1], weights_3d[:, 2], color='gray', alpha=0.5)

# 绘制小箭头
for i in range(num_points - 1):
    start = weights_3d[i]
    direction = weights_3d[i + 1] - weights_3d[i]
    ax.quiver(start[0], start[1], start[2],
              direction[0], direction[1], direction[2],
              color=colors[i], arrow_length_ratio=0.2, linewidth=0.6)

# 标签与标题
ax.set_title("Server参数PCA降维轨迹（3D）")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.legend()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout()
plt.show()
