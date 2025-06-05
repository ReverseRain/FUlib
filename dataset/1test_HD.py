# 修正 Hellinger 距离的计算函数
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 自定义 Hellinger 距离计算
def hellinger(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

# 设置参数
num_clients = 20
num_classes = 10
alpha = 0.1
num_trials = 1000

# 存储HD值
hd_values = []

# 模拟多轮计算HD
for _ in range(num_trials):
    # 为每个client生成一个标签分布（Dirichlet采样）
    client_distributions = np.random.dirichlet([alpha] * num_classes, size=num_clients)

    # 计算所有client对之间的HD
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            hd = hellinger(client_distributions[i], client_distributions[j])
            hd_values.append(hd)

# 计算均值和可视化
mean_hd = np.mean(hd_values)
std_hd = np.std(hd_values)

sns.histplot(hd_values, bins=30, kde=True)
plt.title(f"Hellinger Distance Distribution (α={alpha}, K={num_classes})\nAverage: {mean_hd:.4f}, standard deviation: {std_hd:.4f}")
plt.xlabel("Hellinger Distance")
plt.ylabel("Counts")
plt.grid(True)
plt.tight_layout()
plt.show()
