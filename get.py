import pandas as pd
import pickle
import numpy as np

# 读取 pkl 文件
with open("D:/github/mufc/results/full_postures.pkl", "rb") as f:
    data = pickle.load(f)

# 计算最大长度（找到最长的数组）
max_length = max(value.shape[-1] if isinstance(value, np.ndarray) else 1 for value in data.values())

# 统一所有数据的长度
data_padded = {}
for key, value in data.items():
    if isinstance(value, np.ndarray):
        value = value.flatten()  # 转换为 1D

        if len(value) < max_length:
            # **填充 NaN 保持一致长度**
            padded = np.pad(value, (0, max_length - len(value)), mode='constant', constant_values=np.nan)
        else:
            padded = value  # 长度一致，直接用

        data_padded[key] = padded

# **转换为 DataFrame 并保存**
df = pd.DataFrame(data_padded)
df.to_csv("results_padded.csv", index=False)

print("✅ 数据已成功保存到 results_padded.csv")
