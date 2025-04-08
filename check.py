import pickle
import numpy as np

def inspect_tuple(data, n_rows=3):
    print("\n=== Tuple Inspection ===")
    print("Size (total elements):", len(data))

    # 检查每个元素的形状（如果是数组/张量）
    for i, item in enumerate(data):
        if hasattr(item, 'shape'):  # 适用于 NumPy 数组、PyTorch 张量等
            print(f"Element {i} shape:", item.shape)
        elif isinstance(item, (list, tuple)):
            print(f"Element {i} length:", len(item))

    # 输出前 n_rows 行内容
    print(f"\nFirst {n_rows} elements:")
    for i in range(min(n_rows, len(data))):
        print(f"[{i}]: {data[i]}")


dataset = pickle.load(open("4celltypes_10pca.pkl", "rb"))
print(type(dataset))  # 查看 dataset 的类型
print("Available keys in dataset:", dataset.keys())
print(dataset["full_data"].shape)  # 查看 dataset 的内容
#inspect_tuple(dataset, n_rows=5)


#num_clusters=4
#kmeans_loss=188.08093814549414

#Size (total elements): 3
#Element 0 shape: (12009, 10)
#Element 1 shape: (12009,)