import pickle

# 读取原始 .p 文件
input_file = "kmeans_data_deletion_NeurIPS19_datasets_scaled.p"

with open(input_file, "rb") as f:
    data = pickle.load(f)

# 检查数据类型
if isinstance(data, dict):
    print("数据是字典，包含以下键：", data.keys())

    # 遍历字典，拆分并保存成多个 .pkl 文件
    for key, value in data.items():
        output_file = f"{key}.pkl"  # 例如 Celltype.pkl, Gaussian.pkl
        with open(output_file, "wb") as f_out:
            pickle.dump(value, f_out)
        print(f"已保存 {output_file}")

else:
    print("数据不是字典，可能需要手动检查内容！")
