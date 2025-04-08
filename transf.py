import pickle
import numpy as np


def convert_tuple_to_dict(input_path, output_path, num_clients=100):
    """
    将存储在 input_path 的 tuple 数据文件转换为字典，并均匀分配到 num_clients 个客户端，
    每个客户端的数据保持二维结构 (n_samples_per_client, 10)。

    :param input_path: 原始的 tuple 数据文件路径
    :param output_path: 转换后的 dict 数据文件保存路径
    :param num_clients: 需要分配的客户端数量（默认 100）
    """
    # 加载原始 tuple 文件
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # 确保数据格式正确
    if isinstance(data, tuple) and len(data) == 3:
        full_data, labels, other_info = data

        # 检查数据形状是否符合预期
        if not isinstance(full_data, np.ndarray) or len(full_data.shape) != 2:
            raise ValueError("full_data 必须是二维 numpy 数组")

        total_samples, n_features = full_data.shape
        print(f"🔍 原始数据形状: {full_data.shape}")
        print(f"🔍 特征维度: {n_features}")

        # 计算每个客户端分配的样本数
        samples_per_client = total_samples // num_clients
        remainder = total_samples % num_clients

        # 初始化客户端数据字典
        client_data = {}
        start_idx = 0

        for i in range(num_clients):
            # 计算当前客户端的样本数（前remainder个客户端多分1个样本）
            end_idx = start_idx + samples_per_client
            client_data[f'client_{i}'] = full_data[start_idx:end_idx]
            start_idx = end_idx

        # 验证分配结果
        for i in range(min(3, num_clients)):  # 打印前3个客户端的信息
            print(f"  client_{i} 数据形状: {client_data[f'client_{i}'].shape}")

        # 构造最终的字典
        data_dict = {
            'full_data': full_data,  # 原始完整数据
            'true_label': np.array(labels),
            'num_clusters': 4,
            'kmeans_loss': other_info.get('kmeans_loss', None) if isinstance(other_info, dict) else None,
            **client_data  # 分配后的客户端数据
        }

        # 保存转换后的字典
        with open(output_path, 'wb') as f_out:
            pickle.dump(data_dict, f_out)

        print(f"✅ 数据转换完成，并保存到: {output_path}")
        print(f"💡 共 {num_clients} 个客户端，每个客户端约 {samples_per_client} 个样本")
    else:
        print("❌ Error: 加载的数据格式不正确，必须是包含 3 个元素的 tuple。")


if __name__ == '__main__':
    input_file = r'D:\github\mufc\data\covtype_multiclass1.pkl'  # 原始 tuple 文件
    output_file = r'D:\github\mufc\covtype_multiclass.pkl'  # 输出字典文件
    convert_tuple_to_dict(input_file, output_file, num_clients=100)