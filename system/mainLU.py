import os

import torch

from system.flcore.clients.clientLU import Client
from system.flcore.servers.serverLU import Server


class Args:
    def __init__(self):
        # 联邦设置
        self.client_num = 3
        self.local_file_dir = r"C:\Users\bdxly\Desktop\FUlib\system\data\Data\FB15k-237\C3FL"
        self.agg = "similarity"  # 可选: "weighted", "distance", "similarity"

        # 模型参数
        self.model = "TransE"  # 或 "ComplEx", "RotatE"
        self.double_entity_embedding = False
        self.double_relation_embedding = False
        self.hidden_dim = 256
        self.gamma = 12.0
        self.epsilon = 2.0

        # 如果使用 RotatE 或 ComplEx，需自动调整嵌入结构
        if self.model == "RotatE":
            self.double_entity_embedding = True
            self.negative_adversarial_sampling = True
        elif self.model == "ComplEx":
            self.double_entity_embedding = True
            self.double_relation_embedding = True

        # 训练参数（推荐强化本地训练）
        self.max_epoch = 30             # 每轮本地训练 epoch
        self.learning_rate = 5e-4       # 较快收敛
        self.negative_sample_size = 256
        self.batch_size = 512           # 如果显存够可以更大
        self.test_batch_size = 64

        # 联邦学习控制
        self.fed_mode = "FedLU"         # 或 FedAvg、FedDist
        self.eta = 0.0                  # FedLU 专用
        self.dist_mu = 1e-2
        self.co_dist = False

        # 设备控制
        self.cuda = torch.cuda.is_available()

        # 日志控制
        self.log_epoch = 5
        self.valid_epoch = 5
        self.early_stop_epoch = 15





def log_metrics(mode, round_idx, metrics_list):
    """
    记录并输出训练/验证/测试指标。
    """
    print(f"\n==== {mode.upper()} - Round {round_idx} ====")
    total = 0
    weighted_mrr = 0
    for cid, metrics in enumerate(metrics_list):
        print(f"Client {cid}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        if "MRR" in metrics and "n" in metrics:
            weighted_mrr += metrics["MRR"] * metrics["n"]
            total += metrics["n"]
    if total > 0:
        print(f"Weighted MRR: {weighted_mrr / total:.4f}")
    print("=================================\n")


def main():
    args = Args()

    # 初始化服务端和全局实体嵌入
    server = Server(args)
    server.generate_global_embedding()

    # 获取 client_id 列表（从实际路径扫描中得来）
    client_ids = list(server.client_entities_mapping.keys())
    clients = {cid: Client(cid, args) for cid in client_ids}

    # 向每个 client 分发初始实体嵌入
    init_embeddings = server.assign_embedding_to_clients()
    for cid in client_ids:
        clients[cid].init_model(init_entity_embedding=init_embeddings[cid])

    # 开始联邦训练
    total_rounds = 5
    for rnd in range(total_rounds):
        print(f"####### Round {rnd} #######")

        client_embedding_dict = {}
        round_metrics = []

        for cid in client_ids:
            print(f"--- Client {cid} Training ---")
            metrics = clients[cid].train()
            round_metrics.append(metrics)
            client_embedding_dict[cid] = clients[cid].get_entity_embedding()

        log_metrics("train", rnd, round_metrics)

        # 服务端聚合
        server.aggregate_embedding(client_embedding_dict)

        # 分发聚合后的实体嵌入
        updated_embeddings = server.assign_embedding_to_clients()
        for cid in client_ids:
            clients[cid].init_model(init_entity_embedding=updated_embeddings[cid])  # 或 update_model

    # 测试评估
    print("======== Final Evaluation ========")
    test_metrics = []
    for cid in client_ids:
        print(f"Client {cid} Test:")
        metrics = clients[cid].test()
        test_metrics.append(metrics)
    log_metrics("test", total_rounds, test_metrics)


if __name__ == "__main__":
    main()
