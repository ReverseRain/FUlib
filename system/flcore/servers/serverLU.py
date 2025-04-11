import os
import torch
import numpy as np
from copy import deepcopy


class Server:
    def __init__(self, args):
        """
        初始化 Server 类，读取所有客户端的实体信息并构建全局实体集合。
        :param args: 参数对象
        """
        self.args = deepcopy(args)
        self.nentity = self.load_entities()  # 计算全局实体数
        self.global_entity_embedding = None  # 全局实体嵌入矩阵
        self.client_entities_mapping = {}    # 每个客户端实体在全局中的索引位置

    def load_entities(self):
        """
        读取所有客户端的实体文件，构建全局实体集合并记录每个客户端的实体索引映射。
        :return: 全局实体数量
        """
        local_file_dir = self.args.local_file_dir
        client_entities_dict = {}

        # 收集每个客户端的实体名称
        for client_dir in os.listdir(local_file_dir):
            client_path = os.path.join(local_file_dir, client_dir)
            if not os.path.isdir(client_path): continue

            client_seq = int(client_dir)
            client_entities = []
            with open(os.path.join(client_path, "entities.dict"), "r", encoding="utf-8") as f:
                for line in f:
                    _, entity = line.strip().split()
                    client_entities.append(entity)
            client_entities_dict[client_seq] = client_entities

        # 构建全局实体集合（去重）
        all_entities = list(set(entity for ents in client_entities_dict.values() for entity in ents))
        all_entities.sort()
        self.all_entities = all_entities

        # 构建客户端实体到全局实体索引的映射
        client_entities_mapping = {}
        for cid, entity_list in client_entities_dict.items():
            client_entities_mapping[cid] = [all_entities.index(ent) for ent in entity_list]

        self.client_entities_mapping = client_entities_mapping
        return len(all_entities)

    def generate_global_embedding(self):
        """
        初始化全局实体嵌入矩阵（随机初始化）。
        """
        args = self.args
        dim = args.hidden_dim * 2 if args.double_entity_embedding else args.hidden_dim
        embed_range = (args.gamma + args.epsilon) / dim

        self.global_entity_embedding = torch.empty(self.nentity, dim)
        torch.nn.init.uniform_(self.global_entity_embedding, -embed_range, embed_range)

    def assign_embedding_to_clients(self):
        """
        为每个客户端分配其需要的实体嵌入部分（从全局嵌入中切片）。
        :return: dict -> {client_id: Tensor} 的实体嵌入映射
        """
        embedding_dict = {}
        for cid, indices in self.client_entities_mapping.items():
            embedding_dict[cid] = self.global_entity_embedding[indices].clone()
        return embedding_dict

    def aggregate_embedding(self, client_embedding_dict):
        """
        从所有客户端接收到的局部实体嵌入中，聚合为更新后的全局实体嵌入。
        支持加权/距离/相似度三种聚合策略。
        :param client_embedding_dict: {client_id: Tensor} 客户端上传的实体嵌入
        """
        args = self.args
        dim = args.hidden_dim * 2 if args.double_entity_embedding else args.hidden_dim
        new_global_embed = torch.zeros(self.nentity, dim)
        weights = torch.zeros(self.nentity)

        for cid, local_embedding in client_embedding_dict.items():
            global_indices = self.client_entities_mapping[cid]
            global_emb_part = self.global_entity_embedding[global_indices]

            if args.agg == "weighted":
                weights[global_indices] += 1
                new_global_embed[global_indices] += local_embedding.cpu()

            elif args.agg == "distance":
                # L2范数越小越相似，越重要
                dist = torch.norm(global_emb_part - local_embedding.cpu(), p=2, dim=1)
                importance = torch.exp(-dist)
                weights[global_indices] += importance
                new_global_embed[global_indices] += importance.unsqueeze(1) * local_embedding.cpu()

            elif args.agg == "similarity":
                sim = torch.cosine_similarity(global_emb_part, local_embedding.cpu(), dim=1)
                importance = torch.exp(sim)
                weights[global_indices] += importance
                new_global_embed[global_indices] += importance.unsqueeze(1) * local_embedding.cpu()
            else:
                raise ValueError(f"不支持的聚合方式：{args.agg}")

        # 避免除0，归一化
        weights = torch.where(weights > 0, weights, torch.ones_like(weights))
        self.global_entity_embedding = new_global_embed / weights.unsqueeze(1)

    def save_global_embedding(self, path):
        """
        将当前的全局实体嵌入保存到本地文件。
        :param path: 保存路径
        """
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, "global_entity_embedding.npy"), self.global_entity_embedding.numpy())
