import os
import torch
import random
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from itertools import chain
from system.utils.model import KGEModel
from system.utils.dataloader import TrainDataset, TestDataset

class Client:
    def __init__(self, seq, args):
        """
        初始化客户端，读取本地数据路径、实体/关系数量、并加载数据集。
        :param seq: 客户端编号
        :param args: 参数配置对象
        """
        self.name = seq
        self.args = deepcopy(args)
        self.local_file_dir = os.path.join(args.local_file_dir, str(seq))
        if not os.path.exists(self.local_file_dir):
            raise ValueError(f"local_file_dir {self.local_file_dir} 不存在。")

        self.nrelation = self.load_relations()
        self.nentity = self.load_entities()
        self.load_dataset()

    def load_entities(self):
        """
        读取本地实体字典文件，建立实体到编号的映射。
        :return: 实体总数
        """
        entity2id = {}
        with open(os.path.join(self.local_file_dir, "entities.dict"), "r", encoding="utf-8") as fin:
            for line in fin:
                eid, entity = line.strip().split()
                entity2id[entity] = int(eid)
        self.entity2id = entity2id
        return len(entity2id)

    def load_relations(self):
        """
        读取本地关系字典文件，建立关系到编号的映射。
        :return: 关系总数
        """
        relation2id = {}
        with open(os.path.join(self.local_file_dir, "relations.dict"), "r", encoding="utf-8") as fin:
            for line in fin:
                rid, relation = line.strip().split()
                relation2id[relation] = int(rid)
        self.relation2id = relation2id
        return len(relation2id)

    def read_triples(self, file_path):
        """
        从本地三元组文件中读取训练数据。
        :param file_path: 三元组文件路径
        :return: List of (h, r, t) 元组，已编码为数字
        """
        triples = []
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                h, r, t = line.strip().split()
                triples.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
        return triples

    def load_dataset(self):
        """
        加载本地的训练、验证、测试数据。
        """
        self.traindata = self.read_triples(os.path.join(self.local_file_dir, "train.txt"))
        self.validdata = self.read_triples(os.path.join(self.local_file_dir, "valid.txt"))
        self.testdata = self.read_triples(os.path.join(self.local_file_dir, "test.txt"))

    def init_model(self, init_entity_embedding=None):
        """
        初始化知识图谱嵌入模型，准备优化器和数据加载器。
        :param init_entity_embedding: 初始的实体嵌入，用于接受服务器的初始化参数
        """
        args = self.args
        self.kgeModel = KGEModel(
            args.model,
            self.nentity,
            self.nrelation,
            args.hidden_dim,
            args.gamma,
            epsilon=args.epsilon,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding,
            entity_embedding=init_entity_embedding,
            fed_mode=args.fed_mode,
            eta=args.eta,
        )
        if args.cuda:
            self.kgeModel = self.kgeModel.cuda()

        # 构建训练数据迭代器（头尾预测各一半）
        train_head = DataLoader(
            TrainDataset(self.traindata, self.nentity, self.nrelation, args.negative_sample_size, "head-batch"),
            batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=TrainDataset.collate_fn
        )
        train_tail = DataLoader(
            TrainDataset(self.traindata, self.nentity, self.nrelation, args.negative_sample_size, "tail-batch"),
            batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=TrainDataset.collate_fn
        )
        self.train_iterator = list(chain.from_iterable(zip(train_head, train_tail)))

        # 构建优化器
        self.optimizer = torch.optim.Adam([
            {"params": self.kgeModel.entity_embedding},
            {"params": self.kgeModel.relation_embedding}
        ], lr=args.learning_rate)

        # 构建验证/测试数据加载器
        all_true = self.traindata + self.validdata + self.testdata
        valid_head = DataLoader(TestDataset(self.validdata, all_true, self.nentity, self.nrelation, "head-batch"),
                                batch_size=args.test_batch_size, collate_fn=TestDataset.collate_fn)
        valid_tail = DataLoader(TestDataset(self.validdata, all_true, self.nentity, self.nrelation, "tail-batch"),
                                batch_size=args.test_batch_size, collate_fn=TestDataset.collate_fn)
        self.valid_dataset_list = [valid_head, valid_tail]

        test_head = DataLoader(TestDataset(self.testdata, all_true, self.nentity, self.nrelation, "head-batch"),
                               batch_size=args.test_batch_size, collate_fn=TestDataset.collate_fn)
        test_tail = DataLoader(TestDataset(self.testdata, all_true, self.nentity, self.nrelation, "tail-batch"),
                               batch_size=args.test_batch_size, collate_fn=TestDataset.collate_fn)
        self.test_dataset_list = [test_head, test_tail]

    def train(self):
        """
        本地训练函数。对当前客户端的模型进行多轮训练，并计算平均训练指标。
        """
        args = self.args
        training_logs = []

        for epoch in range(args.max_epoch):
            for pos_sample, neg_sample, weight, mode in self.train_iterator:
                train_log = self.kgeModel.train_step(
                    model=self.kgeModel,
                    optimizer=self.optimizer,
                    positive_sample=pos_sample,
                    negative_sample=neg_sample,
                    subsampling_weight=weight,
                    mode=mode,
                    args=args,
                    nodist=True  # 禁用互蒸馏
                )
                training_logs.append(train_log)

        # 统计训练日志
        metrics = {}
        if training_logs:
            for key in training_logs[0]:
                metrics[key] = sum(log[key] for log in training_logs) / len(training_logs)
                print(f"[Client {self.name}] {key}: {metrics[key]:.4f}")
        return metrics

    def valid(self):
        """
        在本地验证集上评估模型性能。
        :return: 验证指标
        """
        return self.kgeModel.test_step(self.kgeModel, self.valid_dataset_list, self.args)

    def test(self):
        """
        在本地测试集上评估模型性能。
        :return: 测试指标
        """
        return self.kgeModel.test_step(self.kgeModel, self.test_dataset_list, self.args)

    def get_entity_embedding(self):
        """
        获取当前客户端的实体嵌入，上传给服务器用于聚合。
        :return: 实体嵌入张量（Tensor）
        """
        return self.kgeModel.entity_embedding.detach().cpu()
