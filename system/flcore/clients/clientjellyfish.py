import os
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from utils.noise_utils import NoiseGenerator
import time


class clientJellyfish(Client):
    """
    Jellyfish 联邦遗忘客户端

    继承 Client 基类，复用:
      - set_parameters(), clone_model(): 模型参数操作
      - train(), test_metrics(), train_metrics(): 训练与评估
      - load_train_data(), load_test_data(): 数据加载
      - save_item(), load_item(): 模型保存/加载

    新增功能:
      - generate_noise(): 本地代理噪声生成（阶段①）
      - generate_retention_noise(): 保留数据代理噪声生成（阶段④）
    """

    def __init__(self, args, id, train_samples, test_samples, unlearning, **kwargs):
        # 显式调用 Client 基类的构造函数，确保正确挂载本地训练数据流和端侧标识
        super().__init__(args, id, train_samples, test_samples, unlearning, **kwargs)

        # 噪声生成参数
        self.noise_steps = getattr(args, 'noise_steps', 200)
        self.noise_lr = getattr(args, 'noise_lr', 0.1)

        # 代理噪声存储
        self.proxy_noises = None   # 遗忘数据的代理噪声
        self.proxy_labels = None   # 对应标签
        self.retain_noises = None  # 保留数据的代理噪声（阶段④用）
        self.retain_labels = None

    def train(self):
        """
        标准常规联邦训练 (从 clientAVG 完美移植，用于 Learning 预训练阶段)
        """
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        # 模拟慢速客户端节点（保持与基座算法实验扰动一致）
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 慢速节点物理睡眠延迟
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 学习率衰减控制流平衡
        if self.learning_rate_decay and self.train_time_cost['num_rounds'] < self.poison_start_time:
            self.learning_rate_scheduler.step()

        # 累加常规迭代轮数与全局时间开销计数器
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def generate_noise(self, global_model, steps=None, lr=None):
        """
        阶段①：在本地为遗忘数据生成代理噪声。

        流程:
        1. 锁定全局模型参数
        2. 初始化随机高斯噪声 N ~ N(0,1)
        3. 通过对抗优化，使模型将噪声高置信度识别为遗忘类别
        4. 返回优化后的噪声（不含任何原始隐私特征）

        Args:
            global_model: 当前全局模型（会被冻结，不会被修改）
            steps: 优化迭代轮数 (默认使用 self.noise_steps)
            lr: 噪声优化学习率 (默认使用 self.noise_lr)

        Returns:
            noises: list of tensor, 每个元素是一个批次的代理噪声
            labels: list of tensor, 每个元素是对应标签
        """
        if steps is None:
            steps = self.noise_steps
        if lr is None:
            lr = self.noise_lr

        # 创建噪声生成器 (会冻结模型参数)
        frozen_model = copy.deepcopy(global_model)
        noise_gen = NoiseGenerator(
            model=frozen_model,
            device=self.device,
            num_classes=self.num_classes,
            img_size=self._get_img_size()
        )

        print(f"  [Client {self.id}] Generating proxy noise for forget data...")

        # 根据训练数据的类别分布生成代理噪声
        noises, labels = noise_gen.generate_for_client(
            client_train_loader=self.train_loader,
            steps=steps,
            lr=lr
        )

        self.proxy_noises = noises
        self.proxy_labels = labels

        return noises, labels

    def generate_noise_by_class(self, global_model, class_counts, steps=None, lr=None):
        """
        按类别分布生成代理噪声（更精确的控制）。

        Args:
            global_model: 冻结的全局模型
            class_counts: dict, {class_id: num_samples}
            steps: 优化轮数
            lr: 学习率

        Returns:
            noises: list of tensor, 外层包裹一层列表以兼容服务端的聚合管道
            labels: list of tensor, 外层包裹一层列表以兼容服务端的聚合管道
        """
        if steps is None:
            steps = self.noise_steps
        if lr is None:
            lr = self.noise_lr

        frozen_model = copy.deepcopy(global_model)
        noise_gen = NoiseGenerator(
            model=frozen_model,
            device=self.device,
            num_classes=self.num_classes,
            img_size=self._get_img_size()
        )

        print(f"  [Client {self.id}] Generating proxy noise by class distribution...")

        noises, labels = noise_gen.generate_with_class_distribution(
            class_counts=class_counts,
            steps=steps,
            lr=lr
        )

        # 统一格式：在外层包裹一个单元素列表，确保服务器做 Union 遍历时格式恒等
        self.proxy_noises = [noises]
        self.proxy_labels = [labels]

        return [noises], [labels]

    def generate_retention_noise(self, global_model, num_samples=None, steps=100, lr=0.1):
        """
        阶段④：为保留数据生成代理噪声（模型修复用）。

        与遗忘噪声类似，但针对保留数据的类别分布。
        客户端在本地评估后，如果精度下降超过阈值 δ，
        则生成保留代理噪声提交给服务器进行修复。

        Args:
            global_model: 当前全局模型（冻结）
            num_samples: 生成样本数 (默认等于训练数据量)
            steps: 优化轮数 (修复用的噪声可以少迭代一些)
            lr: 学习率

        Returns:
            noises: tensor, 保留数据的代理噪声
            labels: tensor, 对应标签
        """
        if num_samples is None:
            num_samples = len(self.train_loader.dataset)

        frozen_model = copy.deepcopy(global_model)
        noise_gen = NoiseGenerator(
            model=frozen_model,
            device=self.device,
            num_classes=self.num_classes,
            img_size=self._get_img_size()
        )

        # 统计训练数据的类别分布
        class_counts = {}
        for _, y in self.train_loader:
            for label in y:
                cls = label.item()
                class_counts[cls] = class_counts.get(cls, 0) + 1

        # 按类别分布生成保留代理噪声
        print(f"  [Client {self.id}] Generating retention noise for model repair...")
        noises, labels = noise_gen.generate_with_class_distribution(
            class_counts=class_counts,
            steps=steps,
            lr=lr
        )

        self.retain_noises = noises
        self.retain_labels = labels

        return noises, labels

    def get_proxy_noise_loader(self, batch_size=None):
        """
        将代理噪声打包为 DataLoader，供服务端遗忘训练使用。

        Returns:
            DataLoader 对象
        """
        if self.proxy_noises is None:
            raise ValueError("Proxy noise not generated yet. Call generate_noise() first.")

        if batch_size is None:
            batch_size = self.batch_size

        flat_noises = []
        flat_labels = []
        for n, l in zip(self.proxy_noises, self.proxy_labels):
            flat_noises.append(n)
            flat_labels.append(l)

        dataset = list(zip(
            torch.cat(flat_noises, dim=0),
            torch.cat(flat_labels, dim=0)
        ))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def get_class_distribution(self):
        """
        统计遗忘客户端训练数据的类别分布。

        Returns:
            class_counts: dict, {class_id: num_samples}
        """
        class_counts = {}
        for _, y in self.train_loader:
            for label in y:
                cls = label.item()
                class_counts[cls] = class_counts.get(cls, 0) + 1
        return class_counts

    def _get_img_size(self):
        """从训练数据中推断图像尺寸"""
        for x, _ in self.train_loader:
            if isinstance(x, list):
                x = x[0]
            return x.shape[1:]  # (C, H, W)
        return (3, 32, 32)  # 默认 CIFAR 尺寸