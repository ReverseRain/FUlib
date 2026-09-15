import os
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
import time
import numpy as np

from utils.noise_utils import NoiseGenerator


class clientJellyfish(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 这里定义生成噪声的步数与学习率
        self.noise_steps = getattr(args, "noise_steps", 200)
        self.noise_lr = getattr(args, "noise_lr", 0.1)

        # Stage 4 Repair 的 N_r 生成超参数。
        # 论文定义了 E_re / mu_re 以及 error-minimization noise，
        # 但可见实验段没有给出 repair-noise 的固定数值，因此保持可配置。
        self.repair_noise_steps = getattr(
            args,
            "repair_noise_steps",
            100
        )
        self.repair_noise_lr = getattr(
            args,
            "repair_noise_lr",
            0.1
        )

        # 这里定义proxy data的本体和label
        self.proxy_noises = None
        self.proxy_labels = None
        self.retain_noises = None
        self.retain_labels = None

    @staticmethod
    def _extract_tensor_input(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    def _get_img_size(self):
        """从当前客户端训练数据推断图像输入尺寸。"""
        trainloader = self.load_train_data()
        for x, _ in trainloader:
            x_tensor = self._extract_tensor_input(x)
            return tuple(x_tensor.shape[1:])

        return (3, 32, 32)


    def train(self):
        """
        客户端本地标准训练逻辑 (FedAvg 本地更新)
        """
        trainloader = self.train_loader
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if not self.unlearning:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def generate_proxy_noise(self, global_model):
        """
            Jellyfish Step1: Error-Minimization Noise
            输入:
            client的本地数据
            输出:
            proxy noise:N_f以及标签:Y_f
            方法：
            使用Noise_generator中的 generate_for_client 方法为一个client中的dataloader生成代理噪声数据集
        """
        print("dataset图像的size是 ", self._get_img_size())
        noise_generator = NoiseGenerator(
            model = global_model,
            device = self.device,
            num_classes=self.num_classes,
            img_size = self._get_img_size(),
        )
        noises, labels = noise_generator.generate_for_client(
            client_train_loader=self.train_loader,
            steps = self.noise_steps,
            lr = self.noise_lr,
        )
        self.proxy_noises = noises
        self.proxy_labels = labels
        return noises, labels


    def evaluate_local_accuracy(self, model=None):
        """
        Stage 4 Eq.(23):
        评估当前全局模型在本客户端 local remaining test data 上的 accuracy。

        这里只返回 0~1 accuracy，不修改模型参数。
        """
        if model is None:
            model = self.model

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                if isinstance(x, (list, tuple)):
                    x = [
                        item.to(self.device)
                        for item in x
                    ]
                else:
                    x = x.to(self.device)

                y = y.to(self.device)

                logits = model(x)
                pred = torch.argmax(
                    logits,
                    dim=1
                )

                correct += int(
                    (pred == y).sum().item()
                )
                total += int(
                    y.numel()
                )

        return (
            correct / max(total, 1)
        )

    def generate_repair_noise(self, global_model):
        """
        Jellyfish Stage 4 / Section 4.5 / Eq.(24)

        使用客户端 remaining training data D_r^i 的标签分布，
        通过 error-minimization noise 生成 repair proxy N_r^i。

        非常重要：
        NoiseGenerator 在初始化时会 freeze 它持有的 model。
        因此这里必须对 Stage3 后 global_model 做 deepcopy，
        不能直接把 server 的可训练 global_model 交给 NoiseGenerator，
        否则会把真正待 Repair 的模型 requires_grad 关闭。

        Returns:
            noises:
                list[Tensor]，每个 batch 对应 N_r^i 的一部分
            labels:
                list[Tensor]，对应 y_r^i
            remaining_size:
                本次实际生成的 proxy 样本总数；
                用于 server 验证 |D_r^i| 权重。
        """
        print(
            f"[Stage4] Client {self.id}: "
            f"generating remaining proxy N_r, "
            f"img_size={self._get_img_size()}"
        )

        repair_reference_model = copy.deepcopy(
            global_model
        ).to(
            self.device
        )

        repair_reference_model.eval()

        noise_generator = NoiseGenerator(
            model=repair_reference_model,
            device=self.device,
            num_classes=self.num_classes,
            img_size=self._get_img_size(),
        )

        # 使用新的完整本地训练 DataLoader 读取 remaining data，
        # 避免复用某些工程中可能带 drop_last 的训练 loader。
        repair_train_loader = self.load_train_data()

        noises, labels = (
            noise_generator.generate_for_client(
                client_train_loader=repair_train_loader,
                steps=self.repair_noise_steps,
                lr=self.repair_noise_lr,
            )
        )

        self.retain_noises = noises
        self.retain_labels = labels

        proxy_count = sum(
            int(batch_label.numel())
            for batch_label in labels
        )

        # 论文聚合权重使用 |D_r^i|。
        # Client 基类通常保存 train_samples；若当前工程没有该字段，
        # 才退化为本次实际生成的 proxy 数量。
        remaining_size = int(
            getattr(
                self,
                "train_samples",
                proxy_count
            )
        )

        print(
            f"[Stage4] Client {self.id}: "
            f"|D_r|={remaining_size}, "
            f"N_r samples={proxy_count}"
        )

        return (
            noises,
            labels,
            remaining_size
        )

















































