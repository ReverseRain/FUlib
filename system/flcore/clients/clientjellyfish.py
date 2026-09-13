import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flcore.clients.clientbase import Client
from utils.noise_utils import NoiseGenerator


class clientJellyfish(Client):
    """
    Jellyfish 联邦遗忘客户端。

    whole-client unlearning baseline：
    1) 将目标客户端的全部本地训练数据视为 D_f；
    2) 先统计整个目标客户端的类别分布；
    3) 对每个类别分别生成 class-conditioned proxy noise；
    4) 将各类别 proxy 合并为该客户端的 N_f。

    这里不显式使用原图内容、trigger、poison-aware loss 或 gradient matching，
    以保持 Jellyfish 代理数据生成方式作为基线。
    """

    def __init__(self, args, id, train_samples, test_samples, unlearning, **kwargs):
        super().__init__(args, id, train_samples, test_samples, unlearning, **kwargs)

        self.noise_steps = getattr(args, "noise_steps", 200)
        self.noise_lr = getattr(args, "noise_lr", 0.1)

        self.proxy_noises = None
        self.proxy_labels = None
        self.retain_noises = None
        self.retain_labels = None

    def train(self):
        """标准联邦本地训练阶段。"""
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs

        if self.train_slow:
            if max_local_epochs > 1:
                max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
            else:
                max_local_epochs = 1

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
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

        if self.learning_rate_decay:
            if hasattr(self, "poison_start_time"):
                if self.train_time_cost["num_rounds"] < self.poison_start_time:
                    self.learning_rate_scheduler.step()
            else:
                self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    @staticmethod
    def _extract_tensor_input(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    def _scan_train_loader(self, trainloader=None):
        """
        扫描完整本地训练集，返回：
            class_counts: {class_id: num_samples}
            img_size: (C, H, W)

        这里只读取标签和输入尺寸，不把真实图像内容用于 proxy optimization。
        """
        if trainloader is None:
            trainloader = self.load_train_data()

        class_counts = {}
        img_size = None
        total_samples = 0

        for x, y in trainloader:
            x_tensor = self._extract_tensor_input(x)

            if img_size is None:
                if x_tensor.ndim < 2:
                    raise RuntimeError(
                        f"Client {self.id}: invalid input shape {tuple(x_tensor.shape)}."
                    )
                img_size = tuple(x_tensor.shape[1:])

            y_cpu = y.detach().cpu().view(-1)
            for label in y_cpu:
                cls = int(label.item())
                class_counts[cls] = class_counts.get(cls, 0) + 1
                total_samples += 1

        if total_samples == 0:
            raise RuntimeError(f"Client {self.id}: training dataset is empty.")

        if img_size is None:
            raise RuntimeError(f"Client {self.id}: failed to infer image size.")

        return class_counts, img_size

    def generate_noise(self, global_model, steps=None, lr=None):
        """
        Jellyfish 阶段①：为整个待遗忘客户端生成代理遗忘数据集 N_f。

        对完整 D_f 先统计各类别样本数 n_c，再对每个类别 c 独立生成：
            N_{f,c} = argmin_N CE(M(N), c)
            |N_{f,c}| = n_c

        最终：
            N_f = union_c N_{f,c}

        返回 list[tensor] 以兼容现有 server 的 aggregate_client_noises()。
        """
        if steps is None:
            steps = self.noise_steps
        if lr is None:
            lr = self.noise_lr

        trainloader = self.load_train_data()
        class_counts, img_size = self._scan_train_loader(trainloader)

        print(
            f"[Client {self.id}] Forget-set class distribution: "
            f"{dict(sorted(class_counts.items()))}"
        )

        frozen_model = copy.deepcopy(global_model)
        noise_gen = NoiseGenerator(
            model=frozen_model,
            device=self.device,
            num_classes=self.num_classes,
            img_size=img_size,
        )

        noises, labels = noise_gen.generate_with_class_distribution(
            class_counts=class_counts,
            steps=steps,
            lr=lr,
        )

        self.proxy_noises = [noises]
        self.proxy_labels = [labels]

        print(
            f"[Client {self.id}] Proxy dataset generated: "
            f"{labels.numel()} samples, {len(class_counts)} classes."
        )

        return self.proxy_noises, self.proxy_labels

    def generate_noise_by_class(self, global_model, class_counts, steps=None, lr=None):
        """按外部给定的类别分布生成 proxy，主要用于辅助实验/调试。"""
        if steps is None:
            steps = self.noise_steps
        if lr is None:
            lr = self.noise_lr

        class_counts = {
            int(k): int(v)
            for k, v in class_counts.items()
            if int(v) > 0
        }
        if not class_counts:
            raise ValueError("class_counts contains no positive sample counts.")

        frozen_model = copy.deepcopy(global_model)
        noise_gen = NoiseGenerator(
            model=frozen_model,
            device=self.device,
            num_classes=self.num_classes,
            img_size=self._get_img_size(),
        )

        noises, labels = noise_gen.generate_with_class_distribution(
            class_counts=class_counts,
            steps=steps,
            lr=lr,
        )

        self.proxy_noises = [noises]
        self.proxy_labels = [labels]

        return self.proxy_noises, self.proxy_labels

    def generate_retention_noise(self, global_model, steps=100, lr=0.1):
        """
        Jellyfish 阶段④：为 remaining client 生成保留代理数据 N_r^i。

        whole-client unlearning 下，目标遗忘客户端不应调用本函数。
        """
        trainloader = self.load_train_data()
        class_counts, img_size = self._scan_train_loader(trainloader)

        print(
            f"[Client {self.id}] Retention-set class distribution: "
            f"{dict(sorted(class_counts.items()))}"
        )

        frozen_model = copy.deepcopy(global_model)
        noise_gen = NoiseGenerator(
            model=frozen_model,
            device=self.device,
            num_classes=self.num_classes,
            img_size=img_size,
        )

        noises, labels = noise_gen.generate_with_class_distribution(
            class_counts=class_counts,
            steps=steps,
            lr=lr,
        )

        self.retain_noises = noises
        self.retain_labels = labels

        return noises, labels

    def get_proxy_noise_loader(self, batch_size=None):
        """把当前客户端已生成的 proxy noise 打包为 DataLoader。"""
        if self.proxy_noises is None or self.proxy_labels is None:
            raise ValueError(
                "Proxy noise not generated yet. Call generate_noise() first."
            )

        if batch_size is None:
            batch_size = self.batch_size

        noises = torch.cat(
            [n.detach().cpu() for n in self.proxy_noises],
            dim=0,
        )
        labels = torch.cat(
            [l.detach().cpu().long() for l in self.proxy_labels],
            dim=0,
        )

        dataset = TensorDataset(noises, labels)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

    def get_class_distribution(self):
        """统计当前客户端完整训练集类别分布。"""
        trainloader = self.load_train_data()
        class_counts, _ = self._scan_train_loader(trainloader)
        return class_counts

    def _get_img_size(self):
        """从当前客户端训练数据推断图像输入尺寸。"""
        trainloader = self.load_train_data()
        for x, _ in trainloader:
            x_tensor = self._extract_tensor_input(x)
            return tuple(x_tensor.shape[1:])

        return (3, 32, 32)
