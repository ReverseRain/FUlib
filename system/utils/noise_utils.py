import torch
import torch.nn as nn
import copy
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class NoiseGenerator:
    """
    误差最小化噪声生成器 (Error-Minimization Noise Generator)

    核心思想：从随机高斯噪声出发，通过对抗优化使冻结的模型
    将噪声高置信度地识别为目标遗忘类别。
    生成的噪声不包含任何人类可读的原始隐私特征。

    论文公式: L_{N_j} = (1/n_B) * sum_{j=1}^{n_B} -y_f(j) * log(M(N_f(j)))
    """

    def __init__(self, model, device, num_classes, img_size=(3, 32, 32)):
        """
        Args:
            model: 冻结的全局模型（参数不会被更新）
            device: 计算设备
            num_classes: 类别数
            img_size: 输入图像尺寸 (C, H, W)
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.img_size = img_size

        # 冻结模型所有参数
        self._freeze_model()

    def _freeze_model(self):
        """冻结模型参数，确保噪声生成过程中模型不更新"""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def generate_for_class(self, target_class, num_samples, steps=200, lr=0.1):
        """
        为指定的目标遗忘类别生成代理噪声。

        优化过程: 初始化 N ~ N(0,1)，最小化 -y_f * log(M(N))
        使模型以高置信度将噪声识别为 target_class。

        Args:
            target_class: 目标遗忘类别 (int)
            num_samples: 需要生成的样本数量
            steps: 优化迭代轮数 (论文中的 E_no)
            lr: 噪声优化的学习率

        Returns:
            noise: 优化后的代理噪声 tensor, shape=(num_samples, C, H, W)
            labels: 对应的标签 tensor, shape=(num_samples,)
        """
        C, H, W = self.img_size

        # 步骤1: 初始化随机高斯噪声 N_f ~ N(0, 1)
        noise = torch.randn(num_samples, C, H, W, device=self.device, requires_grad=True)

        # 目标标签 (one-hot 的 argmax 形式)
        labels = torch.full((num_samples,), target_class,
                            dtype=torch.long, device=self.device)

        # 噪声优化器 (只优化噪声矩阵，模型参数已冻结)
        optimizer = torch.optim.Adam([noise], lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 步骤2: 对抗优化迭代
        for step in range(steps):
            optimizer.zero_grad()

            # 前向传播: M(N_f)
            output = self.model(noise)

            # 计算损失: -y_f * log(M(N_f))
            # CrossEntropyLoss 默认计算 -sum(y * log(p))
            # 我们最小化它，等价于让模型高置信度识别为目标类别
            loss = criterion(output, labels)

            # 反向传播更新噪声像素值
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    preds = torch.argmax(output, dim=1)
                    acc = (preds == target_class).float().mean().item()
                    conf = torch.softmax(output, dim=1).max(dim=1)[0].mean().item()
                print(f"    [NoiseGen] class={target_class}, step={step+1}/{steps}, "
                      f"loss={loss.item():.4f}, acc={acc:.4f}, conf={conf:.4f}")

        # detach 并返回
        noise = noise.detach()
        labels = labels.detach()

        return noise, labels

    def generate_for_client(self, client_train_loader, steps=200, lr=0.1):
        """
        为一个遗忘客户端的所有训练数据生成代理噪声。

        通过在 Batch 内部拆解并重新平铺组合，完美适配 Non-IID 多类混合场景，
        严格对齐论文公式 3 的矩阵对齐设计，且无需修改 generate_for_class。
        """
        all_noises = []
        all_labels = []

        # 遍历客户端的本地原始数据集 (对应论文 Algorithm 2, line 22)
        for batch_idx, (x, y) in enumerate(client_train_loader):
            if isinstance(x, list):
                x = x[0]
            x = x.to(self.device)
            y = y.to(self.device)

            batch_size = x.shape[0]

            # 1. 统计当前 Batch 中包含哪些独特的类别，以及每个类别出现的频次
            # 例如 y = [3, 3, 5, 3, 5] -> unique_classes = [3, 5], counts = [3, 2]
            unique_classes, counts = torch.unique(y, return_counts=True)

            # 用于临时存放当前 Batch 内部各类别优化好的噪声片段和对应的局部标签
            batch_noises_fragments = []
            batch_labels_fragments = []

            # 2. 针对当前 Batch 包含的每一个类别，分别调用你的 generate_for_class
            for cls, count in zip(unique_classes, counts):
                cls_item = cls.item()
                count_item = count.item()

                # 调用你原本的底层对抗优化函数，生成指定数量的单类噪声片段
                sub_noise, sub_labels = self.generate_for_class(
                    target_class=cls_item,
                    num_samples=count_item,
                    steps=steps,
                    lr=lr
                )
                batch_noises_fragments.append(sub_noise)
                batch_labels_fragments.append(sub_labels)

            # 3. 将各类的片段拼接成一个大 Batch (此时顺序可能是按类别聚集的：如 [3,3,3,5,5])
            flat_batch_noise = torch.cat(batch_noises_fragments, dim=0)
            flat_batch_label = torch.cat(batch_labels_fragments, dim=0)

            # 4. 【核心对齐步】为了严格模拟原数据的分布顺序，我们需要把拼好的噪声还原到原 y 的位置
            # 创建一个与原输入完全一致的空白张量槽位
            C, H, W = self.img_size
            aligned_noise = torch.zeros(batch_size, C, H, W, device=self.device)
            aligned_label = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            # 动态映射指针，按原标签 y 的空间索引把优化好的雪花点噪声填进去
            for cls in unique_classes:
                # 找出原标签中属于 cls 的所有索引位置
                orig_indices = (y == cls).nonzero(as_tuple=True)[0]
                # 抽出刚刚生成的对应类别的噪声片段
                frag_indices = (flat_batch_label == cls).nonzero(as_tuple=True)[0]

                aligned_noise[orig_indices] = flat_batch_noise[frag_indices]
                aligned_label[orig_indices] = flat_batch_label[frag_indices]

            # 5. 将完美恢复顺序、矩阵完全对齐的 Batch 数据追加到结果列表
            all_noises.append(aligned_noise.detach())
            all_labels.append(aligned_label.detach())

            print(f"  [NoiseGen] Batch {batch_idx + 1}/{len(client_train_loader)} 代理噪声制造完毕. "
                  f"包含类别: {unique_classes.cpu().tolist()}")

        return all_noises, all_labels

    def generate_with_class_distribution(self, class_counts, steps=200, lr=0.1):
        """
        根据指定的类别分布批量生成全局代理噪声数据集（严格对应论文 4.2 节多类遗忘场景）。

        通过集中式对抗优化合成各类别专属的“噪声替身”，并进行全局随机重洗（Shuffle），
        以完美模拟分布式网络中真实的混合 Batch 特征激活，并消除批次截断隐私残留隐患。

        Args:
            class_counts: dict, {class_id: num_samples} 各类别需要生成的样本数
            steps: 优化迭代轮数 (论文中的 E_no)
            lr: 学习率 (论文中的 \mu_{no})

        Returns:
            all_noises: 打乱后的全局代理噪声大张量, shape=(total_samples, C, H, W)
            all_labels: 对应的全局目标标签大张量, shape=(total_samples,)
        """
        all_noises = []
        all_labels = []

        # 1. 严格对照论文说明：针对发起请求的各个类别，独立在本地训练噪声矩阵 (对应第 14 页第 394 行)
        for cls, count in class_counts.items():
            if count <= 0:
                continue
            print(f"  [NoiseGen] 正在为类别 {cls} 对抗优化合成 {count} 个代理噪声样本...")

            # 调用底层单类优化流（公式 3 的核心实现）
            noise, labels = self.generate_for_class(
                target_class=cls,
                num_samples=count,
                steps=steps,
                lr=lr
            )
            all_noises.append(noise)
            all_labels.append(labels)

        if len(all_noises) == 0:
            return torch.tensor([]), torch.tensor([])

        # 2. 拼接所有类别的噪声片段，构建论文中统一的全局噪声矩阵集 N_f (对应公式 4 前后的文本定义)
        all_noises = torch.cat(all_noises, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        total_samples = all_noises.shape[0]

        # 3. 【核心优化：消除梯度偏置】
        # 由于拼接后的数据分布是强顺序性的（如[0,0..1,1..]），如果不打乱，在阶段③的Batch训练中会造成严重的梯度方向倾斜
        # 我们在内存中生成全局随机索引，将多类特征噪声彻底揉碎混合
        shuffle_indices = torch.randperm(total_samples, device=self.device)
        all_noises = all_noises[shuffle_indices]
        all_labels = all_labels[shuffle_indices]

        print(f"  [NoiseGen] 全局代理数据集 N_f 组装完毕. 总样本数: {total_samples}, "
              f"包含独特类别数: {len(class_counts)}")

        # 剥离计算图返回，作为安全的全局大矩阵
        return all_noises.detach(), all_labels.detach()


def aggregate_client_noises(client_noises_list, client_labels_list):
    """
    服务器端噪声聚合 (严格对应论文 4.2 节末尾说明及公式 4):

    L_{N_f} = (1/n_f) * sum_{i=1}^{n_f} (1/n_B^i) * sum_{j=1}^{n_B^i} -y_f^i(j) * log(M(N_f^i(j)))

    物理本质: 论文公式 4 的双重求和代表在样本空间上做 Union 拼接。
    严禁在像素级别乘以任何权重系数缩小噪声幅值，否则会导致对抗特征无法激活网络。

    Args:
        client_noises_list: list of list of noise tensors
            每个元素是一个客户端的所有批次噪声列表 [[batch_1], [batch_2]...]
        client_labels_list: list of list of label tensors
            对应各客户端上传的噪声目标标签列表

    Returns:
        aggregated_noises: 拼接后的全局代理噪声大张量, shape=(total_samples, C, H, W)
        aggregated_labels: 对应的一维全局目标标签大张量, shape=(total_samples,)
    """
    aggregated_noises = []
    aggregated_labels = []

    # 严格对齐论文第 14 页陈述: "received noise matrices are aggregated into a unified noise matrix set, denoted as N_f"
    # 服务器将所有接收到的客户端脱敏噪声矩阵及其对应的隐私标签，收集并组织成统一的并集集合
    for noises, labels in zip(client_noises_list, client_labels_list):
        for batch_noise, batch_label in zip(noises, labels):
            # 保持噪声矩阵的原始特征激活幅值，严禁乘以任何 weight 缩放系数
            aggregated_noises.append(batch_noise.detach().clone())
            aggregated_labels.append(batch_label.detach().clone())

    if len(aggregated_noises) == 0:
        raise ValueError("[Server Error] 接收到的遗忘客户端代理噪声为空，无法执行公式 4 聚合！")

    # 在样本维度 (dim=0) 上执行大拼接，完美为公式 4 的双重求和奠定数据结构基石
    aggregated_noises = torch.cat(aggregated_noises, dim=0)
    aggregated_labels = torch.cat(aggregated_labels, dim=0)

    # 为了消除客户端按顺序上传带来的类别局部同质化偏置 (例如一堆类0后面接一堆类1)，
    # 我们在服务器端内存中进行一次随机乱序重洗，使混合 Batch 特征更加平滑
    total_samples = aggregated_noises.shape[0]
    shuffle_indices = torch.randperm(total_samples)
    aggregated_noises = aggregated_noises[shuffle_indices]
    aggregated_labels = aggregated_labels[shuffle_indices]

    print(f"[Server Aggregation] 代理数据集 N_f 整合成功. 总样本数: {total_samples}")
    return aggregated_noises, aggregated_labels


def create_noise_dataloader(noises, labels, batch_size, shuffle=True):
    """
    将整合后的全局代理数据集打包为 DataLoader，供阶段 ③ 联合遗忘训练循环直接读取。

    Args:
        noises: tensor, shape=(N, C, H, W)
        labels: tensor, shape=(N,)
        batch_size: 联合遗忘时的分批大小
        shuffle: 是否打乱

    Returns:
        DataLoader 对象
    """
    dataset = TensorDataset(noises, labels)

    # 修改安全点：将 drop_last 改为 False。
    # 确保在零样本框架下，100% 的隐私数据替身都能被遗忘算法完整扫描到，防止尾部少数类发生隐私截断残留。
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def save_proxy_noise(noises, labels, save_path):
    """保存代理噪声到文件"""
    torch.save({'noises': noises, 'labels': labels}, save_path)
    print(f"[NoiseGen] Proxy noise saved to {save_path}, "
          f"shape={noises.shape}, num_classes={labels.unique().shape[0]}")


def load_proxy_noise(load_path, device='cpu'):
    """从文件加载代理噪声"""
    data = torch.load(load_path, map_location=device)
    print(f"[NoiseGen] Proxy noise loaded from {load_path}, "
          f"shape={data['noises'].shape}")
    return data['noises'], data['labels']
