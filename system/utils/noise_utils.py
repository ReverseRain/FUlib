import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NoiseGenerator:
    """
    Jellyfish Error-Minimization Proxy Noise Generator.

    核心流程：
        1) N ~ Normal(0, 1)
        2) 固定全局模型 M
        3) 只优化输入噪声 N
        4) 对目标类别 c 最小化 CE(M(N), c)

    当前版本用于 Jellyfish baseline：
        - 不显式使用原始图像内容
        - 不显式使用 backdoor trigger
        - 不加入 poison-aware loss
        - 不做 gradient matching
        - 每个类别分别生成 class-conditioned proxy
    """

    def __init__(
        self,
        model,
        device,
        num_classes,
        img_size=(3, 32, 32),
        chunk_size=128,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = int(num_classes)
        self.img_size = tuple(img_size)
        self.chunk_size = int(chunk_size)

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        self._freeze_model()

    def _freeze_model(self):
        """
        冻结模型参数，并保持 eval 模式，避免 proxy noise 更新 BatchNorm
        running_mean / running_var。
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def generate_for_class(
        self,
        target_class,
        num_samples,
        steps=200,
        lr=0.1,
    ):
        """
        为指定类别 c 生成一批 proxy noise。

        数学目标：
            N_c^* = argmin_N CE(M(N), c)

        Args:
            target_class: 目标类别
            num_samples: 当前 chunk 需要生成的 proxy 数量
            steps: 输入噪声优化步数
            lr: 输入噪声优化学习率

        Returns:
            noise:  [num_samples, C, H, W]
            labels: [num_samples]
        """
        target_class = int(target_class)
        num_samples = int(num_samples)
        steps = int(steps)
        lr = float(lr)

        if target_class < 0 or target_class >= self.num_classes:
            raise ValueError(
                f"target_class={target_class} out of range "
                f"[0, {self.num_classes - 1}]"
            )
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if lr <= 0:
            raise ValueError("lr must be positive.")

        C, H, W = self.img_size

        # 1. Gaussian initialization: N ~ N(0,1)
        noise = torch.randn(
            num_samples,
            C,
            H,
            W,
            device=self.device,
            requires_grad=True,
        )

        labels = torch.full(
            (num_samples,),
            fill_value=target_class,
            dtype=torch.long,
            device=self.device,
        )

        # 2. 使用显式梯度下降形式，贴近 Jellyfish Algorithm 2
        optimizer = torch.optim.SGD(
            [noise],
            lr=lr,
        )
        criterion = nn.CrossEntropyLoss()

        # 3. 只优化 noise，模型参数保持冻结
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)

            logits = self.model(noise)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # 诊断信息：观察 proxy 是否逐渐被模型识别为目标类别
            if (
                step == 0
                or (step + 1) % 50 == 0
                or (step + 1) == steps
            ):
                with torch.no_grad():
                    logits_now = self.model(noise)
                    prob_now = torch.softmax(logits_now, dim=1)

                    target_acc = (
                        logits_now.argmax(dim=1) == labels
                    ).float().mean().item()

                    target_conf = (
                        prob_now[:, target_class]
                    ).mean().item()

                print(
                    f"      [NoiseGen] class={target_class} "
                    f"step={step + 1}/{steps} "
                    f"loss={loss.item():.6f} "
                    f"target_acc={target_acc:.4f} "
                    f"target_conf={target_conf:.4f}"
                )

        return noise.detach(), labels.detach()

    def generate_with_class_distribution(
        self,
        class_counts,
        steps=200,
        lr=0.1,
        chunk_size=None,
    ):
        """
        按整个客户端的类别分布生成 class-wise proxy dataset。

        对每个类别 c：
            |N_c| = class_counts[c]
            N_c = argmin_N CE(M(N), c)

        最后：
            N_f = union_c N_c

        为降低显存峰值，每个类别按 chunk_size 分块优化。
        返回结果统一放到 CPU；server 训练时再搬到 GPU。
        """
        if not class_counts:
            raise ValueError("class_counts is empty.")

        if chunk_size is None:
            chunk_size = self.chunk_size

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        normalized_counts = {
            int(cls): int(count)
            for cls, count in class_counts.items()
            if int(count) > 0
        }

        if not normalized_counts:
            raise ValueError(
                "class_counts contains no positive sample counts."
            )

        all_noises = []
        all_labels = []

        # 固定 class 顺序便于复现；最后再统一 shuffle
        for cls in sorted(normalized_counts.keys()):
            count = normalized_counts[cls]

            if cls < 0 or cls >= self.num_classes:
                raise ValueError(
                    f"Class id {cls} is invalid for "
                    f"num_classes={self.num_classes}."
                )

            print(
                f"    [NoiseGen] class={cls}, "
                f"proxy_samples={count}"
            )

            remaining = count

            while remaining > 0:
                current_size = min(chunk_size, remaining)

                noise, labels = self.generate_for_class(
                    target_class=cls,
                    num_samples=current_size,
                    steps=steps,
                    lr=lr,
                )

                # 长期缓存放 CPU，降低显存压力
                all_noises.append(noise.detach().cpu())
                all_labels.append(labels.detach().cpu().long())

                remaining -= current_size

                del noise
                del labels

        if len(all_noises) == 0:
            raise RuntimeError(
                "No proxy samples were generated."
            )

        all_noises = torch.cat(all_noises, dim=0)
        all_labels = torch.cat(all_labels, dim=0).long()

        # 避免 [0,0,...,1,1,...] 的强类别块结构
        total_samples = all_labels.numel()
        permutation = torch.randperm(
            total_samples,
            device=all_labels.device,
        )

        all_noises = all_noises[permutation]
        all_labels = all_labels[permutation]

        return all_noises, all_labels


def aggregate_client_noises(
    client_noises_list,
    client_labels_list,
):
    """
    合并一个或多个待遗忘客户端上传的 proxy dataset。

    当前 whole-client baseline 若只遗忘一个 client，
    简单 concat + shuffle 即可。
    """
    aggregated_noises = []
    aggregated_labels = []

    for noises, labels in zip(
        client_noises_list,
        client_labels_list,
    ):
        if len(noises) != len(labels):
            raise ValueError(
                "Mismatch between noise chunks and label chunks."
            )

        for noise_tensor, label_tensor in zip(noises, labels):
            if noise_tensor.size(0) != label_tensor.size(0):
                raise ValueError(
                    "Noise/label sample count mismatch: "
                    f"{noise_tensor.size(0)} vs {label_tensor.size(0)}"
                )

            aggregated_noises.append(
                noise_tensor.detach().cpu()
            )
            aggregated_labels.append(
                label_tensor.detach().cpu().long()
            )

    if len(aggregated_noises) == 0:
        raise ValueError(
            "[Server Error] No proxy noise received from forget clients."
        )

    aggregated_noises = torch.cat(
        aggregated_noises,
        dim=0,
    )
    aggregated_labels = torch.cat(
        aggregated_labels,
        dim=0,
    ).long()

    total_samples = aggregated_labels.numel()
    permutation = torch.randperm(
        total_samples,
        device=aggregated_labels.device,
    )

    aggregated_noises = aggregated_noises[permutation]
    aggregated_labels = aggregated_labels[permutation]

    return aggregated_noises, aggregated_labels


def create_noise_dataloader(
    noises,
    labels,
    batch_size,
    shuffle=True,
):
    """
    将聚合后的 proxy dataset 转为 DataLoader。

    proxy tensor 默认位于 CPU；
    server 训练循环里再通过 x.to(device), y.to(device) 搬到 GPU。
    """
    if noises.size(0) != labels.size(0):
        raise ValueError(
            "Noises and labels have different sample counts."
        )

    if noises.size(0) == 0:
        raise ValueError(
            "Cannot create DataLoader from an empty proxy dataset."
        )

    dataset = TensorDataset(
        noises,
        labels.long(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def save_proxy_noise(
    noises,
    labels,
    save_path,
):
    """保存 proxy dataset，统一存为 CPU tensor。"""
    torch.save(
        {
            "noises": noises.detach().cpu(),
            "labels": labels.detach().cpu().long(),
        },
        save_path,
    )


def load_proxy_noise(
    load_path,
    device="cpu",
):
    """加载 proxy dataset。"""
    data = torch.load(
        load_path,
        map_location=device,
    )

    noises = data["noises"]
    labels = data["labels"].long()

    return noises, labels
