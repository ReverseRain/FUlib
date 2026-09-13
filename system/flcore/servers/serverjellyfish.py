import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from flcore.clients.clientjellyfish import clientJellyfish
from flcore.servers.serverbase import Server
from utils.attack_utils import attack, train_attack_model
from utils.noise_utils import aggregate_client_noises, create_noise_dataloader, save_proxy_noise


class Jellyfish(Server):
    """
    Jellyfish Server：面向 whole-client federated unlearning 的基线实现。

    本实验中的遗忘对象定义：
        D_f = 目标客户端（或多个目标客户端）的完整本地训练数据；
        D_r = 所有非目标客户端拥有的保留数据。

    与原论文 category-level 实验的区别：
        原论文主要验证类别级遗忘；当前实验把一个包含多个 CIFAR-10 类别
        的 poisoned client 整体视为 D_f。因此这里保持 Jellyfish 原有的
        class-conditioned proxy 生成与后续遗忘机制，用作 baseline，而不额外
        注入 trigger-aware、poison-aware 或 gradient-matching 信息。

    Server 端四个阶段：
        Stage 1  Proxy Noise Generation
                 目标客户端本地生成 N_f，服务器仅负责聚合。

        Stage 2  Knowledge Disentanglement
                 在最后卷积层上计算通道 L1 激活，对低重要性通道施加抑制。

        Stage 3  Joint Unlearning
                 L_unlearn = L_hard + mu_c L_confusion + mu_d L_distillation；
                 同时使用 Drift、Gradient Mask、Gradient Harmonization。

        Stage 4  Zero-shot Repair（可选）
                 只允许 remaining clients 生成 N_r，用于修复遗忘造成的 utility
                 损失；whole-client 删除场景中目标客户端不得参与 repair。

    论文符号在当前 FUlib 项目中的对应关系：
        omega_0   -> self.initial_model
        omega_t   -> load_model + send_models + warm_up 后的 self.global_model
        omega_ref -> self.original_model = deepcopy(omega_t)
        N_f       -> self.aggregated_noises / self.proxy_noise_loader
    """

    def __init__(self, args):
        super().__init__(args)

        # Algorithm 1 explicitly takes the initialized model omega_0 as input.
        # Keep it before load_model() replaces/loads the trained omega_t.
        self.initial_model = copy.deepcopy(self.global_model).to(self.device)
        self.initial_model.eval()
        for p in self.initial_model.parameters():
            p.requires_grad_(False)

        self.set_slow_clients()
        self.set_clients(clientJellyfish)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.unlearn_Budget = []

        self.original_model = None
        self.proxy_noise_loader = None
        self.aggregated_noises = None
        self.aggregated_labels = None

        # Paper-explicit hyperparameters: alpha=0.9, mu_c=mu_d=0.5.
        # Other defaults below are implementation choices because the paper does
        # not publish all of E_dis/E_un/Temp/pi/N_T/delta numerically.
        self.alpha_dis = getattr(args, "alpha", 0.9)
        self.dis_epochs = getattr(args, "dis_epochs", 5)
        self.dis_lr = getattr(args, "dis_lr", 1e-3)

        self.unlearn_epochs = getattr(args, "unlearn_epochs", 3)
        self.unlearn_lr = getattr(args, "unlearn_lr", getattr(args, "unlearn_rate", 5e-3))
        self.mu_c = getattr(args, "mu_c", 0.5)
        self.mu_d = getattr(args, "mu_d", 0.5)
        self.distill_temp = getattr(args, "distill_temp", 4.0)
        self.pi_mask = getattr(args, "pi_mask", 1e-3)
        self.num_bad_teachers = getattr(args, "num_teachers", 3)
        self.mask_microbatch = max(1, getattr(args, "mask_microbatch", 1))
        # Optional numerical safeguard. 0/None means disabled, matching Algorithm 1.
        self.max_update_norm = getattr(args, "max_update_norm", 0.0)

        self.delta_threshold = getattr(args, "delta_threshold", 0.05)
        self.repair_epochs = getattr(args, "repair_epochs", 2)
        self.repair_lr = getattr(args, "repair_lr", 5e-3)
        self.repair_noise_steps = getattr(args, "repair_noise_steps", 100)
        self.repair_noise_lr = getattr(args, "repair_noise_lr", getattr(args, "noise_lr", 0.1))
        self.disable_repair = getattr(args, "disable_repair", False)

    # ------------------------------------------------------------------
    # Standard FL training
    # ------------------------------------------------------------------
    def train(self):
        print("\n" + "=" * 50)
        print("Starting Learning Phase (Standard FedAvg Pre-training)...")
        print("=" * 50)

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.rs_test_acc:
            print("\nBest accuracy during Learning Phase:", max(self.rs_test_acc))

        # Keep your existing MIA pipeline, but evaluate the same target-client
        # group before and after unlearning.
        print("\n[MIA] Training attacker model for baseline evaluation...")
        self.attacker = train_attack_model(
            self.global_model, self.clients, self.num_classes, self.device
        )
        if self.unlearning_clients:
            pre, rec = attack(
                self.global_model,
                self.attacker,
                self.unlearning_clients,
                self.num_classes,
                self.device,
            )
            print(f"MIA target-client precision before unlearning = {pre:.4f}")
            print(f"MIA target-client recall before unlearning    = {rec:.4f}")

        self.save_results()
        self.save_global_model()
        print("[Learning Complete] Global model/attacker snapshot saved.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _freeze_bn_stats(model):
        """
        冻结 BatchNorm 的 running statistics，但不冻结普通可训练参数。

        原因：Stage 2/3 使用的是人工生成 proxy noise。如果直接 model.train()，
        BN 的 running_mean / running_var 会被代理噪声分布改写，产生额外的
        非论文目标更新。因此将 BN 层单独切到 eval()，同时卷积/全连接参数
        仍然可以正常反向传播。
        """
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    @staticmethod
    def _flatten_proxy(noises, labels):
        if torch.is_tensor(noises):
            noises = [noises]
        if torch.is_tensor(labels):
            labels = [labels]
        noises = list(noises)
        labels = list(labels)
        if len(noises) == 0:
            raise RuntimeError("Empty proxy-noise list.")
        return torch.cat(noises, dim=0), torch.cat(labels, dim=0)

    @staticmethod
    def _reset_classifier_head(teacher):
        """Reinitialize only the classifier head of an omega_0-based teacher."""
        head = getattr(teacher, "head", None)
        if head is None:
            return teacher
        for module in head.modules():
            if hasattr(module, "reset_parameters"):
                try:
                    module.reset_parameters()
                except TypeError:
                    pass
        return teacher

    def _build_bad_teachers(self):
        """
        Build the incompetent-teacher set used by Eqs. (13)-(16).

        Algorithm 1 passes omega_0 into the unlearning loss, while the text only
        specifies that M_bad must not have been exposed to D_f and that multiple
        teachers may be used.  We therefore anchor every teacher at omega_0 rather
        than at the trained/disentangled omega_t.  For N_T > 1, classifier heads are
        independently reset to provide the diversity requested by Eq. (16).
        """
        n_teachers = max(int(self.num_bad_teachers), 1)
        teachers = []
        for idx in range(n_teachers):
            teacher = copy.deepcopy(self.initial_model).to(self.device)
            if idx > 0:
                teacher = self._reset_classifier_head(teacher)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            teachers.append(teacher)
        return teachers

    def _global_gradient_harmonization(self, g_f, g_r):
        """
        论文 Eq.(20) 的全参数空间 Gradient Harmonization。

        先把所有层参数视为一个整体向量，计算：
            dot = g_f · g_r

        若 dot < 0，说明遗忘梯度与保留/漂移梯度方向冲突，则执行：
            g_f' = g_f - (g_r·g_f / ||g_r||^2) g_r
            G    = g_f' + g_r

        若不存在冲突，则直接 G = g_f + g_r。

        注意：这里不能逐层分别判断冲突，否则与论文中的全参数向量投影不一致。
        """
        dot = torch.zeros((), device=self.device)
        norm_r_sq = torch.zeros((), device=self.device)

        for name in g_f:
            dot += torch.sum(g_f[name] * g_r[name])
            norm_r_sq += torch.sum(g_r[name] * g_r[name])

        if dot.item() < 0.0 and norm_r_sq.item() > 1e-20:
            coeff = dot / norm_r_sq
            return {
                name: (g_f[name] - coeff * g_r[name]) + g_r[name]
                for name in g_f
            }, float(dot.item())

        return {name: g_f[name] + g_r[name] for name in g_f}, float(dot.item())

    def _clip_gradient_dict(self, grads, max_norm):
        if max_norm is None or max_norm <= 0:
            return grads
        norm_sq = torch.zeros((), device=self.device)
        for g in grads.values():
            norm_sq += torch.sum(g * g)
        total_norm = torch.sqrt(norm_sq + 1e-20)
        if total_norm.item() <= max_norm:
            return grads
        scale = max_norm / total_norm
        return {name: g * scale for name, g in grads.items()}

    def _client_accuracy(self, client, model):
        client.set_parameters(model)
        metrics = client.test_metrics()
        correct, n = metrics[0], metrics[1]
        return float(correct) / float(n) if n > 0 else 0.0

    # ------------------------------------------------------------------
    # Complete client-level unlearning flow
    # ------------------------------------------------------------------
    def unlearning(self):
        """
        Jellyfish 遗忘主流程控制中心。

        重要：下面从 load_model() 到 evaluate() 的 6 个步骤是当前 FUlib
        工程既有的模型恢复 / 同步 / BN warm-up 流程，按项目约束完整保留。

        因此，在本实现中把 warm_up() 之后的 global_model 定义为论文后续
        Jellyfish 阶段使用的遗忘前模型 omega_t，并将其深拷贝到
        self.original_model，后续有三个用途：
            1) Stage 1 代理遗忘集 N_f 的生成参考模型；
            2) Stage 3 Gradient Mask 的敏感度参考模型；
            3) Stage 3 Drift Loss 的参数参考 omega_ref。

        也就是说，在本项目中：
            omega_t := warm_up() 完成后的 global_model
            omega_ref := copy.deepcopy(omega_t)
        """
        if not self.unlearning_clients:
            raise RuntimeError(
                "Jellyfish client-level unlearning needs --unlearning_clients / -uc."
            )

        print("\n" + "=" * 62)
        print("Jellyfish whole-client unlearning")
        print("Target client ids:", [c.id for c in self.unlearning_clients])
        print("=" * 62)

        # ================================================================
        # [项目固定步骤 1] 从磁盘恢复 Learning 阶段保存的模型 / 攻击者等状态。
        # 这一步由 Server 基类实现，必须保留。
        # ================================================================
        self.load_model()

        # ================================================================
        # [项目固定步骤 2] 将刚恢复的最新全局模型同步给全部本地客户端。
        # 这保证 client.model 与 server.global_model 在 warm_up 前一致。
        # ================================================================
        self.send_models()

        # ================================================================
        # [项目固定步骤 3] FUlib 既有 BN warm-up。
        # 根据当前项目约束，这一步不能删除。
        # 其执行完成后的 global_model 被视为本实现中的 omega_t。
        # ================================================================
        self.warm_up()

        # ================================================================
        # [项目固定步骤 4] 切换到 eval 模式，锁定推理态。
        # 注意：后续 Stage 2/3 内部需要训练时，会局部切回 train()，并再次
        # 单独冻结 BN running statistics，避免 proxy noise 改写 BN buffer。
        # ================================================================
        self.global_model.eval()
        for client in self.clients:
            client.model.eval()

        # ================================================================
        # [项目固定步骤 5] 保存遗忘开始前参考模型。
        # 此处 original_model 即 Jellyfish 后续使用的 omega_t / omega_ref。
        # 它必须在 Stage 1、Stage 2、Stage 3 任何遗忘更新发生之前保存。
        # ================================================================
        self.original_model = copy.deepcopy(self.global_model)
        self.original_model.eval()

        # ================================================================
        # [项目固定步骤 6] 记录遗忘前 Utility。
        # ================================================================
        self.evaluate()

        # ================================================================
        # Stage 1: 代理遗忘数据集 N_f
        # 目标客户端只在本地读取自己的完整 D_f 的标签分布；每个类别单独
        # 生成 class-conditioned error-minimization noise，服务器只负责收集、
        # 拼接和打乱，不读取目标客户端原始图像。
        # ================================================================
        print("\n[Phase 1] Proxy-noise generation and aggregation")
        self.collect_proxy_noise()

        # ================================================================
        # Stage 2: Knowledge Disentanglement
        # 根据最后卷积层通道 L1 激活强度，选择低重要性通道并最小化其激活。
        # ================================================================
        if self.dis_epochs > 0:
            print("\n[Phase 2] Knowledge disentanglement")
            self.fit_knowledge_disentanglement()

        # ================================================================
        # Stage 3: Joint Unlearning
        # 包含 Hard / Confusion / Distillation 三项遗忘目标，以及 Drift、
        # Gradient Mask 和 Gradient Harmonization。
        # ================================================================
        print("\n[Phase 3] Joint unlearning")
        self.fit_joint_unlearning()

        # ================================================================
        # Stage 4: Zero-shot Repair（可选）
        # whole-client 场景下，目标遗忘客户端已经没有 retained data，
        # 因此 repair 只允许由非目标客户端生成 N_r。
        # ================================================================
        if not self.disable_repair:
            print("\n[Phase 4] Optional zero-shot repair")
            self.adaptive_model_repair()
        else:
            print("\n[Phase 4] Repair disabled")

        # ================================================================
        # 将最终遗忘模型重新同步给客户端，并执行最终评估。
        # ================================================================
        self.send_models()
        if hasattr(self, "send_models_target"):
            self.send_models_target()

        self.global_model.eval()
        for client in self.clients:
            client.model.eval()

        print("\n[Final] Global evaluation")
        self.evaluate()

        # ================================================================
        # MIA：如果 Learning 阶段已训练 attacker，则对同一个目标客户端集合
        # 做遗忘后评估，保证前后对比对象一致。
        # ================================================================
        if hasattr(self, "attacker") and self.attacker is not None:
            print("\n[MIA] Target-client group after unlearning")
            pre, rec = attack(
                self.global_model,
                self.attacker,
                self.unlearning_clients,
                self.num_classes,
                self.device,
            )
            print(f"MIA target-client precision after unlearning = {pre:.4f}")
            print(f"MIA target-client recall after unlearning    = {rec:.4f}")
            self.save_unlearning(pre)

    # ------------------------------------------------------------------
    # Stage 1：代理遗忘数据 N_f 的服务器聚合
    # ------------------------------------------------------------------
    def collect_proxy_noise(self):
        """
        收集目标客户端上传的 class-conditioned proxy，并组装服务器侧 N_f。

        职责边界：
            Client：统计完整 D_f 的类别分布，逐类别生成 proxy noise。
            Server：只检查数量、聚合、shuffle、构造 DataLoader、持久化。

        whole-client baseline 中若只有一个目标客户端，则：
            N_f = N_f^(target client)

        若以后扩展到多个目标客户端，则服务器会将各客户端 proxy 做 union。
        """
        if not self.unlearning_clients:
            raise RuntimeError("No target client was specified for unlearning.")
        if self.original_model is None:
            raise RuntimeError("omega_ref/omega_t must be snapshotted before Stage 1.")

        client_noises_list = []
        client_labels_list = []
        expected_total = 0

        for client in self.unlearning_clients:
            class_counts = client.get_class_distribution()
            expected = int(sum(class_counts.values()))
            expected_total += expected
            print(
                f"  client {client.id} D_f distribution = "
                f"{dict(sorted(class_counts.items()))}"
            )

            # Generate from the exact pre-unlearning omega_t/reference snapshot.
            noises, labels = client.generate_noise(
                global_model=self.original_model,
                steps=getattr(self.args, "noise_steps", 200),
                lr=getattr(self.args, "noise_lr", 0.1),
            )

            n_proxy = sum(int(x.shape[0]) for x in noises)
            n_label = sum(int(y.shape[0]) for y in labels)
            if n_proxy != n_label:
                raise RuntimeError(
                    f"Client {client.id}: proxy/label mismatch {n_proxy} vs {n_label}."
                )
            if n_proxy != expected:
                raise RuntimeError(
                    f"Client {client.id}: expected {expected} proxies from |D_f| "
                    f"but received {n_proxy}."
                )

            client_noises_list.append(noises)
            client_labels_list.append(labels)
            print(f"  received {n_proxy} proxy samples from client {client.id}")

        self.aggregated_noises, self.aggregated_labels = aggregate_client_noises(
            client_noises_list, client_labels_list
        )

        if int(self.aggregated_labels.numel()) != expected_total:
            raise RuntimeError(
                f"Aggregated N_f size mismatch: expected {expected_total}, "
                f"got {self.aggregated_labels.numel()}."
            )

        self.proxy_noise_loader = create_noise_dataloader(
            self.aggregated_noises,
            self.aggregated_labels,
            batch_size=self.batch_size,
            shuffle=True,
        )

        unique, counts = torch.unique(self.aggregated_labels, return_counts=True)
        distribution = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        print(f"  global N_f size = {len(self.aggregated_labels)}")
        print(f"  global N_f distribution = {dict(sorted(distribution.items()))}")

        client_ids = "_".join(str(c.id) for c in self.unlearning_clients)
        save_proxy_noise(
            self.aggregated_noises,
            self.aggregated_labels,
            f"proxy_noise_{self.dataset}_clients_{client_ids}.pt",
        )

    # ------------------------------------------------------------------
    # Stage 2：Knowledge Disentanglement
    # ------------------------------------------------------------------
    def fit_knowledge_disentanglement(self):
        """
        Stage 2：知识解耦。

        1) 冻结 classifier head，只更新 backbone/base；
        2) hook backbone 中最后一个 Conv2d 的输出 F_conv；
        3) 对每个通道计算空间维度 L1 norm，再在 batch 维求平均；
        4) 选出 bottom-(1-alpha) 的低重要性通道；
        5) 最小化这些通道的平均 L1 激活，使遗忘相关纠缠知识逐渐被压制。

        这里使用 proxy N_f，不访问目标客户端真实图像。
        """
        if not hasattr(self.global_model, "base") or not hasattr(self.global_model, "head"):
            raise RuntimeError("Jellyfish expects a BaseHeadSplit model with .base and .head.")

        self.global_model.head.requires_grad_(False)
        self.global_model.base.requires_grad_(True)

        target_conv_layer = None
        for _, module in self.global_model.base.named_modules():
            if isinstance(module, nn.Conv2d):
                target_conv_layer = module
        if target_conv_layer is None:
            raise RuntimeError("Knowledge disentanglement requires a convolutional backbone.")

        feature_box = []

        def hook_fn(_module, _inputs, output):
            feature_box.append(output)

        hook = target_conv_layer.register_forward_hook(hook_fn)
        optimizer = torch.optim.SGD(self.global_model.base.parameters(), lr=self.dis_lr)

        try:
            self.global_model.train()
            self._freeze_bn_stats(self.global_model)

            for epoch in range(self.dis_epochs):
                total_loss = 0.0
                num_batches = 0

                for x, _ in self.proxy_noise_loader:
                    x = x.to(self.device)
                    feature_box.clear()

                    optimizer.zero_grad(set_to_none=True)
                    _ = self.global_model(x)
                    if not feature_box:
                        raise RuntimeError("Last-conv forward hook did not fire.")

                    f_conv = feature_box[-1]
                    # Paper: L1 norm of each HxW feature map, then aggregate batch.
                    channel_norms = f_conv.abs().sum(dim=(2, 3)).mean(dim=0)
                    c = channel_norms.numel()
                    k = max(1, int(round((1.0 - self.alpha_dis) * c)))
                    k = min(k, c)
                    bottom_idx = torch.topk(channel_norms, k=k, largest=False).indices
                    loss_dis = channel_norms[bottom_idx].mean()

                    loss_dis.backward()
                    optimizer.step()

                    total_loss += float(loss_dis.item())
                    num_batches += 1

                print(
                    f"  disentangle epoch {epoch + 1}/{self.dis_epochs}: "
                    f"loss={total_loss / max(num_batches, 1):.6f}"
                )
        finally:
            hook.remove()
            self.global_model.head.requires_grad_(True)
            self.global_model.eval()

    # ------------------------------------------------------------------
    # Stage 3-A：Gradient Mask，论文 Eq.(21)-(22)
    # ------------------------------------------------------------------
    def build_gradient_mask(self):
        """
        根据论文 Eq.(21)-(22) 构造参数级二值 mask。

        对遗忘参考模型 omega_t (= original_model) 和 proxy N_f 计算普通 CE：
            l_f = CE(M_omega_t(x_f), y_f)

        参数敏感度近似为：
            s = mean_i | grad_omega l_i |

        再按阈值 pi 得到：
            m_s = 1(s < pi)

        后续只对 drift/retention gradient 使用：
            g_r' = g_r * m_s

        默认 mask_microbatch=1 时最接近逐样本绝对梯度。
        """
        model = self.original_model
        model.eval()

        # original_model 平时只作为参考快照使用；构造 mask 时需要对其参数求梯度。
        for p in model.parameters():
            p.requires_grad_(True)

        saliency = {
            name: torch.zeros_like(p, device=self.device)
            for name, p in model.named_parameters()
        }
        total_samples = 0

        for x, y in self.proxy_noise_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            for start in range(0, y.size(0), self.mask_microbatch):
                end = min(start + self.mask_microbatch, y.size(0))
                xm, ym = x[start:end], y[start:end]
                bs = ym.size(0)

                model.zero_grad(set_to_none=True)
                out = model(xm)
                loss = F.cross_entropy(out, ym, reduction="mean")
                loss.backward()

                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            # abs before accumulation avoids cross-batch sign cancellation.
                            saliency[name] += p.grad.detach().abs() * bs
                total_samples += bs

        if total_samples == 0:
            raise RuntimeError("Cannot build gradient mask from an empty N_f.")

        mask = {}
        total_n, total_keep = 0, 0.0
        with torch.no_grad():
            for name, s in saliency.items():
                mean_abs_grad = s / float(total_samples)
                m = (mean_abs_grad < self.pi_mask).float()
                mask[name] = m
                total_n += m.numel()
                total_keep += float(m.sum().item())

        keep_ratio = total_keep / max(total_n, 1)
        print(
            f"  gradient-mask keep ratio={keep_ratio:.6f}, "
            f"masked ratio={1.0 - keep_ratio:.6f}, pi={self.pi_mask:g}"
        )
        if keep_ratio > 0.999:
            print("  [WARN] Mask is almost all ones; pi may be too large.")
        if keep_ratio < 0.001:
            print("  [WARN] Mask is almost all zeros; pi may be too small.")
        return mask

    # ------------------------------------------------------------------
    # Stage 3-B：Joint Unlearning
    # ------------------------------------------------------------------
    def fit_joint_unlearning(self):
        """
        Stage 3：论文联合遗忘核心。

        Forgetting branch：
            L_un = L_hard + mu_c*L_confusion + mu_d*L_distillation
            -> 得到 g_f

        Retention / drift branch：
            L_drift = 1/2 ||omega - omega_ref||_2^2
            g_r = omega - omega_ref
            g_r' = g_r * m_s

        最后在完整参数空间判断 g_f 与 g_r' 是否冲突，执行 Eq.(20) 投影，
        得到 G，并按论文 Algorithm 1 直接更新：
            omega <- omega - mu_un * G

        为避免引入论文之外的额外更新，这里不通过带 momentum/weight_decay
        的 optimizer.step() 更新最终参数。
        """
        for p in self.global_model.parameters():
            p.requires_grad_(True)

        bad_teachers = self._build_bad_teachers()
        gradient_mask = self.build_gradient_mask()
        ref_params = dict(self.original_model.named_parameters())

        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction="batchmean")

        self.global_model.train()
        self._freeze_bn_stats(self.global_model)

        for epoch in range(self.unlearn_epochs):
            total_unlearn = 0.0
            total_drift = 0.0
            conflicts = 0
            n_batches = 0

            for x, y in self.proxy_noise_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # ----- forgetting gradient g_f -----
                self.global_model.zero_grad(set_to_none=True)
                outputs = self.global_model(x)

                # Eq. (9): minimizing +log p_y == negative CE.
                loss_hard = -criterion_ce(outputs, y)

                with torch.no_grad():
                    prob = F.softmax(outputs, dim=1)
                    prob.scatter_(1, y.view(-1, 1), -1.0)
                    y_fake = torch.argmax(prob, dim=1)
                loss_confusion = criterion_ce(outputs, y_fake)

                log_p_student = F.log_softmax(outputs / self.distill_temp, dim=1)
                loss_distill = torch.zeros((), device=self.device)
                for teacher in bad_teachers:
                    with torch.no_grad():
                        p_teacher = F.softmax(teacher(x) / self.distill_temp, dim=1)
                    loss_distill = loss_distill + criterion_kl(log_p_student, p_teacher)
                loss_distill = loss_distill / float(len(bad_teachers))

                loss_unlearn = (
                    loss_hard
                    + self.mu_c * loss_confusion
                    + self.mu_d * loss_distill
                )
                loss_unlearn.backward()

                g_f = {
                    name: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                    for name, p in self.global_model.named_parameters()
                }

                # ----- drift gradient g_r = d(1/2||w-w_ref||^2)/dw -----
                g_r_primed = {}
                loss_drift = torch.zeros((), device=self.device)
                with torch.no_grad():
                    for name, p in self.global_model.named_parameters():
                        diff = p.detach() - ref_params[name].detach()
                        loss_drift += 0.5 * torch.sum(diff * diff)
                        g_r_primed[name] = diff * gradient_mask[name]

                # ----- Eq. (20), full-model projection -----
                g_composite, dot = self._global_gradient_harmonization(g_f, g_r_primed)
                if dot < 0:
                    conflicts += 1
                g_composite = self._clip_gradient_dict(
                    g_composite, self.max_update_norm
                )

                # Literal Algorithm-1 update: w <- w - mu_un * G.
                # No momentum/weight-decay term is injected outside the mask.
                with torch.no_grad():
                    for name, p in self.global_model.named_parameters():
                        p.add_(g_composite[name], alpha=-self.unlearn_lr)

                total_unlearn += float(loss_unlearn.item())
                total_drift += float(loss_drift.item())
                n_batches += 1

            print(
                f"  unlearn epoch {epoch + 1}/{self.unlearn_epochs}: "
                f"L_un={total_unlearn / max(n_batches, 1):.6f}, "
                f"L_drift={total_drift / max(n_batches, 1):.6f}, "
                f"conflict_batches={conflicts}/{n_batches}"
            )

        self.global_model.eval()

    # ------------------------------------------------------------------
    # Stage 4：Zero-shot Adaptive Repair
    # ------------------------------------------------------------------
    def adaptive_model_repair(self):
        """
        Stage 4：可选的零样本自适应修复。

        whole-client 删除场景：
            - target client 的全部数据均属于 D_f，因此绝不能参与 repair；
            - 只检查 remaining clients 的性能下降；
            - 若相对准确率下降超过 delta，则该客户端本地生成 retention proxy N_r^i；
            - 服务器按客户端 retained-data 规模加权组合 N_r，并使用 Algorithm 1
              中的 MSE repair objective 对遗忘后模型进行少量修复。
        """
        forget_ids = {c.id for c in self.unlearning_clients}
        # Correct for whole-client deletion: target client owns no retained data.
        remaining_clients = [c for c in self.clients if c.id not in forget_ids]
        if not remaining_clients:
            print("  no remaining clients; repair skipped")
            return

        repair_parts = []
        for client in remaining_clients:
            pre_acc = self._client_accuracy(client, self.original_model)
            post_acc = self._client_accuracy(client, self.global_model)
            relative_drop = (pre_acc - post_acc) / max(pre_acc, 1e-12)

            print(
                f"  client {client.id}: pre={pre_acc:.4f}, post={post_acc:.4f}, "
                f"relative_drop={relative_drop:.4f}"
            )

            # Jellyfish Eq. (23): relative percentage drop, not absolute points.
            if relative_drop > self.delta_threshold:
                r_x, r_y = client.generate_retention_noise(
                    global_model=self.global_model,
                    steps=self.repair_noise_steps,
                    lr=self.repair_noise_lr,
                )
                n_local = int(getattr(client, "train_samples", r_y.numel()))
                repair_parts.append((r_x.detach().cpu(), r_y.detach().cpu(), n_local))

        if not repair_parts:
            print("  no client crossed delta; repair skipped")
            return

        # Weight each client's repair contribution proportional to |D_r^i|,
        # independently of how many proxy samples NoiseGenerator returned.
        xs, ys, ws = [], [], []
        for x, y, n_local in repair_parts:
            n_proxy = max(int(y.numel()), 1)
            per_proxy_weight = float(n_local) / float(n_proxy)
            xs.append(x)
            ys.append(y)
            ws.append(torch.full((n_proxy,), per_proxy_weight, dtype=torch.float32))

        x_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
        w_all = torch.cat(ws, dim=0)
        loader = DataLoader(
            TensorDataset(x_all, y_all, w_all),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.repair_lr)
        self.global_model.train()
        self._freeze_bn_stats(self.global_model)

        for epoch in range(self.repair_epochs):
            total_loss, n_batches = 0.0, 0
            for x, y, w in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.global_model(x)

                # Algorithm 1 writes MSE for L_repair.  Eq. (24) in the text
                # is the objective for generating N_r, not the server repair loss.
                target = F.one_hot(y, num_classes=self.num_classes).float()
                pred = F.softmax(logits, dim=1)
                per_sample = torch.mean((pred - target) ** 2, dim=1)
                loss = torch.sum(per_sample * w) / torch.sum(w).clamp_min(1e-12)

                loss.backward()
                if self.max_update_norm is not None and self.max_update_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.global_model.parameters(), self.max_update_norm
                    )
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            print(
                f"  repair epoch {epoch + 1}/{self.repair_epochs}: "
                f"loss={total_loss / max(n_batches, 1):.6f}"
            )

        self.global_model.eval()
