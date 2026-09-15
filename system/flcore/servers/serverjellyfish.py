import time
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from flcore.clients.clientjellyfish import clientJellyfish
from flcore.servers.serverbase import Server
from utils.attack_utils import attack, train_attack_model
from utils.noise_utils import *


class Jellyfish(Server):
    def __init__(self, args):
        super().__init__(args)
        # 这一步确认初始模型
        self.initial_model = copy.deepcopy(self.global_model).to(self.device)
        self.initial_model.eval()
        for p in self.initial_model.parameters():
            p.requires_grad_(False)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # 第一步添加的变量
        self.original_model = None
        self.proxy_noise_loader = None
        self.aggregated_noises = None
        self.aggregated_labels = None

        #第二步添加的变量
        self.feature_maps = None
        self.channel_importance = None
        self.channel_mask = None

        # 第四步 Repair 相关变量
        self.repair_noise_loader = None
        self.aggregated_repair_noises = None
        self.aggregated_repair_labels = None
        self.pre_unlearning_client_acc = {}
        self.stage4_history = []
        self.repair_triggered_client_ids = []

        # 选择慢客户端并初始化客户端集合
        self.set_slow_clients()
        self.set_clients(clientJellyfish)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # learning和unlearning的时间和内存成本
        self.Budget = []
        self.unlearn_Budget = []  # 计时 unlearning 的时间

    def train(self):
        """
        标准的 FedAvg 联邦训练过程
        """
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            # 1. 选择当前轮次参与训练的客户端
            self.selected_clients = self.select_clients(state="train")

            # 2. 将当前全局模型的参数广播/发送给被选中的客户端
            self.send_models()

            # 3. 定期评估全局模型在测试集上的性能
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # 4. 客户端在本地执行训练
            for client in self.selected_clients:
                client.train()

            # 5. 服务端接收客户端上传的训练后模型
            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # 6. 使用 FedAvg 的核心：基于样本数量加权聚合参数
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # 7. 训练完成后，训练成员推理攻击模型（MIA）评估遗忘前模型的隐私表现
        self.attacker = train_attack_model(self.global_model, self.clients, self.num_classes, self.device)

        (PRE_old, REC_old) = attack(self.global_model, self.attacker, self.unlearning_clients, self.num_classes,
                                    self.device)
        print("MIA Attacker to old model precision = {:.4f}".format(PRE_old))
        print("MIA Attacker to old model recall = {:.4f}".format(REC_old))

        # 8. 保存训练结果与全局模型
        self.save_results()
        self.save_global_model()

        # 9. 如果有新客户端加入的评估逻辑
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientJellyfish)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

        # 10. 提取所有客户端特征并保存
        results_dict = {}
        self.global_model.eval()
        all_clients = self.clients + self.unlearning_clients

        with torch.no_grad():
            for c in all_clients:
                current_features = []
                current_labels = []

                for _, data in enumerate(c.test_loader):
                    x, y = data

                    if isinstance(x, list):
                        x = [item.to(self.device) for item in x]
                    else:
                        x = x.to(self.device)

                    output = self.global_model(x)

                    current_features.append(output.cpu())
                    current_labels.append(y.cpu())

                results_dict[c.id] = {
                    'features': torch.cat(current_features, dim=0),
                    'labels': torch.cat(current_labels, dim=0)
                }

        torch.save(results_dict, "ALL_sne_grouped.pt")



    def unlearning(self):
        """
        基于 FedAvg 架构的联邦遗忘（Unlearning）过程：
        核心思路：从客户端列表中剔除需要遗忘的客户端，并利用剩余的正常客户端重新进行 FedAvg 训练（或从头训练/微调）。
        """
        self.load_model()
        self.send_models()
        self.warm_up()
        self.global_model.eval()
        for i in self.clients:
            i.model.eval()
        self.original_model = copy.deepcopy(self.global_model) #original_model为遗忘开始前的参考模型，在这之后不能修改
        self.original_model.eval()

        # ============================================================
        # Stage 4 Eq.(23) 所需基线：
        # 在真正开始 unlearning 前，记录每个“剩余客户端”的本地测试准确率 Acc_t^i。
        # 后续 Stage 3 完成后，用它计算相对下降比例 DeltaAcc_i。
        # ============================================================
        self.pre_unlearning_client_acc = (
            self.collect_remaining_client_accuracies(
                model=self.global_model,
                verbose=True
            )
        )

        print("Before unlearning Test")
        self.evaluate() #测量unlearning开始前的model数据

        # 第一步，unlearning clients生成代理数据集，然后由server收集并整合，然后保存这个proxy noise数据集

        (client_noises, client_labels)= self.generate_proxy_noise()
        self.aggregate_proxy_noise(client_noises, client_labels)
        self.save_proxy_noise()
        # print(
        #     "Jellyfish Stage 1 Finished."
        # )

        # 第二步，知识解耦
        self.knowledge_disentanglement()
        # print(
        #     "Jellyfish Stage2 Finished"
        # )
        self.send_models()
        self.global_model.eval()
        self.evaluate()

        # ============================================================
        # Stage 3: Loss Function Construction + Gradient Optimization
        # ============================================================
        self.loss_function_unlearning()
        # print("\nJellyfish Stage 3 Finished.")
        # 同样将 Stage 3 后的 global model 广播给 clients 再评估。
        self.send_models()
        self.global_model.eval()
        print("\n" + "=" * 70)
        print(" Evaluation After Stage 3 ")
        print("=" * 70)
        self.evaluate()
        # 保存 Stage 3 后模型，方便后续 Stage 4 Repair 使用。
        torch.save(
            self.global_model.state_dict(),
            "jellyfish_stage3_unlearned_model.pt"
        )

        # ============================================================
        # Stage 4: Repair (论文 4.5)
        # ============================================================
        # Repair 是可选、一次性的。只有当至少一个剩余客户端满足
        # Eq.(23) 的相对准确率下降阈值 DeltaAcc_i > delta 时才触发。
        self.model_repair()

        # Stage 4 完成后广播并做最终统一评估。
        self.send_models()
        self.global_model.eval()
        print("\n" + "=" * 70)
        print(" Evaluation After Stage 4 Repair ")
        print("=" * 70)
        self.evaluate()

        torch.save(
            {
                "model_state_dict": self.global_model.state_dict(),
                "stage4_history": self.stage4_history,
                "repair_triggered_client_ids": self.repair_triggered_client_ids,
            },
            "jellyfish_stage4_repaired_checkpoint.pt"
        )







    #下面是第一步的生成noise，整合和保存的三个方法
    def generate_proxy_noise(self):
        """
        ============================================================
        Jellyfish Stage 1.1 Client-side Proxy Noise Generation
        功能:
            调用遗忘客户端生成代理遗忘数据 N_f^i
        Server职责:
            1. 提供遗忘前模型 omega_t
            2. 调度client生成noise
            3. 收集client返回结果
        Return:
            client_noises_list
            client_labels_list
        ============================================================
        """
        if len(self.unlearning_clients) == 0:
            raise RuntimeError(
                "No unlearning client."
            )

        if self.original_model is None:
            raise RuntimeError(
                "original_model is None."
            )

        # print("\n")
        # print("=" * 70)
        # print(" Jellyfish Stage 1.1 Generate Proxy Noise ")
        # print("=" * 70)
        client_noises_list = []
        client_labels_list = []
        for client in self.unlearning_clients:
            # print(
            #     f"\nClient {client.id} generating noise..."
            # )
            # 生成噪声和label
            noises, labels = (
                client.generate_proxy_noise(
                    global_model=self.original_model
                )
            )
            client_noises_list.append(noises)
            client_labels_list.append(labels)
        return (client_noises_list, client_labels_list)

    def aggregate_proxy_noise(
            self,
            client_noises_list,
            client_labels_list
    ):
        """
        ============================================================
        Jellyfish Stage 1.2
        功能:

            将多个forget client生成的N_f^I合并为N_f
        输出:
            self.aggregated_noises
            self.aggregated_labels
        ============================================================
        """
        # print("\n")
        # print("=" * 70)
        # print(" Jellyfish Stage 1.2 Aggregate Proxy Noise ")
        # print("=" * 70)

        (noises, labels) = aggregate_client_noises(client_noises_list, client_labels_list)
        self.aggregated_noises = noises
        self.aggregated_labels = labels
        # print(
        #     "Global proxy noise shape:",
        #     noises.shape
        # )
        # 创建后续unlearning使用的数据加载器
        self.proxy_noise_loader = (
            create_noise_dataloader(
                noises,
                labels,
                batch_size=self.batch_size,
                shuffle=True
            )
        )
        return noises, labels

    def save_proxy_noise(self):
        """
        ============================================================
        Jellyfish Stage 1.3
        保存N_f he Y_f
        ============================================================
        """
        if self.aggregated_noises is None:
            raise RuntimeError(
                "No aggregated proxy noise."
            )

        client_ids = "_".join(
            str(c.id)
            for c in self.unlearning_clients
        )

        save_path = (
            f"proxy_noise_"
            f"{self.dataset}_"
            f"clients_{client_ids}.pt"
        )

        save_proxy_noise(
            self.aggregated_noises,
            self.aggregated_labels,
            save_path
        )
        print("Proxy noise saved:", save_path)

    #下面是第二步知识解耦的方法，
    def knowledge_disentanglement(self):
        """
        ============================================================
        Jellyfish Stage 2: Knowledge Disentanglement
        目的:使用Stage1生成的proxy noise N_f对模型最后卷积层feature进行稀疏化约束。
        输入: self.proxy_noise_loader
        输出:更新后的global_model
        ============================================================
        """
        if self.proxy_noise_loader is None:
            raise RuntimeError(
                "Proxy noise loader is empty."
            )
        # 修改全局模型可以训练
        self.global_model.train()

        # 1. 找到最后卷积层
        target_layer = get_last_conv_layer(
            self.global_model
        )
        feature_hook = FeatureHook()
        hook_handle = (
            target_layer.register_forward_hook(
                feature_hook.hook_fn
            )
        )
        # 冻结
        for p in self.global_model.parameters():
            p.requires_grad = False

        # 解冻最后卷积
        for p in target_layer.parameters():
            p.requires_grad = True

        # 整个模型进入训练状态
        self.global_model.train()
        # 但是冻结所有 BatchNorm 的 running statistics
        for module in self.global_model.modules():
            if isinstance(
                    module,
                    nn.modules.batchnorm._BatchNorm
            ):
                module.eval()

        # 训练模型参数
        optimizer = torch.optim.SGD(
            target_layer.parameters(),
            lr=getattr(
                self.args,
                "dis_lr",
                1e-4
            )
        )
        epochs = getattr(
            self.args,
            "dis_epochs",
            5
        )
        alpha = getattr(
            self.args,
            "alpha",
            0.2
        )
        for epoch in range(epochs):
            total_loss = 0
            for x, y in self.proxy_noise_loader:
                x = x.to(self.device)
                # forward
                self.global_model(x)
                feature = (feature_hook.features)
                if feature is None:
                    raise RuntimeError(
                        "No feature captured"
                    )
                # 计算损失并反向传播只更新最后一层卷积层
                loss = threshold_disentangle_loss(
                    feature,
                    alpha
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(
            #     f"Epoch {epoch + 1}/{epochs}, "
            #     f"L_dis={total_loss / len(self.proxy_noise_loader)}"
            # )
        hook_handle.remove()
        torch.save(
            self.global_model.state_dict(),
            "jellyfish_disentangled_model.pt"
        )
        # print(
        #     "Knowledge disentanglement finished"
        # )

    #下面是第三步关于复合损失函数构建的几个方法
    @staticmethod
    def get_fake_labels(logits, true_labels):
        """
        论文 Eq.(10)-(11): 为每个遗忘样本找到概率最高的“错误类别”。
            y_fake = argmax_{i != y_f} P_M(x_f)[i]
        logits:
            当前 student/global model 对 x_f 的输出 [B, num_classes]
        true_labels:
            原始遗忘标签 y_f [B]
        """
        probs = torch.softmax(logits, dim=1)
        fake_probs = probs.clone()

        row_idx = torch.arange(
            true_labels.size(0),
            device=true_labels.device
        )
        # 排除正确类别，使 argmax 一定选到非 y_f 类别。
        fake_probs[row_idx, true_labels] = -float("inf")
        return torch.argmax(fake_probs, dim=1)

    @staticmethod
    def compute_distillation_loss(
            student_logits,
            teacher_logits,
            temperature
    ):
        """
        论文 Eq.(13)-(15)
            P_teacher = softmax(v / Temp)
            P_student = softmax(z / Temp)
            L_distillation =
                D_KL(P_teacher || P_student)
        PyTorch F.kl_div 的调用约定是：
            input  = log P_student
            target = P_teacher
        """
        T = float(temperature)
        if T <= 0:
            raise ValueError(f"temperature must be positive, got {T}")
        teacher_prob = torch.softmax(
            teacher_logits / T,
            dim=1
        )
        student_log_prob = torch.log_softmax(
            student_logits / T,
            dim=1
        )
        # 论文公式没有额外写 T^2，因此这里不额外乘 T^2。
        return F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="batchmean"
        )

    @staticmethod
    def compute_drift_loss(current_model, reference_model):
        """
        论文 Eq.(17):
            L_drift =
                1/2 ||omega_un - omega_t||_2^2
        current_model:
            当前正在执行遗忘的模型 omega_un
        reference_model:
            遗忘开始前固定的模型 omega_t
        """
        loss = None
        for current_param, ref_param in zip(
                current_model.parameters(),
                reference_model.parameters()
        ):
            term = 0.5 * torch.sum(
                (current_param - ref_param.detach()) ** 2
            )

            loss = term if loss is None else loss + term
        if loss is None:
            raise RuntimeError("Model contains no parameters for drift loss.")
        return loss

    @staticmethod
    def _safe_autograd_grad(
            loss,
            params,
            retain_graph=False
    ):
        """
        torch.autograd.grad 的安全包装。

        对极端情况下没有参与当前 loss 的参数，
        将 None gradient 转为 zeros_like(param)，
        这样后续 Gradient Mask / Harmonization 可以保持严格一一对应。
        """
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True
        )

        safe_grads = []

        for grad, param in zip(grads, params):
            if grad is None:
                safe_grads.append(
                    torch.zeros_like(param)
                )
            else:
                safe_grads.append(grad)

        return safe_grads

    @staticmethod
    def _gradient_global_norm(grads):
        """
        计算一组参数梯度构成的整体 L2 norm。

        这里使用 float64 累加，避免像 Stage 3 中出现的超大梯度在
        float32 下执行平方求和时先溢出为 Inf。
        """
        if len(grads) == 0:
            return torch.tensor(0.0)

        device = grads[0].device
        total = torch.zeros((), dtype=torch.float64, device=device)

        for grad in grads:
            grad64 = grad.detach().to(torch.float64)
            total += torch.sum(grad64 * grad64)

        return torch.sqrt(total + 1e-24)

    @staticmethod
    def clip_final_gradients(grads, max_norm=5.0, eps=1e-12):
        """
        对 Jellyfish Gradient Harmonization 之后得到的最终复合梯度 G
        执行 global-norm clipping。

        注意：
            1. 这一步是数值稳定性工程措施，不是论文 Eq.(8)-(22) 的新增损失项。
            2. clipping 放在 Gradient Harmonization 之后，因此不会改变
               g_f 与 g'_r 的 cosine 判断，也不会改变 Eq.(20) 的投影过程。
            3. 所有参数梯度使用同一个缩放系数，因此只缩短 G 的长度，
               不改变 G 的整体方向。

        数学：
            若 ||G||_2 <= max_norm:
                G_clip = G

            若 ||G||_2 > max_norm:
                G_clip = G * max_norm / ||G||_2

        Returns:
            clipped_grads: 截断后的最终梯度
            original_norm: 截断前 ||G||_2（Python float）
            clipped_norm:  截断后 ||G||_2（Python float）
            clip_coef:     实际缩放系数，1 表示本 batch 未触发 clipping
        """
        if max_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be > 0, got {max_norm}"
            )

        if len(grads) == 0:
            return [], 0.0, 0.0, 1.0

        device = grads[0].device
        norm_sq = torch.zeros((), dtype=torch.float64, device=device)

        # 先确认 G 尚未出现 NaN / Inf。
        # clipping 只能控制“很大但有限”的梯度，不能修复已经产生的 NaN / Inf。
        for index, grad in enumerate(grads):
            if not torch.isfinite(grad).all():
                raise RuntimeError(
                    f"final_grads[{index}] already contains NaN/Inf before clipping."
                )

            grad64 = grad.detach().to(torch.float64)
            norm_sq += torch.sum(grad64 * grad64)

        norm = torch.sqrt(norm_sq + 1e-24)

        if not torch.isfinite(norm):
            raise RuntimeError(
                "Final gradient global norm is NaN/Inf before clipping."
            )

        original_norm = float(norm.item())
        clip_coef = min(
            1.0,
            float(max_norm) / (original_norm + eps)
        )

        if clip_coef < 1.0:
            clipped_grads = [
                grad * clip_coef
                for grad in grads
            ]
        else:
            clipped_grads = [
                grad.clone()
                for grad in grads
            ]

        # 因为使用统一系数缩放，所以理论上 clipped_norm = original_norm * clip_coef。
        clipped_norm = original_norm * clip_coef

        return (
            clipped_grads,
            original_norm,
            clipped_norm,
            clip_coef
        )


    def build_gradient_mask(
            self,
            mask_reference_model,
            x_f,
            y_f,
            pi
    ):
        """
        论文 Eq.(21)-(22) 的 Eq.(21) 部分：
            l(x_f, y_f; omega_t)
                = CE(M(x_f; omega_t), y_f)
            m_s =
                1(
                    |grad_omega l(x_f,y_f;omega_t)| < pi
                )
        直觉：
            如果某个参数位置在遗忘数据上的梯度绝对值很大，
            说明该位置和 D_f / N_f 高度相关。
            这些位置 mask=0，
            后续不允许 remembering gradient g_r
            把原始 forgotten knowledge 拉回来。
        注意：
            mask_reference_model 必须是 omega_t 的可求导副本，
            但这个副本永远不执行 optimizer.step()。
        """
        if pi <= 0:
            raise ValueError(f"gradient mask threshold pi must be > 0, got {pi}")

        mask_reference_model.eval()

        ref_params = [
            p for p in mask_reference_model.parameters()
            if p.requires_grad
        ]

        ref_logits = mask_reference_model(x_f)

        # 这里使用普通正向 CE。
        saliency_loss = F.cross_entropy(
            ref_logits,
            y_f
        )

        saliency_grads = self._safe_autograd_grad(
            saliency_loss,
            ref_params,
            retain_graph=False
        )

        masks = []
        kept_elements = 0
        total_elements = 0

        for grad in saliency_grads:
            mask = (
                    torch.abs(grad) < pi
            ).to(grad.dtype)

            masks.append(mask)

            kept_elements += int(mask.sum().item())
            total_elements += mask.numel()

        keep_ratio = (
                kept_elements / max(total_elements, 1)
        )

        return masks, keep_ratio

    @staticmethod
    def apply_gradient_mask(remember_grads, masks):
        """
        论文 Eq.(22):
            g'_r = g_r ⊙ m_s
        """
        if len(remember_grads) != len(masks):
            raise RuntimeError(
                "Gradient/mask length mismatch in gradient masking."
            )

        return [
            grad * mask
            for grad, mask in zip(
                remember_grads,
                masks
            )
        ]

    @staticmethod
    def gradient_harmonization(
            forget_grads,
            remember_grads
    ):
        """
        Jellyfish Eq.(20)

        输入：
            forget_grads  = g_f
            remember_grads = g'_r

        输出：
            final_grads = G
        """

        if len(forget_grads) != len(remember_grads):
            raise RuntimeError(
                "Forgetting/remembering gradient length mismatch."
            )

        device = forget_grads[0].device

        # --------------------------------------------------
        # 使用 float64 计算全局 dot / norm
        # 防止大量参数平方累加导致 float32 overflow
        # --------------------------------------------------

        dot = torch.zeros(
            (),
            dtype=torch.float64,
            device=device
        )

        norm_f_sq = torch.zeros(
            (),
            dtype=torch.float64,
            device=device
        )

        norm_r_sq = torch.zeros(
            (),
            dtype=torch.float64,
            device=device
        )

        for g_f, g_r in zip(
                forget_grads,
                remember_grads
        ):
            gf64 = g_f.detach().to(
                torch.float64
            )

            gr64 = g_r.detach().to(
                torch.float64
            )

            dot += torch.sum(
                gf64 * gr64
            )

            norm_f_sq += torch.sum(
                gf64 * gf64
            )

            norm_r_sq += torch.sum(
                gr64 * gr64
            )

        # --------------------------------------------------
        # 数值检查
        # --------------------------------------------------

        if not torch.isfinite(dot):
            raise RuntimeError(
                "Non-finite dot(g_f, g_r) "
                "in Gradient Harmonization."
            )

        if not torch.isfinite(norm_f_sq):
            raise RuntimeError(
                "Non-finite ||g_f||^2 "
                "in Gradient Harmonization."
            )

        if not torch.isfinite(norm_r_sq):
            raise RuntimeError(
                "Non-finite ||g_r||^2 "
                "in Gradient Harmonization."
            )

        eps = torch.tensor(
            1e-24,
            dtype=torch.float64,
            device=device
        )

        # --------------------------------------------------
        # 特殊情况：
        #
        # Gradient Masking 后 g'_r = 0
        #
        # 不存在 remembering direction，
        # 所以直接使用 g_f
        # --------------------------------------------------

        if norm_r_sq.item() <= eps.item():
            final_grads = [
                g.clone()
                for g in forget_grads
            ]

            return (
                final_grads,
                0.0,
                False
            )

        # 极端情况下 g_f = 0

        if norm_f_sq.item() <= eps.item():
            final_grads = [
                g.clone()
                for g in remember_grads
            ]

            return (
                final_grads,
                0.0,
                False
            )

        # --------------------------------------------------
        # cosine similarity
        # --------------------------------------------------

        cosine64 = (
                dot
                /
                torch.sqrt(
                    (norm_f_sq + eps)
                    *
                    (norm_r_sq + eps)
                )
        )

        cosine = float(
            cosine64.item()
        )

        # --------------------------------------------------
        # Eq.(20)
        #
        # cos < 0:
        # 存在梯度冲突
        # --------------------------------------------------

        if cosine < 0.0:

            projection_coeff64 = (
                    dot
                    /
                    (norm_r_sq + eps)
            )

            corrected_forget_grads = []

            for g_f, g_r in zip(
                    forget_grads,
                    remember_grads
            ):
                # 转回模型参数对应dtype
                coeff = projection_coeff64.to(
                    device=g_f.device,
                    dtype=g_f.dtype
                )

                corrected_forget_grads.append(
                    g_f
                    -
                    coeff * g_r
                )

            has_conflict = True

        else:

            corrected_forget_grads = [
                g.clone()
                for g in forget_grads
            ]

            has_conflict = False

        # --------------------------------------------------
        # Final:
        #
        # G = g'_f + g'_r
        # --------------------------------------------------

        final_grads = [

            g_f_corr + g_r

            for g_f_corr, g_r in zip(
                corrected_forget_grads,
                remember_grads
            )
        ]

        return (
            final_grads,
            cosine,
            has_conflict
        )

    @staticmethod
    def assign_gradients(params, grads):
        """
        将手工构造出来的最终梯度 G 写回 param.grad，
        随后由 optimizer.step() 真正执行模型参数更新。
        """
        if len(params) != len(grads):
            raise RuntimeError(
                "Parameter/final-gradient length mismatch."
            )

        for param, grad in zip(params, grads):
            param.grad = grad.detach().clone()

    def _prepare_incompetent_teachers(self):
        """
        准备论文 Eq.(15)-(16) 中的 incompetent teacher 集合 T_set。

        论文正文：
            - 单个 M_bad 对应 Eq.(15)
            - 为降低单 teacher 偏差，可使用
                  T_set = {T_i}_{i=1}^{N_T}
              并对多个 teacher 的 KL loss 求平均（Eq.16）
        当前工程能确定拿到的“未接触 forgotten data 的模型”是：
            self.initial_model = omega_0
        因此默认 teacher set 为：
            [omega_0]

        如果你以后已经在外部准备好了多个 incompetent teacher，
        可以在调用 Stage 3 前设置：
            self.bad_teacher_models = [teacher1, teacher2, ...]

        本方法会优先使用 self.bad_teacher_models。

        非常重要：
            如果 train 与 unlearning 是两个独立进程，
            最严谨的复现方式是训练开始时保存“真正使用过的 omega_0”，
            unlearning 时加载同一个 omega_0。
        """
        source_teachers = getattr(
            self,
            "bad_teacher_models",
            None
        )

        if source_teachers is None or len(source_teachers) == 0:
            source_teachers = [self.initial_model]

        teachers = []

        for teacher_src in source_teachers:
            teacher = copy.deepcopy(
                teacher_src
            ).to(self.device)

            teacher.eval()

            for p in teacher.parameters():
                p.requires_grad_(False)

            teachers.append(teacher)

        return teachers

    def loss_function_unlearning(self):
        """
        ============================================================
        Jellyfish Stage 3:
        Loss Function Construction + Gradient Masking
        + Gradient Harmonization

        对应论文 Eq.(8)-(22)。

        每一个 proxy forgetting batch (x_f, y_f) 的执行顺序：

        1. Student forward
        2. L_hard
        3. L_confusion
        4. L_distillation
        5. L_unlearn = L_hard + mu_c L_confusion + mu_d L_distillation
        6. g_f = grad(L_unlearn)
        7. L_drift = 1/2 ||omega_un - omega_t||_2^2
        8. g_r = grad(L_drift)
        9. 用 omega_t 在 (x_f,y_f) 上的 CE gradient 构造 mask m_s
        10. g'_r = g_r ⊙ m_s
        11. Gradient Harmonization(g_f, g'_r)
        12. G = g'_f + g'_r
        13. 把 G 写入 param.grad
        14. optimizer.step()
        ============================================================
        """
        if self.proxy_noise_loader is None:
            raise RuntimeError(
                "Proxy noise loader is empty. Stage 1 must run before Stage 3."
            )

        if self.original_model is None:
            raise RuntimeError(
                "original_model (omega_t) is missing."
            )

        # ------------------------------------------------------------
        # Stage 2 结束后只有最后卷积层 requires_grad=True。
        # Stage 3 论文优化的是完整 omega_un，因此必须重新解冻整个 student。
        # ------------------------------------------------------------
        for p in self.global_model.parameters():
            p.requires_grad_(True)

        self.global_model.train()
        for module in self.global_model.modules():
            if isinstance(
                    module,
                    nn.modules.batchnorm._BatchNorm
            ):
                module.eval()

        # omega_t 永远固定。
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.requires_grad_(False)

        # 用于 Eq.(21) 的可求导 omega_t 副本。
        mask_reference_model = copy.deepcopy(
            self.original_model
        ).to(self.device)

        mask_reference_model.eval()
        for p in mask_reference_model.parameters():
            p.requires_grad_(True)

        # incompetent teacher set T_set。
        # 默认只有 omega_0 一个 teacher；若外部设置 self.bad_teacher_models，
        # 则自动实现论文 Eq.(16) 的多 teacher 平均。
        teacher_models = self._prepare_incompetent_teachers()

        # 论文实验中 mu_c = mu_d = 0.5。
        mu_c = float(
            getattr(self.args, "mu_c", 0.5)
        )
        mu_d = float(
            getattr(self.args, "mu_d", 0.5)
        )

        # 论文定义了 Temp，但正文没有明确给出具体实验值。
        # 这里优先读取工程已有 temperature 参数。
        temperature = float(
            getattr(self.args, "temperature", 3.0)
        )

        # pi 是论文 Eq.(21) 的梯度阈值。
        # 论文正文给出了符号定义，但没有在实验超参数段明确给出数值。
        # 因此这里作为实现超参数。
        pi = float(
            getattr(
                self.args,
                "gradient_mask_pi",
                getattr(self.args, "mask_pi", 1e-3)
            )
        )
        # pi之前选用1.4e-6

        # -ugr 在你当前工程中对应 self.unlearning_ground。
        epochs = int(
            getattr(
                self,
                "unlearning_ground",
                getattr(self.args, "unlearn_epochs", 20)
            )
        )

        if epochs <= 0:
            raise ValueError(
                f"Unlearning epochs must be > 0, got {epochs}"
            )

        unlearn_lr = float(
            getattr(
                self.args,
                "unlearn_rate",
                getattr(self.args, "unlearning_rate", 0.005)
            )
        )

        # 最终复合梯度 G 的 global-norm clipping 阈值。
        # 这是为解决 Stage 3 中 -CE hard loss 导致的梯度爆炸而加入的
        # 数值稳定性措施，不修改论文原始 loss / masking / harmonization 公式。
        grad_clip_norm = float(
            getattr(
                self.args,
                "grad_clip_norm",
                5.0
            )
        )

        # Algorithm 1 写的是 omega <- omega - mu_un * G，
        # 因此这里使用 SGD，使代码和论文更新公式保持直接对应。
        params = [
            p for p in self.global_model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.SGD(
            params,
            lr=unlearn_lr
        )

        print("\n" + "=" * 70)
        print(" Jellyfish Stage 3: Loss Function Construction ")
        print("=" * 70)
        print(f"epochs               = {epochs}")
        print(f"unlearning lr        = {unlearn_lr}")
        print(f"mu_c                 = {mu_c}")
        print(f"mu_d                 = {mu_d}")
        print(f"temperature          = {temperature}")
        print(f"gradient mask pi     = {pi}")
        print(f"gradient clip norm   = {grad_clip_norm}")
        print(
            f"number of teachers   = {len(teacher_models)}"
        )
        print(
            "default teacher      = initial model omega_0 "
            "(unless self.bad_teacher_models is provided)"
        )

        self.stage3_history = []

        for epoch in range(epochs):
            epoch_hard = 0.0
            epoch_confusion = 0.0
            epoch_distill = 0.0
            epoch_unlearn = 0.0
            epoch_drift = 0.0
            epoch_cosine = 0.0
            epoch_mask_keep = 0.0
            epoch_grad_norm_before_clip = 0.0
            epoch_final_grad_norm = 0.0
            epoch_clip_coef = 0.0
            clipped_batches = 0
            conflict_batches = 0
            num_batches = 0

            for x_f, y_f in self.proxy_noise_loader:
                x_f = x_f.to(self.device)
                y_f = y_f.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                # ====================================================
                # Eq.(8)-(16): forgetting objective
                # ====================================================
                student_logits = self.global_model(x_f)

                # Eq.(9):
                # L_hard = + y log M(x)
                # 等价于 -CrossEntropy。
                hard_loss = -F.cross_entropy(
                    student_logits,
                    y_f
                )

                # Eq.(10)-(12):
                # 找概率最高的错误类别，并把 student 推向该类别。
                with torch.no_grad():
                    fake_labels = self.get_fake_labels(
                        student_logits.detach(),
                        y_f
                    )

                confusion_loss = F.cross_entropy(
                    student_logits,
                    fake_labels
                )

                # Eq.(13)-(16):
                # Student 模仿 incompetent teacher。
                # 若有多个 teacher，则对每个 KL loss 求平均，对应 Eq.(16)。
                distill_terms = []

                with torch.no_grad():
                    teacher_logits_list = [
                        teacher_model(x_f)
                        for teacher_model in teacher_models
                    ]

                for teacher_logits in teacher_logits_list:
                    distill_terms.append(
                        self.compute_distillation_loss(
                            student_logits,
                            teacher_logits,
                            temperature
                        )
                    )

                distillation_loss = torch.stack(
                    distill_terms
                ).mean()

                # Eq.(8)
                unlearn_loss = (
                        hard_loss
                        + mu_c * confusion_loss
                        + mu_d * distillation_loss
                )

                # g_f = grad L_unlearn
                forget_grads = self._safe_autograd_grad(
                    unlearn_loss,
                    params,
                    retain_graph=False
                )

                # ====================================================
                # Eq.(17): remembering objective
                # ====================================================
                drift_loss = self.compute_drift_loss(
                    self.global_model,
                    self.original_model
                )

                # g_r = grad L_drift
                remember_grads = self._safe_autograd_grad(
                    drift_loss,
                    params,
                    retain_graph=False
                )

                # ====================================================
                # Eq.(21)-(22): Gradient Masking
                # ====================================================
                masks, keep_ratio = self.build_gradient_mask(
                    mask_reference_model,
                    x_f,
                    y_f,
                    pi
                )

                masked_remember_grads = self.apply_gradient_mask(
                    remember_grads,
                    masks
                )

                # ====================================================
                # Eq.(20): Gradient Harmonization
                # ====================================================
                final_grads, cosine, has_conflict = (
                    self.gradient_harmonization(
                        forget_grads,
                        masked_remember_grads
                    )
                )

                # ====================================================
                # Numerical Stability: Global Norm Clipping on final G
                # ====================================================
                # 论文的 Gradient Masking / Harmonization 已经全部完成。
                # 这里只对最终 G 的长度做统一缩放，避免单个 batch 的超大 G
                # 直接把模型参数推到 NaN 区域。
                (
                    final_grads,
                    grad_norm_before_clip,
                    final_grad_norm,
                    clip_coef
                ) = self.clip_final_gradients(
                    final_grads,
                    max_norm=grad_clip_norm
                )

                if clip_coef < 1.0:
                    clipped_batches += 1

                # G_clip -> param.grad -> optimizer.step()
                self.assign_gradients(
                    params,
                    final_grads
                )

                optimizer.step()

                # optimizer.step() 之后立即检查参数，避免把异常拖到下一 batch
                # 的 student_logits 才发现。
                for param_index, param in enumerate(params):
                    if not torch.isfinite(param).all():
                        nan_count = int(torch.isnan(param).sum().item())
                        inf_count = int(torch.isinf(param).sum().item())
                        raise RuntimeError(
                            "[Stage3] Non-finite model parameter after optimizer.step(). "
                            f"epoch={epoch + 1}, batch={num_batches + 1}, "
                            f"param_index={param_index}, NaN={nan_count}, Inf={inf_count}"
                        )

                # 每个 batch 打印一次关键数值，方便判断 clipping 是否有效。
                # print(
                #     f"[Stage3] epoch={epoch + 1}, batch={num_batches + 1}, "
                #     f"L_unlearn={float(unlearn_loss.item()):.6e}, "
                #     f"L_drift={float(drift_loss.item()):.6e}, "
                #     f"||G||_before={grad_norm_before_clip:.6e}, "
                #     f"||G||_after={final_grad_norm:.6e}, "
                #     f"clip_coef={clip_coef:.6e}, "
                #     f"cos={cosine:.6e}, "
                #     f"mask_keep={keep_ratio:.4%}"
                # )

                # ====================================================
                # 统计调试信息
                # ====================================================
                epoch_hard += float(hard_loss.item())
                epoch_confusion += float(confusion_loss.item())
                epoch_distill += float(distillation_loss.item())
                epoch_unlearn += float(unlearn_loss.item())
                epoch_drift += float(drift_loss.item())
                epoch_cosine += cosine
                epoch_mask_keep += keep_ratio
                epoch_grad_norm_before_clip += grad_norm_before_clip
                epoch_final_grad_norm += final_grad_norm
                epoch_clip_coef += clip_coef
                conflict_batches += int(has_conflict)
                num_batches += 1

            denom = max(num_batches, 1)

            stats = {
                "epoch": epoch + 1,
                "hard_loss": epoch_hard / denom,
                "confusion_loss": epoch_confusion / denom,
                "distillation_loss": epoch_distill / denom,
                "unlearn_loss": epoch_unlearn / denom,
                "drift_loss": epoch_drift / denom,
                "cosine": epoch_cosine / denom,
                "mask_keep_ratio": epoch_mask_keep / denom,
                "grad_norm_before_clip": epoch_grad_norm_before_clip / denom,
                "final_grad_norm": epoch_final_grad_norm / denom,
                "clip_coefficient": epoch_clip_coef / denom,
                "clip_ratio": clipped_batches / denom,
                "conflict_ratio": conflict_batches / denom,
            }

            self.stage3_history.append(stats)

            print("\n" + "-" * 70)
            print(
                f"[Stage 3] Epoch {epoch + 1}/{epochs}"
            )
            print(
                f"  L_hard          = {stats['hard_loss']:.6f}"
            )
            print(
                f"  L_confusion     = {stats['confusion_loss']:.6f}"
            )
            print(
                f"  L_distillation  = {stats['distillation_loss']:.6f}"
            )
            print(
                f"  L_unlearn       = {stats['unlearn_loss']:.6f}"
            )
            print(
                f"  L_drift         = {stats['drift_loss']:.6f}"
            )
            print(
                f"  cos(g_f,g'_r)   = {stats['cosine']:.6f}"
            )
            print(
                f"  mask keep ratio = {stats['mask_keep_ratio']:.4%}"
            )
            print(
                f"  conflict ratio  = {stats['conflict_ratio']:.4%}"
            )
            print(
                f"  ||G|| before    = {stats['grad_norm_before_clip']:.6f}"
            )
            print(
                f"  ||G|| after     = {stats['final_grad_norm']:.6f}"
            )
            print(
                f"  clip coefficient= {stats['clip_coefficient']:.6e}"
            )
            print(
                f"  clipped batches = {stats['clip_ratio']:.4%}"
            )

        torch.save(
            {
                "model_state_dict": self.global_model.state_dict(),
                "stage3_history": self.stage3_history,
                "mu_c": mu_c,
                "mu_d": mu_d,
                "temperature": temperature,
                "gradient_mask_pi": pi,
                "grad_clip_norm": grad_clip_norm,
                "unlearn_lr": unlearn_lr,
                "epochs": epochs,
            },
            "jellyfish_stage3_checkpoint.pt"
        )

        print("\n" + "=" * 70)
        print(" Jellyfish Stage 3 Finished ")
        print("=" * 70)



    # ================================================================
    # Stage 4: Repair (论文 Section 4.5 / Algorithm 1 lines 21-28)
    # ================================================================

    def _get_repair_candidate_clients(self):
        """
        返回可以参与 Repair 的“剩余客户端”。

        对于当前工程常见的 client-level unlearning：
        - self.unlearning_clients 对应需要被删除/遗忘的客户端；
        - Repair 只能使用其余客户端的 remaining data D_r；
        - 因此显式排除 unlearning client id，避免任何 forgotten raw data
          或其派生 repair proxy 被重新引入模型。
        """
        forgotten_ids = {
            client.id
            for client in getattr(self, "unlearning_clients", [])
        }

        return [
            client
            for client in self.clients
            if client.id not in forgotten_ids
        ]

    def collect_remaining_client_accuracies(
            self,
            model=None,
            clients=None,
            verbose=False
    ):
        """
        计算论文 Eq.(23) 所需的每客户端本地 remaining-data accuracy。

        论文定义：
            DeltaAcc_i =
                (Acc_t^i - Acc_{t+1}^i) / Acc_t^i * 100%

        这里返回 0~1 范围的 accuracy；
        后续计算 drop ratio 时同样使用 0~1，所以数学上完全等价。
        """
        if model is None:
            model = self.global_model

        if clients is None:
            clients = self._get_repair_candidate_clients()

        results = {}

        for client in clients:
            acc = client.evaluate_local_accuracy(
                model=model
            )
            results[client.id] = float(acc)

            if verbose:
                print(
                    f"[Stage4 Eq.23] Client {client.id} "
                    f"remaining accuracy = {acc:.4%}"
                )

        return results

    def select_repair_clients(self):
        """
        根据论文 Eq.(23) + threshold delta 选择需要触发 Repair 的客户端。

        delta_threshold:
            默认 0.05，表示相对准确率下降超过 5%。
            论文只定义了 delta 的作用，没有在可见实验设置中给出固定数值，
            因此这里保留为可配置工程参数。
        """
        candidates = self._get_repair_candidate_clients()

        if len(candidates) == 0:
            return [], {}

        if len(self.pre_unlearning_client_acc) == 0:
            raise RuntimeError(
                "Stage4 requires pre_unlearning_client_acc for Eq.(23), "
                "but no baseline was recorded."
            )

        current_acc = self.collect_remaining_client_accuracies(
            model=self.global_model,
            clients=candidates,
            verbose=False
        )

        delta_threshold = float(
            getattr(
                self.args,
                "delta_threshold",
                getattr(self.args, "repair_delta_threshold", 0.05)
            )
        )

        if delta_threshold < 0:
            raise ValueError(
                f"delta_threshold must be >= 0, got {delta_threshold}"
            )

        triggered_clients = []
        drop_info = {}

        # print("\n" + "=" * 70)
        # print(" Jellyfish Stage 4.1: Repair Trigger Check (Eq.23) ")
        # print("=" * 70)
        # print(
        #     f"repair delta threshold = {delta_threshold:.2%}"
        # )

        for client in candidates:
            before = float(
                self.pre_unlearning_client_acc.get(
                    client.id,
                    0.0
                )
            )
            after = float(
                current_acc.get(
                    client.id,
                    0.0
                )
            )

            # baseline 为 0 时 Eq.(23) 无法稳定定义。
            # 这种极端情况不自动触发，避免除零。
            if before <= 1e-12:
                drop_ratio = 0.0
            else:
                drop_ratio = (
                    (before - after)
                    / before
                )

            # 准确率上升时 drop_ratio 可能为负；
            # 论文只关心性能下降，因此保留原值用于日志，
            # 但不会触发 Repair。
            should_repair = (
                drop_ratio > delta_threshold
            )

            drop_info[client.id] = {
                "before_acc": before,
                "after_acc": after,
                "drop_ratio": drop_ratio,
                "triggered": should_repair,
            }

            # print(
            #     f"Client {client.id}: "
            #     f"before={before:.4%}, "
            #     f"after={after:.4%}, "
            #     f"DeltaAcc={drop_ratio:.4%}, "
            #     f"repair={'YES' if should_repair else 'NO'}"
            # )

            if should_repair:
                triggered_clients.append(client)

        return triggered_clients, drop_info

    def generate_repair_proxy_noise(self, repair_clients):
        """
        论文 Section 4.5 / Eq.(24):

        对触发修复的客户端 i，使用其 remaining dataset D_r^i
        通过 error-minimization noise 生成 N_r^i。

        Server 只能收到：
            - N_r^i
            - y_r^i
            - remaining sample count

        不收集客户端原始 D_r。
        """
        if len(repair_clients) == 0:
            return [], [], []

        # print("\n" + "=" * 70)
        # print(" Jellyfish Stage 4.2: Generate Remaining Proxy N_r ")
        # print("=" * 70)

        client_noises_list = []
        client_labels_list = []
        client_remaining_sizes = []

        # 论文先将 unlearned global model 下发给所有客户端。
        # 这里使用当前 Stage3 后的 global model 作为生成 N_r 的目标模型。
        self.send_models()

        for client in repair_clients:
            # print(
            #     f"\nClient {client.id} generating repair proxy N_r ..."
            # )

            noises, labels, remaining_size = (
                client.generate_repair_noise(
                    global_model=self.original_model
                )
            )

            client_noises_list.append(noises)
            client_labels_list.append(labels)
            client_remaining_sizes.append(
                int(remaining_size)
            )

        return (
            client_noises_list,
            client_labels_list,
            client_remaining_sizes
        )

    def aggregate_repair_proxy_noise(
            self,
            client_noises_list,
            client_labels_list,
            client_remaining_sizes
    ):
        """
        论文 4.5 “Aggregation of N_r”。

        论文要求各客户端贡献按 |D_r^i| 加权。

        当前实现中 generate_for_client 对本地训练数据中的每一个样本
        生成一个 repair proxy，因此：
            client proxy count ~= |D_r^i|

        将所有 proxy 在样本维度做 union/concat，再均匀 shuffle + batch，
        就等价于让客户端 i 在全局 repair objective 中的期望贡献比例约为：
            |D_r^i| / sum_k |D_r^k|

        这样避免了“像素级 weighted average”破坏不同类别 proxy 的激活结构。
        """
        (
            noises,
            labels,
            aggregation_info
        ) = aggregate_repair_noises(
            client_noises_list,
            client_labels_list,
            client_remaining_sizes
        )

        self.aggregated_repair_noises = noises
        self.aggregated_repair_labels = labels

        self.repair_noise_loader = (
            create_noise_dataloader(
                noises,
                labels,
                batch_size=self.batch_size,
                shuffle=True
            )
        )

        # print(
        #     "Global repair proxy N_r shape:",
        #     tuple(noises.shape)
        # )

        return noises, labels, aggregation_info

    def save_repair_proxy_noise(self):
        """
        保存 Stage 4 聚合后的 N_r，便于复现实验。
        """
        if self.aggregated_repair_noises is None:
            raise RuntimeError(
                "No aggregated repair proxy noise."
            )

        client_ids = "_".join(
            str(client_id)
            for client_id in self.repair_triggered_client_ids
        )

        save_path = (
            f"repair_proxy_noise_"
            f"{self.dataset}_"
            f"clients_{client_ids}.pt"
        )

        save_proxy_noise(
            self.aggregated_repair_noises,
            self.aggregated_repair_labels,
            save_path
        )

        # print(
        #     "Repair proxy noise saved:",
        #     save_path
        # )

    def repair_global_model(self):
        """
        论文 Algorithm 1, lines 21-28:

            Repair(optional):
                D_r <- N_r list
                for epoch = 1 ... E_re:
                    for batch in D_r:
                        x_r, y_r <- D_r
                        L_repair <- MSE(M(x_r), y_r)
                        omega_t <- omega_t - mu_re * grad L_repair

        实现说明：
        - PyTorch 分类模型输出通常是 logits；
        - 论文 M(x_r) 表示预测概率分布；
        - 因此先 softmax(logits)，再与 one-hot(y_r) 做 MSE。
        """
        if self.repair_noise_loader is None:
            raise RuntimeError(
                "Repair proxy loader is empty."
            )

        repair_epochs = int(
            getattr(
                self.args,
                "repair_epochs",
                20
            )
        )

        repair_lr = float(
            getattr(
                self.args,
                "repair_lr",
                0.005
            )
        )

        # 论文正文在 4.5 中将聚合后的 N_r 关联到 Eq.(24) 的 CE，
        # 而 Algorithm 1 lines 21-28 又明确写 L_repair = MSE(M(x_r), y_r)。
        # 默认按 Algorithm 1 使用 mse；保留 ce 作为论文文本一致性对照。
        repair_loss_type = str(
            getattr(
                self.args,
                "repair_loss_type",
                "mse"
            )
        ).lower()

        if repair_loss_type not in {
            "mse",
            "ce"
        }:
            raise ValueError(
                "repair_loss_type must be 'mse' or 'ce', "
                f"got {repair_loss_type}"
            )

        if repair_epochs <= 0:
            raise ValueError(
                f"repair_epochs must be > 0, got {repair_epochs}"
            )

        if repair_lr <= 0:
            raise ValueError(
                f"repair_lr must be > 0, got {repair_lr}"
            )

        # Repair 更新全局模型 omega_t。
        for p in self.global_model.parameters():
            p.requires_grad_(True)

        self.global_model.train()

        # 工程稳定措施：
        # N_r 是合成 proxy，不让它改写 BN running statistics。
        # BN 的 weight / bias 仍然可求导并参与 repair。
        for module in self.global_model.modules():
            if isinstance(
                    module,
                    nn.modules.batchnorm._BatchNorm
            ):
                module.eval()

        # Algorithm 1 是直接梯度下降形式，使用 SGD 与公式最直观对应。
        optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=repair_lr
        )

        self.stage4_history = []

        print("\n" + "=" * 70)
        print(" Jellyfish Stage 4.3: Global Model Repair ")
        print("=" * 70)
        print(
            f"repair epochs = {repair_epochs}"
        )
        print(
            f"repair lr     = {repair_lr}"
        )
        print(
            f"repair batches= {len(self.repair_noise_loader)}"
        )
        print(
            f"repair loss   = {repair_loss_type}"
        )

        for epoch in range(repair_epochs):
            total_loss = 0.0
            num_batches = 0

            for x_r, y_r in self.repair_noise_loader:
                x_r = x_r.to(self.device)
                y_r = y_r.to(self.device)

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = self.global_model(
                    x_r
                )

                if repair_loss_type == "mse":
                    # Algorithm 1:
                    # L_repair = MSE(M(x_r), y_r)
                    # 将 M(x_r) 解释为类别概率分布，
                    # y_r 转为 one-hot。
                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    )

                    y_one_hot = F.one_hot(
                        y_r,
                        num_classes=probabilities.shape[1]
                    ).to(
                        dtype=probabilities.dtype
                    )

                    repair_loss = F.mse_loss(
                        probabilities,
                        y_one_hot
                    )
                else:
                    # Section 4.5 文本 / Eq.(24) 的 CE 对照实现。
                    repair_loss = F.cross_entropy(
                        logits,
                        y_r
                    )

                if not torch.isfinite(repair_loss):
                    raise RuntimeError(
                        "[Stage4] Non-finite repair loss detected. "
                        f"epoch={epoch + 1}, batch={num_batches + 1}"
                    )

                repair_loss.backward()
                optimizer.step()

                # 更新后立即检查参数。
                for param_index, param in enumerate(
                        self.global_model.parameters()
                ):
                    if not torch.isfinite(param).all():
                        raise RuntimeError(
                            "[Stage4] Non-finite model parameter after repair step. "
                            f"epoch={epoch + 1}, batch={num_batches + 1}, "
                            f"param_index={param_index}"
                        )

                total_loss += float(
                    repair_loss.item()
                )
                num_batches += 1

            mean_loss = (
                total_loss
                / max(num_batches, 1)
            )

            # 仅记录 triggered clients 的 remaining accuracy。
            repair_clients = [
                client
                for client in self._get_repair_candidate_clients()
                if client.id in self.repair_triggered_client_ids
            ]

            local_acc = self.collect_remaining_client_accuracies(
                model=self.global_model,
                clients=repair_clients,
                verbose=False
            )

            # evaluate_local_accuracy 会把模型切到 eval；
            # 如果还有下一轮 Repair，需要恢复 train，
            # 同时再次冻结 BN running statistics。
            if epoch + 1 < repair_epochs:
                self.global_model.train()
                for module in self.global_model.modules():
                    if isinstance(
                            module,
                            nn.modules.batchnorm._BatchNorm
                    ):
                        module.eval()

            mean_local_acc = (
                sum(local_acc.values())
                / max(len(local_acc), 1)
            )

            stats = {
                "epoch": epoch + 1,
                "repair_loss": mean_loss,
                "mean_triggered_client_acc": mean_local_acc,
                "triggered_client_acc": local_acc,
            }

            self.stage4_history.append(
                stats
            )

            # print(
            #     f"[Stage 4] Epoch {epoch + 1}/{repair_epochs}, "
            #     f"L_repair={mean_loss:.6f}, "
            #     f"mean triggered-client acc={mean_local_acc:.4%}"
            # )

        self.global_model.eval()

        torch.save(
            {
                "model_state_dict": self.global_model.state_dict(),
                "stage4_history": self.stage4_history,
                "repair_lr": repair_lr,
                "repair_epochs": repair_epochs,
                "repair_loss_type": repair_loss_type,
                "repair_triggered_client_ids": self.repair_triggered_client_ids,
            },
            "jellyfish_stage4_repair_only.pt"
        )

    def model_repair(self):
        """
        Stage 4 总入口。

        严格按论文 4.5 的逻辑：
        1. Eq.(23) 检查每个 remaining client 的相对准确率下降；
        2. 只有 DeltaAcc_i > delta 的客户端发起一次 Repair；
        3. 客户端生成 N_r^i（Eq.24）；
        4. 服务端按 remaining-data 规模聚合 N_r；
        5. Algorithm 1 使用 MSE 做一次全局 Repair；
        6. 修复后的 global model 再广播给所有客户端。

        如果没有客户端超过 threshold，本次 Stage 4 直接跳过。
        """
        if bool(
                getattr(
                    self.args,
                    "disable_repair",
                    False
                )
        ):
            # print(
            #     "\n[Stage4] Repair disabled by args.disable_repair."
            # )
            self.stage4_history = [
                {
                    "skipped": True,
                    "reason": "disable_repair",
                }
            ]
            return

        repair_clients, drop_info = (
            self.select_repair_clients()
        )

        self.repair_triggered_client_ids = [
            client.id
            for client in repair_clients
        ]

        if len(repair_clients) == 0:
            # print(
            #     "\n[Stage4] No client exceeds repair threshold. "
            #     "Repair is skipped."
            # )
            self.stage4_history = [
                {
                    "skipped": True,
                    "drop_info": drop_info,
                }
            ]
            return

        (
            client_noises_list,
            client_labels_list,
            client_remaining_sizes
        ) = self.generate_repair_proxy_noise(
            repair_clients
        )

        (
            _,
            _,
            aggregation_info
        ) = self.aggregate_repair_proxy_noise(
            client_noises_list,
            client_labels_list,
            client_remaining_sizes
        )

        self.save_repair_proxy_noise()

        self.repair_global_model()

        # 把 Eq.(23) 触发信息和聚合信息并入历史记录，便于复现实验。
        if len(self.stage4_history) > 0:
            self.stage4_history[0][
                "trigger_drop_info"
            ] = drop_info
            self.stage4_history[0][
                "aggregation_info"
            ] = aggregation_info

        self.send_models()

        # print("\n" + "=" * 70)
        # print(" Jellyfish Stage 4 Repair Finished ")
        # print("=" * 70)


