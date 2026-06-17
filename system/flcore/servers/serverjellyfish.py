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
from utils.noise_utils import aggregate_client_noises, create_noise_dataloader, save_proxy_noise
from utils.attack_utils import attack, train_attack_model


class Jellyfish(Server):
    """
    Jellyfish: 零样本联邦遗忘学习框架 (论文 4.3 节与 4.4 节全量落地版)

    四大阶段:
      ① 代理数据生成 (Noise Generation)      -- 客户端本地
      ② 知识解耦 (Knowledge Disentanglement)  -- 服务端
      ③ 多目标联合遗忘 (Joint Unlearning)     -- 服务端
      ④ 零样本模型修复 (Model Repair)          -- 可选
    """

    def __init__(self, args):
        super().__init__(args)

        self.set_slow_clients()
        self.set_clients(clientJellyfish)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.unlearn_Budget = []

        # Jellyfish 特有资产
        self.original_model = None  # 遗忘前的模型副本 (\omega^t)
        self.proxy_noise_loader = None  # 聚合后的代理噪声 DataLoader (N_f)
        self.aggregated_noises = None  # 聚合后的代理噪声 tensor
        self.aggregated_labels = None  # 对应标签

        # 4.3 节超参数
        self.alpha_dis = getattr(args, 'alpha', 0.9)  # 通道保留比例 alpha，论文默认 0.9
        self.dis_epochs = getattr(args, 'dis_epochs', 5)  # 解耦迭代 epoch 数 E_dis，论文默认 5

        # 4.4 节超参数动态挂载 (严格对齐论文第 16、17、23 页实验参数设置)
        self.unlearn_epochs = getattr(args, 'unlearn_epochs', 3)  # 遗忘迭代周期 E_un
        self.unlearn_lr = getattr(args, 'unlearn_lr', 0.01)  # 遗忘学习率 \mu_un
        self.mu_c = getattr(args, 'mu_c', 0.5)  # 混淆损失超参数 \mu_c
        self.mu_d = getattr(args, 'mu_d', 0.5)  # 蒸馏损失超参数 \mu_d
        self.distill_temp = getattr(args, 'distill_temp', 4.0)  # 蒸馏温度 Temp
        self.pi_mask = getattr(args, 'pi_mask', 1e-3)  # 梯度掩蔽阈值 \pi
        self.num_bad_teachers = getattr(args, 'num_teachers', 3)  # 劣质教师数量 N_T

        # 4.5 节修复超参数动态挂载 (对齐论文第 20 页与 23 页实验参数设置)
        self.delta_threshold = getattr(args, 'delta_threshold', 0.05)  # 精度下降触发门限 \delta (默认 5%)
        self.repair_epochs = getattr(args, 'repair_epochs', 2)  # 修复微调 epoch 数 (默认 2)
        self.repair_lr = getattr(args, 'repair_lr', 0.005)  # 修复学习率

        # 预训练参数持久化路径（对齐基类 save_global_model 的路径格式）
        # 当 attack='False' 且 learning_state!='retrain' 时，路径为 models_seed{X}_resnet/{dataset}/_server.pt
        self.pretrain_model_path = os.path.join(
            "models_seed" + str(self.args.seed_num) + "_resnet",
            self.dataset,
            "_server.pt"
        )

    def train(self):
        """标准联邦预训练阶段 (Learning Phase)"""
        # if os.path.exists(self.pretrain_model_path):
        #     print(f"\n[Learning Skip] 检测到已存在历史 Learning 阶段结果: {self.pretrain_model_path}")
        #     print(">>> 正在直接跳过常规预训练，进入阶段①代理数据集生产流...")
        #     return

        print("\n" + "=" * 50)
        print("Starting Learning Phase (Standard FedAvg Pre-training)...")
        print("=" * 50)

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy during Learning Phase:")
        print(max(self.rs_test_acc))

        # ======= 修复代码：在此处训练攻击者模型，规避 NotFittedError =======
        print("\n[MIA] Training Attacker model for baseline evaluation...")
        self.attacker = train_attack_model(self.global_model, self.clients, self.num_classes, self.device)

        # 打印未遗忘前的攻击基准
        (PRE_old, REC_old) = attack(self.global_model, self.attacker, self.unlearning_clients, self.num_classes,
                                    self.device)
        print("MIA Attacker to old model precision = {:.4f}".format(PRE_old))
        print("MIA Attacker to old model recall = {:.4f}".format(REC_old))
        # ===============================================================

        self.save_results()
        # 此时调用基类方法，将可以安全、完美地持久化 global_model 和已训练的 attacker
        self.save_global_model()
        print(f"[Learning Complete] 原始收敛权重与攻击者快照已成功固化至本地文件。")

    # def unlearning(self):
    #     """
    #     Jellyfish 遗忘流程控制中心 (对齐 BatchNorm 与 warm_up 完美版)
    #     """
    #     print("\n" + "=" * 50)
    #     print("Initializing Jellyfish Unlearning Phase Flow...")
    #     print("=" * 50)
    #
    #     # 1. 调用基类内置方法全量重载大模型快照与攻击者模型
    #     self.load_model()
    #
    #     # 2. 关键修复：先将最新的全局模型参数同步分发给所有的本地客户端
    #     print("[Sync] 正在将重载的全局权重下发至各客户端实体...")
    #     self.send_models()
    #
    #     # 3. 关键修复：执行 warm_up，利用本地数据流冲刷并纠正各端 BN 层的 running_mean/var
    #     print("[BN Warm-up] 正在触发端侧热身，对齐 BatchNorm 滚动统计量...")
    #     self.warm_up()
    #
    #     # 4. 关键保护：此时参数已完美对齐，锁死为评估模式准备纯推理测试
    #     self.global_model.eval()
    #     for client in self.clients:
    #         client.model.eval()
    #
    #     # 5. 备份原始大模型快照 \omega^t
    #     self.original_model = copy.deepcopy(self.global_model)
    #     self.original_model.eval()
    #
    #     # 6. 当场测量 Utility 性能
    #     print("\n[验证] 正在校验初始载入权重的 Utility 性能...")
    #     self.evaluate()
    #
    #     # 7. 严格复刻参考算法的 MIA 攻击基准评估逻辑
    #     from utils.attack_utils import attack
    #
    #     print("\n[MIA Evaluation] 正在使用重载的 Attacker 评估未遗忘前的隐私抗性...")
    #     (PRE_unlearning, REC_unlearning) = attack(
    #         self.global_model,
    #         self.attacker,
    #         self.clients,
    #         self.num_classes,
    #         self.device
    #     )
    #
    #     print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
    #     print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
    #
    #     # 8. 持久化遗忘阶段的指标结果
    #     self.save_unlearning(PRE_unlearning)
    #
    #     print("\n" + "=" * 50)
    #     print("[Jellyfish Unlearning Done] 已成功通过基类管道对齐历史指标。")
    #     print("=" * 50)

    def unlearning(self):
        """
        Jellyfish 遗忘主流程控制中心
        """
        print("\n" + "=" * 50)
        print("Initializing Jellyfish Unlearning Phase Flow...")
        print("=" * 50)

        # 1. 调用基类内置方法全量重载大模型快照与攻击者模型
        self.load_model()

        # 2. 关键修复：先将最新的全局模型参数同步分发给所有的本地客户端
        print("[Sync] 正在将重载的全局权重下发至各客户端实体...")
        self.send_models()

        # 3. 关键修复：执行 warm_up，利用本地数据流冲刷并纠正各端 BN 层的 running_mean/var
        print("[BN Warm-up] 对齐 BatchNorm 统计量...")
        self.warm_up()

        # 4. 关键保护：此时参数已完美对齐，锁死为评估模式准备纯推理测试
        self.global_model.eval()
        for client in self.clients:
            client.model.eval()

        # 5. 备份原始大模型快照 \omega^t
        self.original_model = copy.deepcopy(self.global_model)
        self.original_model.eval()

        # 6. 当场测量 Utility 性能
        print("\n[验证] 正在校验初始载入权重的 Utility 性能...")
        self.evaluate()


        # # ====== 阶段①: 代理数据生成与服务器重洗整合 ======
        print("\nPhase ①: Triggering Local Proxy Noise Generation & Server Assembly...")
        self.collect_proxy_noise()
        #
        # # ====== 阶段②: 知识解耦 (论文 4.3 节核心) ======
        print("\nPhase ②: Entering Server-Side Knowledge Disentanglement...")
        self.fit_knowledge_disentanglement()
        # 监控点 1：测量知识解耦后的精度演进
        self.send_models()  # 必须先将解耦后的 base 层权重同步给客户端
        self.global_model.eval()
        for c in self.clients: c.model.eval()
        print("\n>>> [测量 1] 阶段②知识解耦执行完成后测量结果:")
        self.evaluate()
        #
        # # ====== 阶段③: 多目标联合遗忘 (论文 4.4 节核心) ======
        print("\nPhase ③: Entering Multi-Objective Joint Unlearning Trajectory...")
        self.fit_joint_unlearning()
        # 监控点 2：测量执行完多目标冲突梯度消除后的精度演进
        self.send_models()  # 必须把阶段③手术修正后的参数灌入客户端
        self.global_model.eval()
        for c in self.clients: c.model.eval()
        print("\n>>> [测量 2] 阶段③多目标联合遗忘执行完成后测量结果:")
        self.evaluate()
        #
        # # ====== 阶段④: 零样本模型修复 (严格对齐论文 4.5 节) ======
        print("\nPhase ④: Entering Zero-Shot Adaptive Model Repair Stage...")
        self.adaptive_model_repair()

        print("\n[Final Sync] 整个 Jellyfish 框架全量演进结束，正在同步最终模型...")
        self.send_models()
        self.send_models_target()

        self.global_model.eval()
        for c in self.clients: c.model.eval()

        print("\n>>> [测量 3 - 最终结果] 零样本自适应修复微调后最终测量结果:")
        self.evaluate()

        # 评估最终的成员推理攻击 MIA
        print("\n[MIA Evaluation] 正在评估最终模型的隐私抗性...")
        (PRE_unlearning, REC_unlearning) = attack(
            self.global_model,
            self.attacker,
            self.clients,
            self.num_classes,
            self.device
        )

        print("MIA Attacker to unlearning model precision = {:.4f}".format(PRE_unlearning))
        print("MIA Attacker to unlearning model recall = {:.4f}".format(REC_unlearning))
        self.save_unlearning(PRE_unlearning)
        print("\n" + "=" * 50)
        print("[Jellyfish Final Flow Done] 逐步性能链条分析完毕。")
        print("=" * 50)



    def collect_proxy_noise(self):
        """
        阶段①核心：收集各遗忘客户端的代理噪声并在服务器端组装
        """
        self.send_models_target()

        client_noises_list = []
        client_labels_list = []

        for client in self.unlearning_clients:
            print(f"[Server] Requesting proxy noise from Target Client {client.id}...")
            noises, labels = client.generate_noise(
                global_model=self.global_model,
                steps=self.args.noise_steps,
                lr=self.args.noise_lr
            )
            client_noises_list.append(noises)
            client_labels_list.append(labels)

            total_client_samples = sum([n.shape[0] for n in noises])
            print(f"[Server] 接收成功：Client {client.id} 已成功上传 {total_client_samples} 个脱敏代理样本")

        # 合并构建统一噪声矩阵集合 N_f (对应论文公式 4)
        self.aggregated_noises, self.aggregated_labels = aggregate_client_noises(
            client_noises_list, client_labels_list
        )

        # 组装全局洗牌后的训练 Dataloader
        self.proxy_noise_loader = create_noise_dataloader(
            self.aggregated_noises,
            self.aggregated_labels,
            batch_size=self.batch_size,
            shuffle=True
        )

        print(f"[Server] Global Proxy Dataset N_f 串联组装完毕 (总样本数: {self.aggregated_noises.shape[0]}).")

        # 固化备份
        save_path = f"proxy_noise_{self.dataset}_clients_{'_'.join(map(str, [c.id for c in self.unlearning_clients]))}.pt"
        save_proxy_noise(self.aggregated_noises, self.aggregated_labels, save_path)

    def fit_knowledge_disentanglement(self):
        """
        阶段②算法落地：通过在前向传播中截获特征图并压制弱激活通道实现跨类解耦。
        """
        self.global_model.head.requires_grad_(False)
        self.global_model.base.requires_grad_(True)

        target_conv_layer = None
        for name, module in self.global_model.base.named_modules():
            if isinstance(module, nn.Conv2d):
                target_conv_layer = module

        if target_conv_layer is None:
            raise RuntimeError("[Disentangle Error] 骨干网络中未检测到合规的 Conv2d 激活层。")

        F_conv_container = []

        def forward_hook_fn(module, input, output):
            F_conv_container.append(output)

        hook_handle = target_conv_layer.register_forward_hook(forward_hook_fn)

        dis_lr = getattr(self.args, 'local_learning_rate', 0.01)
        # optimizer_dis = torch.optim.Adam(self.global_model.base.parameters(), lr=dis_lr)
        optimizer_dis = torch.optim.SGD(
            self.global_model.base.parameters(),
            lr=1e-4,
            momentum=0.9,
            weight_decay=1e-4
        )

        self.global_model.train()
        for epoch in range(self.dis_epochs):
            loss_all = 0.0
            for batch_idx, (x, y) in enumerate(self.proxy_noise_loader):
                x, y = x.to(self.device), y.to(self.device)
                F_conv_container.clear()

                _ = self.global_model(x)

                if len(F_conv_container) == 0:
                    raise ValueError("[Disentangle Error] 特征图截获探针未被成功触发！")

                F_conv = F_conv_container[0]
                num_channels = F_conv.shape[1]

                # 公式 (7)：计算各个通道的 L1 范数
                channel_norms = torch.mean(torch.abs(F_conv), dim=[0, 2, 3])

                # 公式 (5)：计算分割线 Thr
                k_idx = max(1, int((1.0 - self.alpha_dis) * num_channels))
                Thr = torch.kthvalue(channel_norms, k_idx).values

                # 公式 (7) 分流：保留小于 Thr 的部分进行惩罚，其余（主力通道）置 0 保护
                norms_filtered = torch.where(channel_norms < Thr, channel_norms, torch.zeros_like(channel_norms))

                # 公式 (6)：解耦损失计算
                denominator = (1.0 - self.alpha_dis) * num_channels + 1e-8
                loss_disentangle = torch.sum(norms_filtered) / denominator

                optimizer_dis.zero_grad()
                loss_disentangle.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.base.parameters(), max_norm=1.0)  # 🌟 物理防爆
                optimizer_dis.step()


                loss_all += loss_disentangle.item()

            mean_epoch_loss = loss_all / len(self.proxy_noise_loader)
            print(f"    Epoch [{epoch + 1}/{self.dis_epochs}] - Disentangle Mean Loss: {mean_epoch_loss:.6f}")

        hook_handle.remove()
        print("  [Disentangle Engine] 知识解耦阶段完成。跨类别纠缠特征已被物理压制清零。")

    def fit_joint_unlearning(self):
        """
        阶段③全量落地：严格按照论文 4.4 节公式组装核心遗忘训练引擎（集成多目标损失、梯度掩蔽与梯度协调）
        """
        # 全量激活所有参数，进行端到端全量微调更新
        for param in self.global_model.parameters():
            param.requires_grad = True

        # 1. 产生不合格教师网络集合 T_set (严格遵循公式 16 机制，通过随机化产生 N_T 个全盲教师)
        bad_teachers = []
        for t_idx in range(self.num_bad_teachers):
            teacher = copy.deepcopy(self.global_model)
            # 通过对最后一层分类头引入重度扰动高斯随机化破坏，使其对遗忘类吐出杂乱、均匀的 Logits 得分 v_j
            for param in teacher.head.parameters():
                param.data.copy_(torch.randn_like(param.data) * 2.0)
            teacher.eval()
            bad_teachers.append(teacher)

        # 2. 离线精准抽取敏感参数梯度，生产二值化梯度掩码 m_s (严格对齐公式 21)
        print("  [Unlearn Engine] Pre-calculating binary gradient mask m_s...")
        gradient_mask = {}
        for name, param in self.global_model.named_parameters():
            gradient_mask[name] = torch.zeros_like(param.data)

        # 使用干净的原始大模型计算遗忘数据样本点上的基础交叉熵，求取偏导敏感度
        self.original_model.zero_grad()
        mask_counter = 0
        for x_m, y_m in self.proxy_noise_loader:
            x_m, y_m = x_m.to(self.device), y_m.to(self.device)
            out_m = self.original_model(x_m)
            loss_m = nn.CrossEntropyLoss()(out_m, y_m)
            loss_m.backward()
            mask_counter += len(y_m)

        # 依据敏感度阈值 \pi 实施二值硬切分
        with torch.no_grad():
            for name, param in self.original_model.named_parameters():
                if param.grad is not None:
                    # 绝对值小于 \pi 设为 1 (安全区)；大于等于 \pi 设为 0 (高度纠缠的隐私敏感重灾区)
                    avg_grad = param.grad / float(mask_counter)
                    gradient_mask[name] = (torch.abs(avg_grad) < self.pi_mask).float().to(self.device)
                else:
                    gradient_mask[name] = torch.ones_like(param.data).to(self.device)

        # 3. 实例化多目标遗忘的核心闭环执行引擎
        optimizer_unlearn = torch.optim.Adam(self.global_model.parameters(), lr=self.unlearn_lr)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        print(f"  [Unlearn Engine] Running Joint Multi-Objective Unlearning for {self.unlearn_epochs} epochs...")

        self.global_model.train()
        for epoch in range(self.unlearn_epochs):
            loss_unlearn_all = 0.0
            loss_drift_all = 0.0

            for batch_idx, (x, y) in enumerate(self.proxy_noise_loader):
                x, y = x.to(self.device), y.to(self.device)

                # =========================================================
                # 步骤 A: 分流计算遗忘任务梯度 g_f (对齐公式 8, 9, 12, 16)
                # =========================================================
                optimizer_unlearn.zero_grad()
                outputs = self.global_model(x)

                # ① Hard Loss (公式 9): 反向最大化正确类别的损失，击碎原判决边界
                loss_hard = -criterion_ce(outputs, y)

                # ② Confusion Loss (公式 10, 11, 12): 定向寻找相似伪标签 y_fake 并强力拉近
                with torch.no_grad():
                    prob_m = F.softmax(outputs, dim=1)
                    # 强行将正确标签位置的概率置零，在除自身以外的剩余域中提取 argmax 最大概率目标作为 y_fake
                    prob_m.scatter_(1, y.view(-1, 1), 0.0)
                    y_fake = torch.argmax(prob_m, dim=1)

                loss_confusion = criterion_ce(outputs, y_fake)

                # ③ Distillation Loss (公式 13, 14, 15, 16): 引入 N_T 个不合格教师模型的输出提供随机性
                loss_distill = 0.0
                p_student = F.log_softmax(outputs / self.distill_temp, dim=1)

                for teacher in bad_teachers:
                    with torch.no_grad():
                        teacher_outputs = teacher(x)
                        p_teacher = F.softmax(teacher_outputs / self.distill_temp, dim=1)
                    loss_distill += criterion_kl(p_student, p_teacher) * (self.distill_temp ** 2)

                loss_distill = loss_distill / self.num_bad_teachers

                # 融合成联合任务损失 L_unlearn (公式 8)
                loss_unlearn = loss_hard + self.mu_c * loss_confusion + self.mu_d * loss_distill
                loss_unlearn.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)

                # 完美抽取并打包独立干净的遗忘梯度向量 g_f
                g_f = {}
                for name, param in self.global_model.named_parameters():
                    if param.grad is not None:
                        g_f[name] = param.grad.clone()
                    else:
                        g_f[name] = torch.zeros_like(param.data)

                # =========================================================
                # 步骤 B: 独立分流计算记忆/权重漂移任务梯度 g_r (对齐公式 17)
                # =========================================================
                optimizer_unlearn.zero_grad()

                loss_drift = 0.0
                for name, param in self.global_model.named_parameters():
                    # 严格计算对原始模型的参数漂移惩罚
                    loss_drift += 0.5 * torch.sum((param - self.original_model.state_dict()[name]) ** 2)

                loss_drift.backward()

                # 完美抽取并打包独立的记忆梯度向量 g_r，并立刻执行公式 22 的二值掩蔽过滤 (g_r' = g_r * m_s)
                g_r_primed = {}
                for name, param in self.global_model.named_parameters():
                    if param.grad is not None:
                        # 仅保留安全通配区的记忆更新，直接物理抹杀敏感区的遗忘残留
                        g_r_primed[name] = param.grad.clone() * gradient_mask[name]
                    else:
                        g_r_primed[name] = torch.zeros_like(param.data)

                # =========================================================
                # 步骤 C: 部署多任务冲突外科手术：梯度协调 (严格对齐公式 20)
                # =========================================================
                g_composite = {}
                with torch.no_grad():
                    for name, param in self.global_model.named_parameters():
                        gf_vec = g_f[name]
                        gr_vec = g_r_primed[name]

                        # 计算两个高维梯度稠密矩阵的余弦相似度
                        dot_product = torch.sum(gf_vec * gr_vec)
                        norm_f = torch.norm(gf_vec)
                        norm_r = torch.norm(gr_vec)

                        # 如果余弦夹角小于0，说明发生剧烈冲突，启动正交投影擦除
                        if dot_product < 0 and norm_r > 1e-8:
                            # 严格执行公式 (20)：从 g_f 中剔除在 g_r' 方向上的伴生负贡献分量
                            projection = (dot_product / (norm_r ** 2)) * gr_vec
                            gf_corrected = gf_vec - projection
                        else:
                            gf_corrected = gf_vec

                        # 严格执行公式 (20) 末尾的线性融合：G = g_f' + g_r'
                        g_composite[name] = gf_corrected + gr_vec

                # =========================================================
                # 步骤 D: 参数空间物理回填，手动执行常规梯度优化步骤
                # =========================================================
                optimizer_unlearn.zero_grad()
                with torch.no_grad():
                    for name, param in self.global_model.named_parameters():
                        if name in g_composite:
                            # 将外科手术式修正后的合成梯度 G 手动挂载回实体的 .grad 容器中
                            param.grad = g_composite[name].clone()

                optimizer_unlearn.step()

                loss_unlearn_all += loss_unlearn.item()
                loss_drift_all += loss_drift.item()

            mean_unlearn = loss_unlearn_all / len(self.proxy_noise_loader)
            mean_drift = loss_drift_all / len(self.proxy_noise_loader)
            print(
                f"    Epoch [{epoch + 1}/{self.unlearn_epochs}] - L_unlearn: {mean_unlearn:.4f} | L_drift: {mean_drift:.4f}")

        print("  [Unlearn Engine] 多目标联合遗忘训练完成。敏感贡献已被精准剥离。")

    def adaptive_model_repair(self):
        """
        阶段④算法全量落地：严格执行论文 4.5 节设计的零样本修复策略。
        依据各正常客户端的测试准确率退化程度，自适应触发端侧本地保留噪声构建并执行公式(23)、(24)。
        """
        print("  [Repair Stage] Assessing accuracy drop on remaining clients...")

        # 1. 过滤识别出联邦网络中未参与遗忘的正常存留客户端集合 C_r
        forget_client_ids = [c.id for c in self.unlearning_clients]
        remaining_clients = [c for c in self.clients if c.id not in forget_client_ids]

        if len(remaining_clients) == 0:
            print("  [Repair Skip] 服务器未检测到任何剩余客户端资产，无须执行性能修复。")
            return

        # 2. 依次评估各正常端在执行遗忘微调之后的性能降幅
        triggered_client_noises = []
        triggered_client_labels = []

        for client in remaining_clients:
            # 评估微调前该端基础性能 (读取预训练模型历史快照资产，对齐底层返回的 7 个指标)
            # temp_model = copy.deepcopy(self.global_model)
            # temp_model.load_state_dict(torch.load(self.pretrain_model_path, map_location=self.device))
            #  修复代码：显式指定 weights_only=False，并直接作为完整模型对象载入
            temp_model = torch.load(self.pretrain_model_path, map_location=self.device, weights_only=False)
            client.set_parameters(temp_model)
            ct_pre, ns_pre, _, _, _, _, _ = client.test_metrics()
            pre_acc = (ct_pre * 1.0) / ns_pre if ns_pre > 0 else 0.0  # 计算得到准确率

            # 评估遗忘微调之后的该端端侧精度
            client.set_parameters(self.global_model)
            ct_post, ns_post, _, _, _, _, _ = client.test_metrics()
            post_acc = (ct_post * 1.0) / ns_post if ns_post > 0 else 0.0

            acc_drop = pre_acc - post_acc
            print(
                f"    - Client {client.id}: Pre-Acc = {pre_acc * 100:.2f}%, Post-Acc = {post_acc * 100:.2f}%, Drop = {acc_drop * 100:.2f}%")

            # 严格对齐论文4.5节触发准则：如果测试准确率下降超过预定义阈值 \delta
            if acc_drop > self.delta_threshold:
                print(
                    f"      >>> [Triggered] 精度损伤达 {acc_drop * 100:.2f}% (>{self.delta_threshold * 100}%)，下发修复噪声命令...")
                # 客户端在本地按保留类别分布定制生成保留代理噪声 N_r^i
                r_noise, r_label = client.generate_retention_noise(global_model=self.global_model)
                triggered_client_noises.append(r_noise)
                triggered_client_labels.append(r_label)
            else:
                print("      >>> [Pass] 性能保持良好，免于提交修复数据。")

        # 3. 严格执行公式 (23)：如果存在触发端，对其上报的保留噪声数据集取并集进行聚合
        if len(triggered_client_noises) == 0:
            print("  [Repair Skip] 正常客户端未发生严重性能受损，模型修复流自适应熔断。")
            return

        print(
            f"  [Repair Engine] Aggregating {len(triggered_client_noises)} clients' retention datasets via Eq.(23)...")
        # 合并构建统一的全局保留噪声数据集 \mathcal{N}_r
        aggregated_r_noises = torch.cat(triggered_client_noises, dim=0)
        aggregated_r_labels = torch.cat(triggered_client_labels, dim=0)

        self.repair_noise_loader = DataLoader(
            list(zip(aggregated_r_noises, aggregated_r_labels)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        print(
            f"  [Repair Engine] Global Retention dataset \\mathcal{{N}}_r created (Total size: {aggregated_r_noises.shape[0]})")

        # 4. 严格执行公式 (24)：使用交叉熵损失函数对遗忘后的模型进行知识回填微调
        # optimizer_repair = torch.optim.Adam(self.global_model.parameters(), lr=self.repair_lr)
        #  改用带重度权重衰减的随机梯度下降，锁死参数边界
        optimizer_repair = torch.optim.SGD(
            self.global_model.parameters(),
            lr=0.001,  # 修复学习率不宜过大，降到 0.001
            momentum=0.9,
            weight_decay=0.001  # 强行用 L2 正则化把大参数拽回来
        )
        criterion_repair = nn.CrossEntropyLoss()

        print(f"  [Repair Tuning] Fine-tuning the global model for {self.repair_epochs} epochs...")
        self.global_model.train()

        for rep_epoch in range(self.repair_epochs):
            loss_rep_total = 0.0
            for r_x, r_y in self.repair_noise_loader:
                r_x, r_y = r_x.to(self.device), r_y.to(self.device)

                optimizer_repair.zero_grad()
                rep_outputs = self.global_model(r_x)

                # 计算公式 (24)：在全局保留代理噪声数据集上的交叉熵损失
                loss_repair = criterion_repair(rep_outputs, r_y)
                loss_repair.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)
                optimizer_repair.step()

                loss_rep_total += loss_repair.item()

            print(
                f"    Repair Epoch [{rep_epoch + 1}/{self.repair_epochs}] - L_repair Loss: {loss_rep_total / len(self.repair_noise_loader):.6f}")

        print("  [Repair Engine] 零样本模型修复微调结束。丢失的正常域泛化能力成功回填。")