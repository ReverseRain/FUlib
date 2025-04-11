# Plato: A New Framework for Scalable Federated Learning Research

Welcome to *Plato*, a software framework to facilitate scalable, reproducible, and extensible federated learning research. Please refer to the [Plato documentation](https://platodocs.netlify.app/) for details on installing, running and deploying Plato.



# Knot算法逻辑解析（基于《Asynchronous Federated Unlearning》论文）

## 1. 核心思想
Knot是一种面向**异步联邦学习**的**聚类聚合机制**，旨在解决联邦遗忘（Federated Unlearning）中因数据删除导致的全局模型重新训练开销问题。其核心逻辑是通过**分簇隔离**和**优化分簇策略**，将重新训练的范围限制在单个簇内。

---

## 2. 关键步骤

### 2.1 聚类聚合机制
1. **分簇隔离**  
   - 将客户端（Clients）划分为多个簇（Clusters），每个簇独立进行模型聚合。
   - 当某个客户端请求删除数据时，仅需重新训练该客户端所在簇内的模型，其他簇可继续正常训练。

2. **异步训练优势**  
   - 各簇的训练进度可独立推进（异步），避免同步等待。
   - 快速客户端（训练速度快）优先聚合，提升整体收敛速度。

### 2.2 客户端分簇优化
Knot通过**优化问题**动态分配客户端到簇，目标是最小化重新训练的时钟时间。具体分两步：

#### 2.2.1 定义匹配评分（Match Rating）
- **训练时间**（`T_k`）：客户端完成本地训练的耗时。
- **模型差异度**（`S_k`）：客户端数据分布与全局模型的差异（通过余弦相似度衡量）。
- **目标值**：为每个簇预设目标训练时间（`T̄_n`）和目标差异度（`S̄_n`）。
- **匹配评分公式**：  
  `d_{kn} = || [a(T̄_n - T_k), b(S̄_n - S_k)] ||_2`  
  （`a`和`b`为超参数，加权平衡时间和差异度）

#### 2.2.2 优化问题建模
- **目标**：最小化所有客户端与其所属簇的匹配评分，且优先降低最大评分（Lexicographical Minimization）。
- **约束条件**：
  - 每个客户端至少分配到一个簇（`∑x_{kn} ≥ 1`）。
  - 限制每个客户端分配的最大簇数（`∑x_{kn} ≤ c1`）。
  - 控制每簇的客户端数量上下限（`c2 ≤ ∑x_{kn} ≤ c3`）。

### 2.3 问题求解
1. **整数规划转线性规划**  
   - 原问题为整数词典序最小化问题，通过以下步骤转换：
     - 将匹配评分离散化为整数（`D_{kn}`）。
     - 目标函数转换为可分离凸函数：`min Σ (KN)^{D_{kn}x_{kn}}`。
     - 证明约束矩阵为全单模（Totally Unimodular），确保LP解为整数。
   - 最终使用现成LP求解器（如Mosek）高效求解。

2. **分簇结果应用**  
   - 求解后得到客户端-簇分配矩阵`x_{kn}`，指导实际训练中的聚合范围。

---

## 3. 算法优势
1. **降低重新训练成本**  
   - 数据删除时仅需重新训练单个簇，而非全局客户端。
2. **兼容异步训练**  
   - 各簇独立推进训练，避免同步阻塞。
3. **理论保证**  
   - 通过线性规划高效求解最优分簇，实验显示相比随机分簇提升显著。

---

## 4. 实验验证
- **数据集**：CIFAR-10、FEMNIST、Purchase-100、Tiny Shakespeare。
- **结果**：
  - 相比基线（如FedBuff），Knot减少85%的时钟时间。
  - 与近似算法（如FedEraser）相比，Knot完全符合GDPR要求（彻底删除数据痕迹）。

---

## 5. 总结
Knot通过**分簇隔离**和**优化客户端分配**，在异步联邦学习中实现了高效的联邦遗忘。其核心创新在于将复杂的客户端分簇问题转化为可高效求解的线性规划问题，同时保持对数据删除请求的快速响应能力。
