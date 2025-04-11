# Plato: A New Framework for Scalable Federated Learning Research

Welcome to *Plato*, a software framework to facilitate scalable, reproducible, and extensible federated learning research. Please refer to the [Plato documentation](https://platodocs.netlify.app/) for details on installing, running and deploying Plato.



# Knot算法逻辑解析（基于《Asynchronous Federated Unlearning》论文）

# 📘 Asynchronous Federated Unlearning 算法逻辑解析：KNOT

## 一、背景简介

随着《通用数据保护条例》（GDPR）等法规的推行，用户拥有“被遗忘权”，即删除其私有数据在模型中的所有影响。然而，在 **联邦学习（Federated Learning, FL）** 中，数据不集中存储于服务器，而是分布于客户端，本地训练后上传模型更新。因此，一旦某客户端请求删除数据，其影响已在多轮通信中逐步传播到其他客户端，**重新训练全体模型的成本极高**。

## 二、KNOT 核心思想

KNOT 提出了 **聚类式异步聚合（Clustered Asynchronous Aggregation）** 机制，通过将客户端划分为多个 **聚类（Clusters）**，只在各聚类内部进行模型聚合：

- 若某客户端请求遗忘，仅影响其所在聚类的模型；
- 其余聚类可继续训练，避免全局重训；
- 使用 **异步联邦学习** 提升效率，服务器不等待所有客户端完成训练。

## 三、KNOT 算法逻辑详解

### 1. 客户端特征提取

为实现高效聚类，需要考虑两个客户端特征：

- **训练时间 `Tk`**：客户端完成本地训练+上传所需的时间（越短越好）；
- **模型偏差 `Sk`**：客户端更新与全局模型更新方向的夹角（越小越好，代表数据更“典型”）。

> 使用 **余弦相似度** 衡量更新方向的相似性：
> 
> \[
Sk = \frac{1 - \cos(\omega_0 - \omega_1, \Delta_k)}{2}
\]

### 2. 匹配度定义（Match Rating）

设有 `N` 个聚类，定义每个聚类的目标训练时间与偏差：

\[
\tilde{T}_n = T_{min} + \frac{T_{max} - T_{min}}{N-1} \cdot (n-1)
\quad
\tilde{S}_n = \frac{n}{N}
\]

客户端 `C_k` 与聚类 `L_n` 的匹配度为：

\[
d_{kn} = \left\| \left[ a(\tilde{T}_n - T_k),\ b(\tilde{S}_n - S_k) \right] \right\|_2
\]

### 3. 优化建模：词典序最小化

引入二值变量 `x_kn ∈ {0, 1}` 表示客户端是否属于聚类，构建向量：

\[
f = (d_{11}x_{11},\ d_{12}x_{12},\ ..., d_{KN}x_{KN})
\]

目标为对 `f` 进行词典序最小化（lexicographical minimization）：
- 先最小化最大值，再最小化次大值，依此类推；
- 防止某聚类存在明显异常值影响训练。

### 4. 加入合理约束

为保证聚类结构合理，添加如下约束：

- 每个客户端最多属于 `c1` 个聚类（通常取 `1`）；
- 每个聚类需包含 `[c2, c3]` 个客户端，保证训练质量；
- 每个客户端必须至少属于一个聚类。

### 5. 转换为线性规划（Linear Programming）

- 原问题为整数非线性优化，难以高效求解；
- 通过引入指数型目标函数 + 构造 **全单模约束矩阵**，转化为 **可解的线性规划问题**；
- 使用商用 LP 求解器（如 Mosek）可在 1 秒内完成求解。

---

## 四、KNOT 的优缺点分析

### ✅ 优点

1. **高效性显著提升**  
   - 聚类局部重训，避免全局模型回滚；
   - 在多个数据集上最高提升达 **85%**（相较 FedBuff）；
   - 异步训练进一步提升性能，减少等待慢客户端的时间浪费。

2. **法律合规性好**  
   - 相较于近似删除方法（如 FedEraser），KNOT 支持“真实重训”，符合法规对数据“彻底擦除”的要求。

3. **算法通用性强**  
   - 可与任意重训机制结合（支持 naive、FedEraser、AdaHessian 等）；
   - 聚类机制独立于重训方式。

4. **理论扎实、可高效求解**  
   - 通过构造匹配函数与线性规划模型，保证解法具有良好数学基础；
   - LP 转化可用现成求解器处理，无需自定义优化器。

---

### ⚠️ 缺点与挑战

1. **聚类前的开销**  
   - 需要预估每个客户端的训练时间与模型偏差，可能在新环境下需额外通信或预训练一轮。

2. **异步机制精细度要求高**  
   - 异步机制中需处理模型陈旧（staleness）问题；
   - 需设置合理的最小客户端数量以维持全局收敛。

3. **数据分布依赖性**  
   - 聚类效果较依赖客户端数据的异质性程度；
   - 非 i.i.d. 分布下可能影响局部模型精度，需额外调参。

4. **仍需部分重训**  
   - 虽然不需全局重训，但仍需聚类内部完整 retraining，成本未降为零。

