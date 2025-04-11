# FUlib 

## 一、联邦学习的相关背景

#### 定义：

在真实世界中，不同机构（如企业、医院、研究组织）拥有**分布式的、敏感的、异构的知识图谱数据**，无法直接共享。这就需要 **联邦学习**：分布式机器学习框架，允许数据保留在本地，仅共享模型参数。

#### 挑战：

数据异构型，例如在KG中不同的客户端的不会包含整个知识图谱，而只是包含了知识图谱的一部分的子图，这样其他client以及server无法访问其他client的数据

#### 机制：

- 客户端（如用户设备）在本地训练模型；
- 只将模型参数或梯度上传给服务器，而不是原始数据；
- 服务器聚合这些参数构建全局模型（如 FedAvg 算法）；





## 二、联邦卸载的相关背景

#### 挑战：

1. 数据不上传服务器，模型参数中仍然可能泄露用户信息：模型可能记住某个客户端的数据特征。因此可以通过MIA攻击，使用影子数据集训练一个攻击分类器来取得一些训练集中的隐藏数据；或者可能通过后门攻击通过污染训练数据集使得模型在遇到特定训练集个体的时候会输出特定的结果
2. client自行退出训练不希望训练集被用于训练。
3. 如果重新开始训练需要耗费的资源很大
4. 客户端数据无法直接访问；
5. 训练过程是分布式、异步、不可复现的；

#### 目的：

- 实现对特定客户端或其数据贡献的**有针对性删除**；

- 移除模型中恶意客户端的影响（如后门注入）；

- 保留有用信息，尽可能避免从头重训。

#### 现有方法不足：

- 很多方法仍依赖部分重训练；

- 容易忘掉好的信息，retrain之后的结果下降很多；

- 需要太多存储，如历史模型、更新记录等；
-  标准化缺失：
  - 缺乏统一的评估数据集和指标；
  - 不同论文用不同攻击测试方式





## 三、复现算法回顾

#### FedEraser:Clinet level

FedEraser 利用联邦训练期间保存的模型轨迹，设计了一个定向矢量校准更新公式来实现卸载：
$$
G_t' = G_{t-1}' + | \Delta_t | \cdot \frac{\Delta_t'}{| \Delta_t' |} 
$$
其中：

- Gt−1：为当前重构的 global model；
- Δt：旧训练轨迹中，未卸载客户端的本地平均模型与旧 global 的差值；
- Δt′：使用当前 global model Gt−1′G_{t-1}'Gt−1′ 在未卸载客户端重新训练得到的方向；
- ∥Δt∥：保留旧的训练“步长”；
- ∥Δt′∥：校准出的训练“方向”。

具体过程，和FedE相似： 

1、校准训练（Calibration Training）

- 非目标客户端 以最近一次的卸载模型 M 为起点，执行 E 轮本地训练，生成新更新：
  $$
  \widehat{U}_{k_c}^{(t_j)} = \text{Train}(C_{k_c}, \widetilde{\mathbf{M}}^{(t_j)})
  $$
  

2、更新校准（Update Calibrating）

- 将之前保留的原始更新 $U_{k_c}^{(t_j)}$ 校准为：
  $$
  \widetilde{U}_{k_c}^{(t_j)} = \|U_{k_c}^{(t_j)}\| \cdot \frac{\widehat{U}_{k_c}^{(t_j)}}{\|\widehat{U}_{k_c}^{(t_j)}\|}
  $$
  即保持原始更新的幅度 |U|，替换为新方向 。

3、聚合校准更新

- 对所有非目标客户端的校准更新进行平均聚合：
  $$
  \widetilde{U}^{(t_j)} = \frac{1}{K-1} \sum_{k_c \neq k_u} w_{k_c} \widetilde{U}_{k_c}^{(t_j)}
  $$
  

4、更新模型（Model Updating）

- 应用上述聚合更新至当前模型：
  $$
  \widetilde{\mathbf{M}}^{(t_{j+1})} = \widetilde{\mathbf{M}}^{(t_j)} + \widetilde{U}^{(t_j)}
  $$
  递推更新直到构造出最终的 Model。



#### FedLU: Instance level

每个客户端 k 拥有本地知识图谱 Gk={(h,r,t)∣h,t∈Ek,r∈Rk}。

整个联邦嵌入学习目标为：

整个联邦嵌入学习目标为：
$$
E = \arg\min_E \sum_{k \in \mathcal{K}} \frac{|G_k|}{|G|} \mathcal{L}(P_k E; G_k)
$$


- E：全局实体嵌入
- Pk：投影矩阵，映射出客户端 k 所需的嵌入子集
- L：自对抗负采样损失

首先定义损失函数：对于每个三元组 (h,r,t)∈Gk(h, r, t) \in G_k(h,r,t)∈Gk，定义如下三个损失：

- **预测损失（对比预测）**：

$$
\mathcal{L}_{\text{predict}} = -\log \sigma(S_{\text{local}}(h, r, t)) - \frac{1}{n} \sum_{(h', r, t') \in N} \log \sigma(-S_{\text{local}}(h', r, t'))
$$



- **蒸馏损失**（知识迁移）：

$$
\mathcal{L}_{\text{distill}} = \text{KL}(P_{\text{local}}, P_{\text{global}})
$$



- **联合优化**：

$$
\mathcal{L}_{\text{local}} = \sum_{(h, r, t)} \left[ \mathcal{L}_{\text{predict}} + \mu_{\text{distill}} \mathcal{L}_{\text{distill}} \right]
$$

然后定义联邦训练的流程：

1. 初始化全局嵌入E0

2. 每轮通信 t：

   - 选取子集客户端 Kt

   - 对于每个客户端 k∈Kt

     - 服务器发送 Pk x Et作为客户端初始嵌入

     - 客户端优化本地嵌入：
       $$
       E_k^{\text{local}} = \arg\min_{E_k} \mathcal{L}_{\text{local}}(E_k; P_k E_t, G_k)
       $$
       

     - 客户端用该本地嵌入进一步反向优化全局嵌入：
       $$
       E_k^{\text{global}} = \arg\min_{E_k} \mathcal{L}_{\text{global}}(E_k; E_k^{\text{local}}, G_k)
       $$

     - 上传 Ekglobal

   - 服务端更新全局嵌入：
     $$
     E_{t+1} = \left( \frac{1}{\sum_k v_k} \right) \otimes \sum_{k \in \mathcal{K}_t} P_k^\top E_k^{\text{global}}
     $$

然后训练完成后定义联邦卸载的流程：FedLU结合了 **逆行干扰**(逆数据集训练硬混淆)与 **被动遗忘**（软混淆）。

 Step 1主动干扰：使用三种损失联合扰乱嵌入对删除样本的记忆：

1. **Hard Confusion Loss**：将卸载三元组视为负样本：

$$
\mathcal{L}_{\text{hard}} = -\log \sigma(-S_{\text{local}}(h, r, t)) - \frac{1}{n} \sum_{(h', r, t')} \log \sigma(-S_{\text{local}}(h', r, t'))
$$

2. **Soft Confusion Loss**：让得分靠近负样本，防止留下“删除痕迹”：

$$
\mathcal{L}_{\text{soft}} = \frac{1}{n} \sum_{(h', r, t')} \| S_{\text{local}}(h', r, t') - S_{\text{local}}(h, r, t) \|^2
$$



3. **Distillation Loss**：继续保持与全局嵌入的一致性。

最终干扰损失：
$$
\mathcal{L}_{\text{interference}} = \mathcal{L}_{\text{hard}} + \mu_{\text{soft}} \mathcal{L}_{\text{soft}} + \mu_{\text{distill}} \mathcal{L}_{\text{distill}}
$$
Step 2：被动遗忘

在保留集 Gk 上继续通过双向蒸馏训练：

- 将本地嵌入作为 teacher 更新全局嵌入
- 再反过来用全局优化本地嵌入



## 四、当前工作的不足及改进

FedEraser：

1. 递归改变模型的校准方向，同时删除了一部分的数据集，如果之前的数据之间有关系可能会导致这种关系被破坏，accuracy变低
2. 在MIA实验中发现没有很大的准确率变化，在后门攻击中发现有时候在经过了卸载脏数据之后的客户端的后门攻击成功率变高
3. non-IID的clients不同客户端的更新方向相差很大，这样做可能会导致模型优化方向不确定



FedLU:

1. 原文没有关于nonIID的实验，不知道会不会在noniid的数据集上有不同的表现效果
2. 双向知识蒸馏的时间复杂度和计算量太大
3. （原文）对于不同客户端中表述不同但是实际含义相同的最后的映射的向量表示不同，可能有一定的影响