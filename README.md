# Provably Accurate Federated Clustering with Unlearning Mechanism
An efficient method for federated (K-means) clustering and its corresponding unlearning procedure, which is introduced in our paper:

- [ICLR 2023] [Machine Unlearning of Federated Clusters](https://openreview.net/pdf?id=VzwfoFyYDga)

# Datasets
`Celltype`, `Gaussian`, `Postures`, `Covtype` can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LqazOJuH3uOgFxHtBodwon6htEE2Wq13) provided by the authors of [DC-Kmeans](https://arxiv.org/abs/1907.05012). `FEMNIST` can be downloaded from the [Leaf Project](https://leaf.cmu.edu/). `TCGA` and `TMI` may contain potentially sensitive biological data and can be downloaded after logging into the databases ([TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga), [TMI](https://commonfund.nih.gov/hmp)). We can provide the data processing pipelines upon reasonable requests via emails.

We also provide a utility function `generate_data` in `utils.py` to generate the data for clients in federated setting, where `data_input` is the raw global feature matrix. Please refer to the function for more details. One example of the `Celltype` dataset after data generation is included in this repository.

# Usage
Two other methods, [DC-Kmeans](https://arxiv.org/abs/1907.05012) and [K-FED](https://arxiv.org/abs/2103.00697), are also implemented in this repository for comparison.

To run the methods on the example dataset, you can use the following command
```
python mufc_main.py --num_clusters=4 --num_clients=100 --data_path=celltype_processed.pkl --num_removes=10 \
                    --k_prime=4  --split=non-iid  --compare_kfed --compare_dc --client_kpp_only --verbose --update_centralized_loss
```
or simply run the shell script
```
chmod +x run.sh
./run.sh
```

# Contact
Please contact Chao Pan (chaopan2@illinois.edu), Jin Sima (jsima@illinois.edu), Saurav Prakash (sauravp2@illinois.edu) if you have any question.

# Citation
If you find our code or work useful, please consider citing our paper:
```
@inproceedings{
pan2023machine,
title={Machine Unlearning of Federated Clusters},
author={Chao Pan and Jin Sima and Saurav Prakash and Vishal Rana and Olgica Milenkovic},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=VzwfoFyYDga}
}
```



#  Machine Unlearning of Federated Clusters (MUFC) 算法逻辑总结

## 一、背景介绍

- 联邦聚类（Federated Clustering, FC）是一种无监督学习任务，广泛用于推荐系统、医疗等应用场景；
- 用户有“被遗忘”的权利（GDPR、CCPA等法规），但聚类任务中的模型通常难以高效删除指定用户影响；
- 本文首次提出 **联邦聚类下的机器遗忘问题（FC Unlearning）**，并构建了一个高效、安全、可遗忘的聚类框架。

---

## 二、MUFC 框架设计核心

###  1. 客户端聚类（Client Clustering）

- 每个客户端并行运行 K-means++ 初始化，得到局部聚类中心 `C(l)`；
- 每个聚类的大小也会被记录，并以向量 `q(l)` 形式表示；
- 向量经均匀量化后，映射到对应“量化区间”，用于后续加密聚合。

###  2. SCMA 安全聚合机制（Secure Compressed Multiset Aggregation）

- 使用 **Reed-Solomon 编码 + 多项式编码** 实现加密通信；
- 每个客户端向服务器发送压缩向量的“扰动哈希”，服务器聚合后可恢复总体统计量 `q`；
- 不暴露单个客户端的任何聚类信息。

###  3. 服务器重构（Server Aggregation）

- 服务器根据 `q` 生成加权数据点集合 `Xs`；
- 在 `Xs` 上运行完整的 K-means++ 聚类，得到全局聚类中心 `Cs`。

---

## 三、MUFC 机器遗忘算法逻辑

##  Algorithm 1：Secure Federated Clustering (MUFC)

**输入：**
- 分布在 L 个客户端的数据集 X = (X^{(1)}, ..., X^{(L)})

**步骤：**
1. 每个客户端并行运行 K-means++ 初始化：
   得到初始质心集 C^{(l)}，并记录每个聚类的大小 |C^{(l)}_k|。
2. 每个客户端对质心进行均匀量化，生成向量 q^{(l)}：
   - 将每个 c^{(l)}_k 映射到对应的量化 bin。
   - 若 c^{(l)}_k 落在 bin j，则 q^{(l)}_j = |C^{(l)}_k|。
   - 其余 q^{(l)}_j = 0。
3. 使用 Algorithm 2（SCMA）在服务器端加密聚合所有 q^{(l)}，得到 q。
4. 对 q 中每个非零元素 j，采样 q_j 个点，构成服务器数据集 X_s。
5. 服务器在 X_s 上运行 K-means++ 聚类，得到全局质心集 C_s。

**输出：**
- 每个客户端保存 C^{(l)}，服务器保存 (X_s, q, C_s)

---

##  Algorithm 2：Secure Compressed Multiset Aggregation (SCMA)

**输入：**
- L 个客户端向量 q^{(l)}，长度 B^d，最多 K 个非零元素
- 有限域 F_p（p ≥ max{n, B^d}）

**步骤：**
1. 每个客户端计算 S^{(l)}_i = Σ q_j * j^{i-1} + z_i（i = 1,...,2KL），z_i 为随机掩码。
2. 所有 S^{(l)}_i 发送到服务器，服务器聚合得到 S_i。
3. 服务器构造多项式并解码根位置 j，使用 Reed-Solomon 编码求得非零位置。
4. 解线性方程组还原 q。

**输出：**
- 稀疏向量 q（服务器端）

---

##  Algorithm 3：K-means++ Initialization-based Unlearning

**输入：**
- 数据集 X，质心集合 C，删除数据 X_R

**步骤：**
1. 若所有 cj ∉ X_R：返回 C。
2. 若有 cj ∈ X_R：
   - 找到第一个满足条件的 cj，设索引为 i。
   - i = 0 时：C′ ← ∅，X′ ← X \ X_R。
   - 否则：C′ ← {c1, ..., ci}，X′ ← X \ X_R。
3. 对 j = i+1,...,K，按加权概率从 X′ 采样点加入 C′。

**输出：**
- 新质心集合 C′

---

##  Algorithm 4：Unlearning Federated Clusters

**输入：**
- L 个客户端数据 X = (X^{(1)}, ..., X^{(L)})
- MUFC 输出：(C^{(l)}, X_s, q, C_s)
- 删除请求：X^{(l)}_R 或客户端集合 L_R

**步骤：**
1. 若为单客户端删除：客户端 l 运行 Algorithm 3，若更新则生成新的 q^{(l)}。
2. 若为多客户端删除：所有 l ∈ L_R，设置 q^{(l)} = 0。
3. 聚合 q^{(l)} 得 q′，若 q′ == q：C_s′ = C_s。
4. 否则重新生成 X_s′ 并运行 K-means++ 得到新的 C_s′。

**输出：**
- 更新的 C^{(l)}′，X_s′，q′ 和 C_s′

---

##  理论保证

- 聚类误差上界：  
  E[ϕ_f(X; C_s)] ≤ O(log² K)·ϕ*_c(X) + O(ndγ² log K)
- 删除复杂度：
  - 随机删除：O(RK²d)，对抗删除：O(RK³ϵ₁ϵ₂d)
  - 完整重训：O(nKd)
## 四、算法性能与优势分析

### 优点

1. **高效可扩展**：平均加速约 84x（相较完全重训）；
2. **隐私保护**：SCMA 保证只暴露总量统计，不泄漏单个客户端信息；
3. **兼顾通信成本**：客户端通信复杂度为 `O(KL log d)`，远优于线性方法；
4. **理论保证**：提供了严格的误差上界（`O(log² K)`）和复杂度分析；
5. **适应异构数据**：特别适用于聚类规模不平衡的非 i.i.d. 数据环境。

### 缺点

1. **服务器需重聚类**：每次删除仍需服务器重新聚类（有一定代价）；
2. **受限于初始化方法**：仅支持 K-means++，不适用于所有聚类算法；
3. **对参数 γ 依赖敏感**：量化步长 γ 过大会影响聚类精度；
4. **对客户端聚类质量敏感**：若初始客户端质心选择不合理，可能影响最终效果。

---
