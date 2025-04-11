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

### 1. 精确遗忘机制（Exact Unlearning via Re-seeding）

- 若需要移除的数据点未被选为初始质心 → 模型可直接保留；
- 否则，从受影响的聚类开始重跑 K-means++ 初始化（算法 3）；
- 该过程保证新模型与移除该数据后的模型具有等价分布。

###  2. 联邦聚类遗忘过程（Algorithm 4）

- **单客户端数据删除**：
  - 客户端运行算法 3；
  - 若质心变化，客户端重发 `q(l)` 至服务器；
  - 服务器运行聚类生成新的 `Cs`；
- **多客户端删除**：
  - 删除客户端向服务器发送 `q(l)=0`；
  - 服务器重聚合并更新全局模型；
- 未受影响的客户端无需参与操作。

---

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
