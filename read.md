参数说明
    -s:一共有三种情况learning，unlearning和retrain
       learning会保存一个格式为"unleanring_clients+attack".pt的模型参数
       unlearning会根据参数load对应模型
       retrain会自动忽略遗忘的客户端从头学习
    -att：True代表遗忘客户端是有毒数据(后门攻击)，False代表室正常数据
    -ugr：unlearning的通讯轮数
    -gr：训练时的通讯轮数
    -uc：遗忘客户端的id号(可以是多个)
# FedEraser

训练：使用FedAVG聚合

卸载：每一轮中将忘记的客户端的影响“逆向加回去”，具体更新如下：
$$
θ_{t+1}^{new}=θ_{t+1}^{old}+\frac{1}{N}(θ_{t}^{old}-θ_{t}^{f})

$$
θt(f)：表示被遗忘客户端在第 t 轮的本地模型

θt(old)：为第 t 轮原来的全局模型





# FedLU

#### 仿照FedE的训练过程：

##### step0：初始化，使用controller.py

1. 初始化服务器 Server，客户端 Client[i]；

2. 服务器构建并初始化 global_entity_embedding；

3. 将全局嵌入按客户端实体映射下发给各客户端：

##### step1：Client训练：

每个客户端本地持有：

- 训练集三元组 `train.txt`
- 嵌入模型 `KGEModel`
- 两份嵌入副本：
  - `E_local`：本地学习优化方向
  - `E_global`：从服务器继承的嵌入，作为蒸馏教师

 损失函数设计：

每一轮训练时的总损失是两部分组成的：

| 损失项      | 含义                                                       |
| ----------- | ---------------------------------------------------------- |
| `L_predict` | 对本地三元组进行评分预测的标准 loss（正负采样 + margin）   |
| `L_distill` | 蒸馏损失，保持 local 与 global 嵌入输出分布一致（KL 散度） |

$$
L = L_{predict} + μ * L_{distill}
$$

其中`μ` 是蒸馏强度调节参数，超参数。



##### step2：上传本地实体嵌入

每个客户端训练完一轮之后，会上传本地更新后的实体嵌入（或固定嵌入）到服务器：



##### 步骤 3：服务器聚合全局嵌入

Server 接收来自所有客户端的实体嵌入更新，使用指定策略进行聚合：

| 聚合策略（args.agg） | 说明                                 |
| -------------------- | ------------------------------------ |
| `weighted`           | 直接平均                             |
| `distance`           | 距离大的客户端权重更大（变化更多）   |
| `similarity`         | 与旧全局嵌入越相似权重越大（更一致） |

最终更新 `global_entity_embedding`。



##### 步骤 4：服务器分发新嵌入

Server把更新后的 `global_entity_embedding` 的每个子集分发给对应客户端：



##### 步骤 5：验证与早停控制（Controller ）

- 每隔 `valid_iter` 轮，执行验证（每个客户端在 `valid.txt` 上）
- 使用加权 MRR 衡量全局效果
- 提前满足性能条件则早停并保存最好的一轮模型状态



#### 联邦遗忘过程：















