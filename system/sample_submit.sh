#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan      # 作业提交的指定分区队列为titan
#SBATCH --qos=titan            # 指定作业的QOS
#SBATCH -J myFirstGPUJob       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 

export HDF5_USE_FILE_LOCKING=FALSE

/home/wangdx_lab/cse12210626/.conda/envs/pfllib/bin/python \
main.py -data Cifar10_noniid -m cnn \
-algo FedEraser -gr 20 -did 0 \
-uc 1 -nc 10 -s learning -att True    
# uc 是指遗忘客户端的id号 nc 是总的客户端数量 -s代表训练状态（retrain，Unlearning。。） 其中retrain会根据uc的id号自动忽略该客户端。
# -ugr是遗忘时的轮数 -gr是训练时的轮数 -att 为True时 代表有后门攻击 否则没有
# -ulr代表 Unlearning时的学习率。   Cifar10 对应iid数据集   Cifar10_noniid是noniid的数据集