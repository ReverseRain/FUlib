#!/bin/bash
#SBATCH -J initialize_model_cifar100            # 作业名称
#SBATCH -o initialize_model_run_cifar100.log        # 标准输出日志
#SBATCH --partition=a100                # 所属分区（如a100）
#SBATCH --qos=a100                      # QOS等级
#SBATCH --gres=gpu:1                    # 请求1个GPU
#SBATCH --ntasks=1                      # 1个任务
#SBATCH --cpus-per-task=4               # 每个任务使用4个CPU核
#SBATCH --mem=32G                       # 32G内存
#SBATCH --time=2-00:00:00               # 最长运行时间（2天）

# 明确指定 Conda 环境中的 Python 路径
PYTHON_BIN="$HOME/.conda/envs/myenv310/bin/python"

# 打印 Python 路径和 Torch 版本信息，便于调试
echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print('Torch version:', torch.__version__)"

# 运行 FedEraser 实验，保存输出日志
$PYTHON_BIN main.py -data Cifar100 -m cnn -algo FedEraser -gr 300 -did 0 -uc 1 -nc 10 -s learning -att False -nb 100 \
    > initialize_model_run_cifar100.out 2>&1
