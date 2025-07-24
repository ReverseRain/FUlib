#!/bin/bash
#SBATCH -J run_unlearning_all_cifar100           # 作业名称
#SBATCH -o run_all_unlearning_cifar100.log     # Slurm系统日志输出
#SBATCH --partition=a100                # 所属分区
#SBATCH --qos=a100                      # QOS等级
#SBATCH --gres=gpu:1                    # 请求1个GPU
#SBATCH --ntasks=1                      # 1个任务
#SBATCH --cpus-per-task=4               # 每个任务使用4个CPU核
#SBATCH --mem=32G                       # 32G内存
#SBATCH --time=2-00:00:00               # 最长运行时间（2天）

# 指定 Python 路径（按需修改你的环境路径）
PYTHON_BIN="$HOME/.conda/envs/myenv310/bin/python"

# 显示 Python 路径和 Torch 版本用于调试
echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print('Torch version:', torch.__version__)"

echo "Initializing Model..."
$PYTHON_BIN main.py -data Cifar100 -m cnn -algo FedEraser -gr 500 -did 0 -uc 1 -nc 10 -s learning -nb 100 \
    > initialize_model_run_cifar100.out 2>&1

echo "Running FedGEM..."
$PYTHON_BIN main.py -data Cifar100 -m cnn -algo FedGEM -did 0 -uc 1 -nc 10 -s unlearning -ugr 10 -nb 100 -ulr 0.0001 \
    > FedGEM_unBackdoor_cifar100.out 2>&1

echo "Running FedBU..."
$PYTHON_BIN main.py -data Cifar100 -m cnn -algo FedBU -did 0 -uc 1 -nc 10 -s unlearning -ugr 10 -nb 100 -ulr 0.0001 \
    > FedBU_unBackdoor_cifar100.out 2>&1

echo "Running FedEE..."
$PYTHON_BIN main.py -data Cifar100 -m cnn -algo FedEE -did 0 -uc 1 -nc 10 -s unlearning -ugr 10 -nb 100 -ulr 0.0001 \
    > FedEE_unBackdoor_cifar100.out 2>&1

echo "Running FedOSD..."
$PYTHON_BIN main.py -data Cifar100 -m cnn -algo FedOSD -did 0 -uc 1 -nc 10 -s unlearning -ugr 10 -nb 100 -ulr 0.0001 \
    > FedOSD_unBackdoor_cifar100.out 2>&1

echo "Running FedROME..."
$PYTHON_BIN main.py -data Cifar100 -m cnn -algo FedROME -did 0 -uc 1 -nc 10 -s unlearning -ugr 10 -nb 100 -ulr 0.0001 \
    > FedROME_unBackdoor_cifar100.out 2>&1
