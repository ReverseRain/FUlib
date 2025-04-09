"""
Knot: a clustered aggregation mechanism designed for federated unlearning.
"""

import knot_server
import knot_algorithm
import knot_client
import knot_trainer
import sys
import os

# 获取 `knot_server.py` 所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算 `plato` 目录的绝对路径
plato_dir = os.path.abspath(os.path.join(current_dir, "../../plato"))

# 添加 `plato` 目录到 `sys.path`
sys.path.append(plato_dir)


def main():
    """
    Knot: a clustered aggregation mechanism designed for federated unlearning.
    """
    algorithm = knot_algorithm.Algorithm
    trainer = knot_trainer.Trainer
    client = knot_client.Client(algorithm=algorithm, trainer=trainer)
    server = knot_server.Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
