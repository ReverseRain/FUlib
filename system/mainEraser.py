# main.py
import copy

import torch
import numpy as np
from system.flcore.clients.clientEraser import Client
from system.flcore.servers.servereraser import Server
from system.utils.data_preprocess import *
from system.utils.model_initiation import *


class Args:
    def __init__(self):
        self.N_total_client = 100
        self.N_client = 10
        self.global_epoch = 10
        self.local_epoch = 5
        self.local_batch_size = 64
        self.local_lr = 0.01
        self.test_batch_size = 128
        self.cuda_state = torch.cuda.is_available()
        self.data_name = 'mnist'
        self.seed = 42
        self.forget_client_idx = 2
        self.use_gpu = self.cuda_state


def fed_unlearning(global_models, client_models, forget_client_idx, FL_params):
    """
    FedEraser 遗忘：从训练轨迹中移除指定客户端的影响
    global_models: List[global_model_t] 每一轮的全局模型
    client_models: List[client_model_i_t] 所有客户端每轮模型（顺序：轮数 * 客户端数量）
    """
    num_clients = FL_params.N_client
    global_model = copy.deepcopy(global_models[-1])  # 当前模型
    device = torch.device("cuda" if FL_params.use_gpu else "cpu")
    global_model.to(device)

    for t in range(FL_params.global_epoch):
        # 当前轮中，被遗忘客户端的模型参数
        idx = t * num_clients + forget_client_idx
        delta = {}

        for name, param in global_model.named_parameters():
            param.data = param.data.clone()

            # 获取对应轮次全局模型和被遗忘客户端模型
            global_param_prev = global_models[t].state_dict()[name].to(device)
            client_param = client_models[idx].state_dict()[name].to(device)

            # 差值传播公式：从全局模型中移除该客户端对模型的贡献
            delta[name] = global_param_prev - client_param
            param.data += delta[name] / num_clients  # 反向加回被移除的那份

    print(f"[FedEraser] 已移除 Client {forget_client_idx} 的影响。")
    return global_model.cpu()


def average_models(models):
    avg_model = copy.deepcopy(models[0])
    state_dict = avg_model.state_dict()

    for key in state_dict:
        state_dict[key] = torch.stack([m.state_dict()[key] for m in models], dim=0).mean(dim=0)

    avg_model.load_state_dict(state_dict)
    return avg_model

def fed_unlearning_with_calibration(global_models, client_models, client_loaders, args):
    """
    FedEraser 带校准卸载方法：论文中的方向 × 步长 公式
    """
    num_clients = args.N_client
    forget_idx = args.forget_client_idx
    global_epoch = args.global_epoch

    # 构造新模型
    new_global_model = copy.deepcopy(global_models[0])
    new_global_models = [new_global_model]

    for t in range(global_epoch):
        print(f"[FedEraser-Calibrated] Unlearning Round {t+1}")
        prev_model = new_global_models[-1]

        # 获取原始步长, 旧的 client model - old global
        old_global = global_models[t]
        step_length = {}
        for name in prev_model.state_dict():
            old_clients = [
                client_models[t * num_clients + i]
                for i in range(num_clients) if i != forget_idx
            ]
            avg_client = average_models(old_clients).state_dict()[name]
            step_length[name] = avg_client - old_global.state_dict()[name]

        # 获取新方向, 重新用 prev_model 训练 client
        ref_clients = []
        for i in range(num_clients):
            if i == forget_idx:
                continue
            client = Client(i, copy.deepcopy(prev_model), client_loaders[i], args)
            client.train()
            ref_clients.append(client)

        new_clients = [client.model for client in ref_clients]
        avg_new_client = average_models(new_clients)

        step_direction = {}
        for name in prev_model.state_dict():
            diff = avg_new_client.state_dict()[name] - prev_model.state_dict()[name]
            step_direction[name] = diff / (torch.norm(diff) + 1e-8)

        # 更新 new_global_model = prev_model + step_length * step_direction
        updated_state = {}
        for name in prev_model.state_dict():
            update = torch.norm(step_length[name]) * step_direction[name]
            updated_state[name] = prev_model.state_dict()[name] + update

        updated_model = copy.deepcopy(prev_model)
        updated_model.load_state_dict(updated_state)
        new_global_models.append(updated_model)

    return new_global_models[-1]







def main():
    args = Args()
    torch.manual_seed(args.seed)
    print(args.cuda_state)

    # 使用 data_init 加载数据
    client_loaders, test_loader = data_init(args)
    global_model = model_init(args.data_name)
    clients = []
    for i in range(args.N_total_client):
        model_copy = model_init(args.data_name)
        clients.append(Client(i, model_copy, client_loaders[i], args))
    server = Server(global_model, clients, test_loader, args)


    # 添加模型保存容器
    global_model_snapshots = [copy.deepcopy(server.global_model)]
    client_model_snapshots = []
    # 联邦训练主循环
    for round in range(args.global_epoch):
        print(f"\n--- Global Round {round + 1} ---")
        selected_clients = np.random.choice(clients, args.N_client, replace=False)

        server.distribute_model(selected_clients)

        # 保存当前轮的客户端模型副本
        current_client_models = []
        for client in selected_clients:
            client.train()
            current_client_models.append(copy.deepcopy(client.model))
        client_model_snapshots.extend(current_client_models)

        server.aggregate_models(selected_clients)
        server.evaluate()

        # 保存当前全局模型
        global_model_snapshots.append(copy.deepcopy(server.global_model))



    # 执行卸载
    unlearned_model = fed_unlearning(
        global_models=global_model_snapshots,
        client_models=client_model_snapshots,
        forget_client_idx=args.forget_client_idx,
        FL_params=args
    )
    print("\n[Server Evaluation] After FedEraser Unlearning:")
    server.global_model = unlearned_model
    server.evaluate()



    # 执行 FedEraser 卸载（with calibration）
    print("\n[Start] FedEraser With Calibration")
    calibrated_unlearned_model = fed_unlearning_with_calibration(
        global_models=global_model_snapshots,
        client_models=client_model_snapshots,
        client_loaders=client_loaders,
        args=args
    )
    print("\n[Server Evaluation] After FedEraser With Calibration:")
    server.global_model = calibrated_unlearned_model
    server.evaluate()




if __name__ == '__main__':
    main()

