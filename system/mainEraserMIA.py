import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm


from system.flcore.clients.clientEraser import Client as BaseClient
from system.flcore.servers.servereraser import Server
from system.utils.data_preprocess import data_init_with_shadow
from system.utils.model_initiation import model_init
from system.utils.attack_utils import train_attack_model, attack
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

class ClientMIA(BaseClient):
    def __init__(self, client_id, model, train_loader, test_loader, args):
        super().__init__(client_id, model, train_loader, args)
        self.test_loader = test_loader

    def load_train_data(self):
        return self.train_loader

    def load_test_data(self):
        return self.test_loader


def main_mia():
    args = Args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # 加载训练数据和影子数据
    client_loaders, test_loader, shadow_client_loaders, shadow_test_loader = data_init_with_shadow(args)

    # 获取类别数 N_class
    example_input, _ = next(iter(test_loader))
    model = model_init(args.data_name).to(device)
    with torch.no_grad():
        N_class = model(example_input.to(device)).shape[1]

    # 构建真实训练客户端
    global_model = model_init(args.data_name)
    clients = [BaseClient(i, model_init(args.data_name), client_loaders[i], args) for i in range(args.N_client)]
    server = Server(global_model, clients, test_loader, args)

    print("[FL] Running standard federated training...")
    global_model_snapshots = [copy.deepcopy(server.global_model)]
    client_model_snapshots = []
    for epoch in tqdm(range(args.global_epoch), desc="Federated Rounds"):
        selected_clients = np.random.choice(clients, args.N_client, replace=False)
        server.distribute_model(selected_clients)

        local_models = []
        for client in selected_clients:
            client.train()
            local_models.append(copy.deepcopy(client.model))

        server.aggregate_models(selected_clients)
        global_model_snapshots.append(copy.deepcopy(server.global_model))
        client_model_snapshots.extend(local_models)

    # 构造影子客户端并训练攻击模型
    print("[MIA] Training shadow attack model...")
    shadow_model = model_init(args.data_name)
    shadow_model.load_state_dict(global_model_snapshots[-1].state_dict())

    shadow_clients = [
        ClientMIA(i, model_init(args.data_name), shadow_client_loaders[i], shadow_test_loader, args)
        for i in range(args.N_client)
    ]
    attacker = train_attack_model(shadow_model, shadow_clients, N_class, device)

    # 构造目标客户端
    target_clients = [
        ClientMIA(i, model_init(args.data_name), client_loaders[i], test_loader, args)
        for i in range(args.N_client)
    ]

    # 原模型上的攻击效果
    print("[MIA] Evaluating attack on original model")
    pre, rec = attack(global_model_snapshots[-1], attacker, target_clients, N_class, device)
    orig_acc = (pre + rec) / 2
    print(f"[MIA] Precision: {pre:.4f}, Recall: {rec:.4f}, Avg: {orig_acc:.4f}")

    mia_accs = [orig_acc]

    # 每次卸载一个 client 并评估攻击准确率
    for forget_id in range(args.N_client):
        print(f"\n[Unlearning] Forgetting client {forget_id}")
        unlearned_model = fed_unlearning(global_model_snapshots, client_model_snapshots, forget_id, args)

        remaining_clients = [
            ClientMIA(i, model_init(args.data_name), client_loaders[i], test_loader, args)
            for i in range(args.N_client) if i != forget_id
        ]

        pre, rec = attack(unlearned_model, attacker, remaining_clients, N_class, device)
        avg = (pre + rec) / 2
        mia_accs.append(avg)
        print(f"[MIA] After Unlearning - Precision: {pre:.4f}, Recall: {rec:.4f}, Avg: {avg:.4f}")

    # 绘制结果
    plt.figure()
    plt.plot(range(len(mia_accs)), mia_accs, marker='o')
    plt.xlabel("Number of Forgotten Clients")
    plt.ylabel("MIA Attack Accuracy")
    plt.title("MIA Accuracy vs Forgotten Clients")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main_mia()
