import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from system.flcore.clients.clientEraser import Client
from system.flcore.servers.servereraser import Server
from system.utils.data_preprocess import *
from system.utils.model_initiation import *

class Args:
    def __init__(self):
        self.N_total_client = 10
        self.N_client = 10
        self.global_epoch = 20
        self.local_epoch = 5
        self.local_batch_size = 64
        self.local_lr = 0.01
        self.test_batch_size = 128
        self.cuda_state = torch.cuda.is_available()
        self.data_name = 'mnist'
        self.seed = 42
        self.backdoor_clients = [2]  # 多个后门客户端
        self.use_gpu = self.cuda_state

def add_backdoor(dataset, trigger_label=7):
    """对所有数据添加3x3黑格子触发器并将标签改为 trigger_label"""
    def add_pattern(img):
        img[:, 25, 25] = 0.0
        return img

    backdoored_data = []
    for x, y in dataset:
        x = add_pattern(x.clone())
        y = trigger_label
        backdoored_data.append((x, y))
    return backdoored_data

def fed_unlearning(global_models, client_models, forget_client_indices, FL_params):
    """
    forget_client_indices: list of client ids to unlearn
    """
    num_clients = FL_params.N_client
    global_model = copy.deepcopy(global_models[-1])
    device = torch.device("cuda" if FL_params.use_gpu else "cpu")
    global_model.to(device)

    for t in range(FL_params.global_epoch):
        for client_id in forget_client_indices:
            idx = t * num_clients + client_id
            if idx >= len(client_models):
                continue  # 防止索引越界
            for name, param in global_model.named_parameters():
                global_param_prev = global_models[t].state_dict()[name].to(device)
                client_param = client_models[idx].state_dict()[name].to(device)
                param.data += (global_param_prev - client_param) / num_clients

    print(f"[FedEraser] 已移除 Clients {forget_client_indices} 的影响。")
    return global_model.cpu()

def evaluate_backdoor(model, device, target_label=7):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=False, transform=transform)

    poisoned = []

    def add_pattern(x):
        x[:, 25, 25] = 0.0
        return x

    for x, y in dataset:
        x = add_pattern(x.clone())
        poisoned.append((x, target_label))

    loader = torch.utils.data.DataLoader(poisoned, batch_size=64, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total if total > 0 else 0.0

def main():
    args = Args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # 数据加载
    client_loaders, test_loader = data_init(args)

    # 初始化
    global_model = model_init(args.data_name).to(device)
    clients = [Client(i, model_init(args.data_name).to(device), client_loaders[i], args)
               for i in range(args.N_total_client)]
    server = Server(global_model, clients, test_loader, args)

    global_model_snapshots = [copy.deepcopy(global_model)]
    client_model_snapshots = []

    backdoor_injected = False
    for round in tqdm(range(args.global_epoch), desc="Federated Rounds"):
        print(f"\n--- Global Round {round + 1}/{args.global_epoch} ---")

        if not backdoor_injected and round == args.global_epoch - 10:
            print(f"\n[Backdoor] Injecting into clients {args.backdoor_clients} at round {round + 1}")
            for idx in args.backdoor_clients:
                backdoor_dataset = add_backdoor(client_loaders[idx].dataset)
                client_loaders[idx] = torch.utils.data.DataLoader(
                    backdoor_dataset, batch_size=args.local_batch_size, shuffle=True
                )
                clients[idx] = Client(
                    idx,
                    model_init(args.data_name).to(device),
                    client_loaders[idx],
                    args
                )
            backdoor_injected = True

        selected_clients = np.random.choice(clients, args.N_client, replace=False)

        if backdoor_injected:
            selected_ids = {c.id for c in selected_clients}
            need_inject = [idx for idx in args.backdoor_clients if idx not in selected_ids]
            if need_inject:
                selected_clients = list(selected_clients)
                for i in range(min(len(need_inject), len(selected_clients))):
                    selected_clients[i] = clients[need_inject[i]]
                selected_clients = np.array(selected_clients)

        server.distribute_model(selected_clients)

        current_models = []
        for client in selected_clients:
            client.train()
            current_models.append(copy.deepcopy(client.model))

        client_model_snapshots.extend(current_models)
        server.aggregate_models(selected_clients)
        server.evaluate()
        global_model_snapshots.append(copy.deepcopy(server.global_model))

    # 后门攻击评估（训练结束后）
    print("\n[Backdoor Evaluation] Before Unlearning")
    asr_before = evaluate_backdoor(server.global_model.to(device), device)
    print(f"[ASR Before Unlearning] Attack Success Rate: {asr_before:.4f}")

    # 联邦卸载多个污染客户端
    unlearned_model = fed_unlearning(global_model_snapshots, client_model_snapshots,
                                     args.backdoor_clients, args)

    # 卸载后评估 ASR
    print("\n[Backdoor Evaluation] After Unlearning")
    asr_after = evaluate_backdoor(unlearned_model.to(device), device)
    print(f"[ASR After Unlearning] Attack Success Rate: {asr_after:.4f}")

    # 可视化前后对比
    plt.bar(["Before", "After"], [asr_before, asr_after], color=["red", "green"])
    plt.title("Attack Success Rate Before and After Unlearning")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.show()

if __name__ == '__main__':
    main()
