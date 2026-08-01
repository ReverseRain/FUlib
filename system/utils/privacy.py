from opacus import PrivacyEngine
import torch

MAX_GRAD_NORM = 1.0
DELTA = 1e-5

def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine



def apply_dp_gradients(model, sigma, max_grad_norm=MAX_GRAD_NORM):
    """轻量版 DP 加噪：只对整体梯度做裁剪 + 高斯加噪。
    DP-SGD 的核心两步：
      1. 梯度裁剪到 max_grad_norm；
      2. 加入标准差为 sigma * max_grad_norm 的高斯噪声。
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_grad_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)
            noise_std = sigma * max_grad_norm
            noise = torch.randn_like(p.grad) * noise_std
            p.grad.data.add_(noise)


# ---------------------------------------------------------------
# 轻量版 DP 的隐私预算 (ε, δ) 统计
# 使用 RDP（Rényi 差分隐私）会计：每加噪一步记录一次，最后反解 ε。
# ---------------------------------------------------------------
_dp_accountant = None


def init_dp_accountant():
    """初始化隐私会计。在开始 unlearning 前调用一次。"""
    global _dp_accountant
    from opacus.accountants import RDPAccountant
    _dp_accountant = RDPAccountant()
    return _dp_accountant


def log_dp_step(sigma, batch_size, total_samples):
    """在每次 apply_dp_gradients 之后调用，记录一步的隐私消耗。

    Args:
        sigma: 噪声乘子（noise_multiplier）。
        batch_size: 当前 batch 大小。
        total_samples: 该客户端总的训练样本数。
    """
    global _dp_accountant
    if _dp_accountant is None:
        raise RuntimeError("请先调用 init_dp_accountant() 初始化隐私会计")
    # 采样率 = batch / 总样本数
    sample_rate = batch_size / total_samples
    _dp_accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)


def get_epsilon_delta(delta=DELTA):
    """返回当前累计的 (epsilon, delta)。

    Args:
        delta: 允许的隐私破例概率，默认用文件顶部的 DELTA。
    Returns:
        (epsilon, delta) 元组。
    """
    global _dp_accountant
    if _dp_accountant is None:
        raise RuntimeError("请先调用 init_dp_accountant() 初始化隐私会计")
    epsilon = _dp_accountant.get_epsilon(delta=delta)
    return epsilon, delta


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA