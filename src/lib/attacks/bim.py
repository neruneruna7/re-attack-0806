from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

def _default_mean_std(channels: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """チャネル数に応じた既定の平均 / 標準偏差を返す（MNIST / CIFAR を想定）。
    戻り値は (mean, std) で、形状は [1, C, 1, 1]。
    """
    if channels == 1:
        mean = torch.tensor([0.1307], device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor([0.3081], device=device, dtype=dtype).view(1, -1, 1, 1)
    else:
        # CIFAR 風の既定値
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std

def bim_attack(image: Tensor,
               epsilon: float,
               alpha: float,
               num_iter: int,
               target: Tensor,
               model: nn.Module,
               device: torch.device,
               *,
               mean: Optional[Tensor] = None,
               std: Optional[Tensor] = None) -> Tensor:
    """Basic Iterative Method (BIM / iterative FGSM).

    Args:
        image: 非正規化テンソル [B,C,H,W]
        epsilon: 最大摂動量（L-inf）
        alpha: 各ステップのステップサイズ
        num_iter: 反復回数
        target: ラベルテンソル
        model: PyTorch モデル
        device: 実行デバイス
        mean, std: 正規化用テンソル（無ければデフォルトを選択）

    Returns:
        perturbed: 非正規化された摂動後テンソル（detach, clamp済み）
    """
    image_denorm = image.clone().detach().to(device)
    B, C, H, W = image_denorm.shape
    if mean is None or std is None:
        mean_t, std_t = _default_mean_std(C, device, image_denorm.dtype)
    else:
        mean_t = mean.to(device).view(1, -1, 1, 1)
        std_t = std.to(device).view(1, -1, 1, 1)

    perturbed = image_denorm.clone().detach()
    orig = image_denorm.clone().detach()

    # ターゲットを正しいデバイスとデータ型に移す
    if target is not None:
        target = target.to(device)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        target = target.long()

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_iter):
    # 勾配計算用に正規化した入力を準備
        perturbed_norm = (perturbed - mean_t) / std_t
        perturbed_norm = perturbed_norm.clone().detach().requires_grad_(True)

        outputs = model(perturbed_norm)
        loss = loss_fn(outputs, target)
    # 逆伝播
        model.zero_grad()
        loss.backward()

        grad_norm = perturbed_norm.grad
        if grad_norm is None:
            raise RuntimeError("BIM: grad is None")

        grad_pixel = grad_norm / std_t
        perturbed = perturbed + alpha * grad_pixel.sign()

    # epsilon-球に射影してクリップ
        delta = torch.clamp(perturbed - orig, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(orig + delta, 0.0, 1.0).detach()

    return perturbed


def bim_reattack(model: nn.Module, x_adv: Tensor, y_adv: Tensor, device: torch.device, epsilon: float = 0.3,
                 alpha: float = 0.05, num_iter: int = 10,
                 mean: Optional[Tensor] = None,
                 std: Optional[Tensor] = None) -> Tensor:
    """既存の re-attack 用ユーティリティ: 予測ラベルをターゲットに BIM を適用する関数。

    - model: 評価モードで呼び出す前提
    - x_adv: 非正規化された入力 [B,C,H,W]
    - y_adv: ラベルテンソル [B]
    """
    model.eval()
    x_adv_prime = x_adv.clone().detach().to(device)
    y_adv = y_adv.to(device)
    x_orig = x_adv.clone().detach().to(device)

    B, C, H, W = x_adv_prime.shape
    if mean is None or std is None:
        mean_t, std_t = _default_mean_std(C, device, x_adv_prime.dtype)
    else:
        mean_t = mean.to(device).view(1, -1, 1, 1)
        std_t = std.to(device).view(1, -1, 1, 1)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        # normalized input for gradient computation
        x_adv_norm = (x_adv_prime - mean_t) / std_t
        x_adv_norm = x_adv_norm.clone().detach().requires_grad_(True)

        outputs = model(x_adv_norm)
        loss = loss_fn(outputs, y_adv)

        grad_norm = x_adv_norm.grad
        if grad_norm is None:
            # backward explicitly to populate grad
            loss.backward()
            grad_norm = x_adv_norm.grad
        if grad_norm is None:
            raise RuntimeError("bim_reattack: gradient is None")

        grad_pixel = grad_norm / std_t
        x_adv_prime = x_adv_prime.detach() + alpha * grad_pixel.sign()

        delta = torch.clamp(x_adv_prime - x_orig, min=-epsilon, max=epsilon)
        x_adv_prime = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv_prime
