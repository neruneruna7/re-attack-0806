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
        # CIFAR-ish defaults
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


def fgsm_attack(image: Tensor, epsilon: float, target: Tensor, model: nn.Module, device: torch.device, *,
                mean: Optional[Tensor] = None, std: Optional[Tensor] = None) -> Tensor:
    """One-step FGSM attack.

    Args:
        image: 非正規化された入力テンソル [B,C,H,W], 値域 0..1
        epsilon: ピクセルスケールでの摂動量
        target: 正解ラベルテンソル ([B])
        model: PyTorch モデル
        device: 実行デバイス
        mean, std: 正規化用の平均 / 標準偏差（テンソル、形状 [1,C,1,1]）。指定しない場合はチャネル数に応じた既定値を使用。

    Returns:
        perturbed: 非正規化された摂動後テンソル（detach され clamp 済み）
    """
    # コピーしてデバイスに移す（純粋関数に近づけるためオブジェクトを変更しない）
    image_denorm = image.clone().detach().to(device)

    B, C, H, W = image_denorm.shape
    if mean is None or std is None:
        mean_t, std_t = _default_mean_std(C, device, image_denorm.dtype)
    else:
        mean_t = mean.to(device).view(1, -1, 1, 1)
        std_t = std.to(device).view(1, -1, 1, 1)

    # 正規化した入力をrequires_grad True の leaf として用意
    image_norm = (image_denorm - mean_t) / std_t
    image_norm = image_norm.clone().detach().requires_grad_(True)

    model.zero_grad()
    outputs = model(image_norm)
    loss = F.cross_entropy(outputs, target)
    loss.backward()

    grad_norm = image_norm.grad
    if grad_norm is None:
        raise RuntimeError("FGSM: gradient is None. Ensure model was called with requires_grad input.")

    # 正規化空間の勾配 -> ピクセル空間の勾配
    grad_pixel = grad_norm / std_t

    sign_grad = grad_pixel.sign()
    perturbed = image_denorm + epsilon * sign_grad
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()

    # cleanup
    image_norm.grad = None

    return perturbed
