import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor

# FGSM attack code
def fgsm_attack(image: Tensor, epsilon: float, target: Tensor, model: nn.Module, device: torch.device) -> Tensor:
    """
    image: 非正規化されたテンソル [B,C,H,W] (値域 0..1)
    epsilon: ピクセルスケールの摂動量
    target: 正解ラベルテンソル (device 上)
    戻り値: perturbed_image (非正規化, clamp され detach 済み)
    """
    

    # mean/std on device
    mean = torch.tensor([0.1307], dtype=image.dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor([0.3081], dtype=image.dtype, device=device).view(1, -1, 1, 1)

    # prepare normalized input as a leaf with requires_grad
    image_denorm = image.clone().detach().to(device)
    image_norm = (image_denorm - mean) / std
    image_norm = image_norm.clone().detach().requires_grad_(True)
    
    # forward / loss / backward (local, does not modify external tensors)
    output = model(image_norm)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    # backward to get dL/dx_norm
    loss.backward()
    data_grad_norm = image_norm.grad.data  # gradient w.r.t. normalized input
    print(f"data_grad_norm: {data_grad_norm}")

    # convert gradient to pixel space: dL/dx_pixel = dL/dx_norm * (1/std)
    grad_pixel = data_grad_norm / std

    # FGSM in pixel space
    sign_data_grad = grad_pixel.sign()
    perturbed = image_denorm + epsilon * sign_data_grad
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()

    # cleanup grads to avoid side effects
    image_norm.grad = None

    return perturbed

def bim_attack(image: Tensor, epsilon: float, alpha: float, n: int, target: Tensor, model: nn.Module, device: torch.device) -> Tensor:
    """
    image: 非正規化されたテンソル [B,C,H,W] (値域 0..1)
    epsilon: 摂動量
    alpha: ステップサイズ
    n: 反復回数
    target: 正解ラベルテンソル (device 上)
    戻り値: perturbed_image (非正規化, clamp され detach 済み)
    """
    # mean/std on device
    mean = torch.tensor([0.1307], dtype=image.dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor([0.3081], dtype=image.dtype, device=device).view(1, -1, 1, 1)

    # prepare normalized input as a leaf with requires_grad
    image_denorm = image.clone().detach().to(device)

    perturbed = image_denorm.clone().detach()

    for _ in range(n):
        image_norm = (image_denorm - mean) / std
        image_norm = image_norm.clone().detach().requires_grad_(True)

        # forward / loss / backward (local, does not modify external tensors)
        output = model(image_norm)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        # backward to get dL/dx_norm
        loss.backward()
        data_grad_norm = image_norm.grad.data  # gradient w.r.t. normalized input

        # convert gradient to pixel space: dL/dx_pixel = dL/dx_norm * (1/std)
        grad_pixel = data_grad_norm / std

        # FGSM in pixel space
        sign_data_grad = grad_pixel.sign()
        perturbed = perturbed + alpha * sign_data_grad

        # 元画像からの接道を epsilon 以内にクリップ
        delta = torch.clamp(perturbed - image_denorm, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(image_denorm + delta, 0.0, 1.0).detach()

    # cleanup grads to avoid side effects
    # image_norm.grad = None

    return perturbed
