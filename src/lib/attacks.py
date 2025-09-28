import torch
import torch.nn as nn
import torch.nn.functional as F
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
    model.zero_grad()
    output = model(image_norm)
    output = F.log_softmax(output, dim=1)

    # loss = F.cross_entropy(output, target)
    loss = F.nll_loss(output, target)

    # backward to get dL/dx_norm
    loss.backward()

    grad: Tensor | None = image_denorm.grad
    if grad is None :
        raise RuntimeError("grad is None. model may not have been called with requires_grad input.")
    data_grad_norm = grad.detach()  # gradient w.r.t. normalized input
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

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

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
        grad = image_norm.grad
        if grad is None :
            raise RuntimeError("grad is None. model may not have been called with requires_grad input.")
        data_grad_norm = grad.data  # gradient w.r.t. normalized input

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

import torch
import torch.nn as nn

def bim_reattack(model: nn.Module, x_adv: Tensor, y_adv: Tensor, device: torch.device, epsilon=0.3, alpha=0.05, num_iter=10, ) -> Tensor:
    """
    BIMを用いた再攻撃 (Re-attack using BIM)

    Args:
        model: PyTorchの分類モデル
        x_adv: 検出された敵対的サンプル (torch.Tensor, shape: [B, C, H, W])
        y_adv: x_advに対応するラベル (torch.Tensor, shape: [B])
        epsilon: 最大摂動量 (L∞ノルム制約)
        alpha: ステップサイズ
        num_iter: 反復回数
        device: "cuda" または "cpu"

    Returns:
        x_adv_prime: 再攻撃後の敵対的サンプル
    """
 
    # モデルを評価モードに
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    # 入力をコピーして再攻撃用に準備
    x_adv_prime = x_adv.clone().detach().to(device)
    y_adv = y_adv.to(device)

    # 元の入力を保存（クリップの基準）
    x_orig = x_adv.clone().detach().to(device)

    for _ in range(num_iter):
        x_adv_prime.requires_grad = True

        # 順伝播
        outputs = model(x_adv_prime)
        loss = loss_fn(outputs, y_adv)

        # 勾配計算
        grad = torch.autograd.grad(loss, x_adv_prime,
                                   retain_graph=False,
                                   create_graph=False)[0]
        
        # print(f"grad: {grad}")
        # print(f"grad.sign(): {grad.sign()}")


        # 勾配の符号方向に摂動を加える
        x_adv_prime = x_adv_prime.detach() + alpha * grad.sign()

        # 元の入力からε以内にクリップ
        delta = torch.clamp(x_adv_prime - x_orig, min=-epsilon, max=epsilon)
        x_adv_prime = torch.clamp(x_orig + delta, 0, 1).detach()

    return x_adv_prime
