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

def bim(input_norm: Tensor,
        target: Tensor,
        eval_model: nn.Module,
        device: torch.device,
        eps_norm: float | Tensor,
        alpha_norm: float | Tensor,
        num_iter: int,
        *,
        orig_norm: Optional[Tensor] = None) -> Tensor:
    """汎用 BIM 実装（正規化空間）。

    - input_norm: 攻撃開始点（正規化済みテンソル）。
    - orig_norm: クリップの基準（None の場合は input_norm を基準とする＝新規攻撃）。
    - target: ラベルテンソル（スカラーや [B] を受け付ける）。
    - eval_model: eval() 済みのモデル（関数内部で状態を変更しない）。

    これにより attack / reattack の両方の用途を同じ実装で扱える。
    """
    input_norm = input_norm.clone().detach().to(device)
    perturbed_norm = input_norm.clone().detach()
    if orig_norm is None:
        orig_norm = input_norm.clone().detach()
    else:
        orig_norm = orig_norm.clone().detach().to(device)

    # ターゲット整形
    if target is not None:
        target = target.to(device)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        target = target.long()

    loss_fn = nn.CrossEntropyLoss()

    # eps_norm / alpha_norm をテンソル化してブロードキャスト可能にする
    if not isinstance(eps_norm, Tensor):
        eps_norm = torch.tensor(eps_norm, device=perturbed_norm.device, dtype=perturbed_norm.dtype)
    if not isinstance(alpha_norm, Tensor):
        alpha_norm = torch.tensor(alpha_norm, device=perturbed_norm.device, dtype=perturbed_norm.dtype)

    for _ in range(num_iter):
        req = perturbed_norm.clone().detach().requires_grad_(True)
        outputs = eval_model(req)
        loss = loss_fn(outputs, target)

        grad_norm = torch.autograd.grad(loss, req, retain_graph=False, create_graph=False)[0]
        if grad_norm is None:
            raise RuntimeError("bim: gradient is None")

        perturbed_norm = perturbed_norm.detach() + alpha_norm * grad_norm.sign()

        # delta を elementwise に clamp する（eps_norm はテンソルでもスカラーでも対応）
        delta = perturbed_norm - orig_norm
        delta = torch.max(torch.min(delta, eps_norm), -eps_norm)
        perturbed_norm = (orig_norm + delta).detach()
    return perturbed_norm


def _to_norm_params(epsilon: float, alpha: float, std_t: Tensor) -> tuple[Tensor, Tensor]:
    """ヘルパ: ピクセル空間の epsilon/alpha を正規化空間のテンソルに変換する。

    戻り値は (eps_norm, alpha_norm) で、std_t と同じ形 [1,C,1,1] のテンソル。
    """
    # std_t はテンソルなのでブロードキャストされる
    eps_norm = torch.tensor(epsilon, device=std_t.device, dtype=std_t.dtype) / std_t
    alpha_norm = torch.tensor(alpha, device=std_t.device, dtype=std_t.dtype) / std_t
    return eps_norm, alpha_norm


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
    """従来互換ラッパー: 非正規化入力を受け取り、非正規化出力を返す。

    内部で正規化して `bim_attack_norm` を呼び出します。
    """
    image_denorm = image.clone().detach().to(device)
    B, C, H, W = image_denorm.shape
    if mean is None or std is None:
        _, std_t = _default_mean_std(C, device, image_denorm.dtype)
        mean_t, _ = _default_mean_std(C, device, image_denorm.dtype)
    else:
        mean_t = mean.to(device).view(1, -1, 1, 1)
        std_t = std.to(device).view(1, -1, 1, 1)

    # normalize
    input_norm = (image_denorm - mean_t) / std_t

    eps_norm, alpha_norm = _to_norm_params(epsilon, alpha, std_t)

    perturbed_norm = bim(input_norm, target, model, device, eps_norm, alpha_norm, num_iter)

    # convert back to denorm (pixel space)
    perturbed = perturbed_norm * std_t + mean_t
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()
    return perturbed


def bim_reattack(model: nn.Module, x_adv: Tensor, y_adv: Tensor, device: torch.device, epsilon: float = 0.3,
                 alpha: float = 0.05, num_iter: int = 10,
                 mean: Optional[Tensor] = None,
                 std: Optional[Tensor] = None) -> Tensor:
    """従来互換ラッパー: 非正規化された adversarial を受け取り、非正規化で返す。

    内部で正規化して `bim_reattack_norm` を呼び出します。
    """
    x_adv_denorm = x_adv.clone().detach().to(device)
    B, C, H, W = x_adv_denorm.shape
    if mean is None or std is None:
        mean_t, std_t = _default_mean_std(C, device, x_adv_denorm.dtype)
    else:
        mean_t = mean.to(device).view(1, -1, 1, 1)
        std_t = std.to(device).view(1, -1, 1, 1)

    x_adv_norm = (x_adv_denorm - mean_t) / std_t

    eps_norm, alpha_norm = _to_norm_params(epsilon, alpha, std_t)

    perturbed_norm = bim(x_adv_norm, y_adv, model, device, eps_norm, alpha_norm, num_iter, orig_norm=x_adv_norm)

    perturbed = perturbed_norm * std_t + mean_t
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()
    return perturbed
