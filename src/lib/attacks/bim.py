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

def bim_attack_norm(input_norm: Tensor,
                    eps_norm: float | Tensor,
                    alpha_norm: float | Tensor,
                    num_iter: int,
                    target: Tensor,
                    eval_model: nn.Module,
                    device: torch.device) -> Tensor:
    """Basic Iterative Method operating in *normalized* space.

    この関数は正規化済みテンソルを受け取り、正規化済みテンソルを返します。
    - 入力 / 出力: 正規化されたテンソル (通常は transforms.Normalize 後のテンソル)。
    - eps_norm / alpha_norm: 正規化空間での L-inf 制約とステップ幅（ピクセル空間ではなく正規化空間の値）。
    - eval_model: 評価モードであるべきモデル（関数はモデルの状態を変更しません。呼び出し側で eval() を保証してください）。

    Args:
        input_norm: 正規化済み入力テンソル [B,C,H,W]
        eps_norm: 正規化空間での最大摂動量 (L-inf)
        alpha_norm: 正規化空間でのステップサイズ
        num_iter: 反復回数
        target: ラベルテンソル (shape [B] または scalar)
        eval_model: 評価モードであることが前提の nn.Module
        device: 実行デバイス

    Returns:
        perturbed_norm: 正規化済みの摂動後テンソル（detach, clamp済み）
    """
    # 入力は既に正規化済みであることを想定する
    input_norm = input_norm.clone().detach().to(device)
    perturbed_norm = input_norm.clone().detach()
    orig_norm = input_norm.clone().detach()

    # ターゲットを正しいデバイスとデータ型に移す
    if target is not None:
        target = target.to(device)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        target = target.long()

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        # 勾配計算用に正規化された入力を用意
        perturbed_norm_req = perturbed_norm.clone().detach().requires_grad_(True)

        outputs = eval_model(perturbed_norm_req)
        loss = loss_fn(outputs, target)

        # 入力に対する勾配のみを取得して副作用を避ける
        grad_norm = torch.autograd.grad(loss, perturbed_norm_req, retain_graph=False, create_graph=False)[0]
        if grad_norm is None:
            raise RuntimeError("bim_attack_norm: gradient is None")

        # 正規化空間で更新を行う
        perturbed_norm = perturbed_norm.detach() + alpha_norm * grad_norm.sign()

        # orig_norm からの距離を eps_norm に射影して clamp
        delta = torch.clamp(perturbed_norm - orig_norm, min=-eps_norm, max=eps_norm)
        perturbed_norm = torch.clamp(orig_norm + delta, -float('inf'), float('inf')).detach()

    return perturbed_norm


def bim_reattack_norm(eval_model: nn.Module,
                      x_adv_norm: Tensor,
                      y_adv: Tensor,
                      device: torch.device,
                      eps_norm: float | Tensor = 0.3,
                      alpha_norm: float | Tensor = 0.05,
                      num_iter: int = 10) -> Tensor:
    """再攻撃ユーティリティ（正規化空間版）。

    - 入力/出力は正規化済みテンソル（eval_model は eval() が呼ばれていることを呼び出し側で保証してください）。
    - x_adv_norm を基準(orig) としてその周りに再攻撃を行い、正規化空間のテンソルを返します。
    """
    # 関数はモデルの状態を変更しない前提（eval_model は eval() が呼ばれていること）
    x_adv_prime = x_adv_norm.clone().detach().to(device)
    y_adv = y_adv.to(device)
    orig_norm = x_adv_norm.clone().detach().to(device)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        x_adv_req = x_adv_prime.clone().detach().requires_grad_(True)

        outputs = eval_model(x_adv_req)
        loss = loss_fn(outputs, y_adv)

        grad_norm = torch.autograd.grad(loss, x_adv_req, retain_graph=False, create_graph=False)[0]
        if grad_norm is None:
            raise RuntimeError("bim_reattack_norm: gradient is None")

        x_adv_prime = x_adv_prime.detach() + alpha_norm * grad_norm.sign()

        delta = torch.clamp(x_adv_prime - orig_norm, min=-eps_norm, max=eps_norm)
        x_adv_prime = torch.clamp(orig_norm + delta, -float('inf'), float('inf')).detach()

    return x_adv_prime


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

    perturbed_norm = bim_attack_norm(input_norm, eps_norm, alpha_norm, num_iter, target, model, device)

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

    perturbed_norm = bim_reattack_norm(model, x_adv_norm, y_adv, device, eps_norm, alpha_norm, num_iter)

    perturbed = perturbed_norm * std_t + mean_t
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()
    return perturbed
