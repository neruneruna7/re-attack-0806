from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from re_attack_0806.utils.normTensor import *


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


def _to_norm_eps(epsilon: float, std_t: Tensor) -> Tensor:
    """ピクセル空間の epsilon を正規化空間のテンソルに変換する。

    戻り値は `std_t` と同じ形 [1,C,1,1] のテンソル。
    """
    return torch.tensor(epsilon, device=std_t.device, dtype=std_t.dtype) / std_t


def fgsm(input_norm: NormTensor,
              eps_norm: float | Tensor,
              target: Tensor,
              eval_model: nn.Module,
              device: torch.device,
              *,
              orig_norm: Optional[NormTensor] = None) -> NormTensor:
    """FGSM のコア（正規化空間での実装）。

    - `input_norm`: 正規化済みテンソルを持つ `NormTensor`。
    - `eps_norm`: 正規化空間での摂動許容量（スカラーまたはテンソル）。
    - `orig_norm`: クリップの基準（再攻撃時に使用）、未指定なら `input_norm` を基準とする。
    """
    input_norm_inner = input_norm.tensor.clone().detach().to(device)
    if orig_norm is None:
        orig_norm_inner = input_norm.tensor.clone().detach().to(device)
    else:
        orig_norm_inner = orig_norm.tensor.clone().detach().to(device)

    if target is not None:
        target = target.to(device)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        target = target.long()

    # eps_norm をテンソル化
    if not isinstance(eps_norm, Tensor):
        eps_norm = torch.tensor(eps_norm, device=input_norm_inner.device, dtype=input_norm_inner.dtype)

    req = input_norm_inner.clone().detach().requires_grad_(True)
    outputs = eval_model(req)
    loss = F.cross_entropy(outputs, target)

    grad_norm = torch.autograd.grad(loss, req, retain_graph=False, create_graph=False)[0]
    if grad_norm is None:
        raise RuntimeError("fgsm_norm: gradient is None")

    perturbed_norm = (input_norm_inner + eps_norm * grad_norm.sign()).detach()

    # delta を clamp（orig を基準）
    delta = perturbed_norm - orig_norm_inner
    delta = torch.max(torch.min(delta, eps_norm), -eps_norm)
    perturbed_norm = (orig_norm_inner + delta).detach()

    return NormTensor(perturbed_norm, input_norm.state)
