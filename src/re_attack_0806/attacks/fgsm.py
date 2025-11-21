from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from re_attack_0806.utils.normTensor import *


def fgsm(input_norm: NormTensor,
        target: Tensor,
        eval_model: nn.Module,
        device: torch.device,
        eps: float,
        mean_t: Tensor,
        std_t: Tensor,
        *,
        min_val: float = 0.0, 
        max_val: float = 1.0, 
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

    # もっともfoolbox実装に近づけるため，min/max_valも正規化する 
    # 正規化しないと，出力される画像が異様になる
    # 正規化するのが正解と思われる

    eps_norm = eps / std_t
    min_val_norm = (min_val - mean_t) / std_t
    max_val_norm = (max_val - mean_t) / std_t


    if target is not None:
        target = target.to(device)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        target = target.long()


    req = input_norm_inner.clone().detach().requires_grad_(True)
    outputs = eval_model(req)
    loss = F.cross_entropy(outputs, target)

    grad_norm = torch.autograd.grad(loss, req, retain_graph=False, create_graph=False)[0]
    if grad_norm is None:
        raise RuntimeError("fgsm_norm: gradient is None")

    perturbed_norm = (input_norm_inner + eps_norm * grad_norm.sign()).detach()

    perturbed_norm = torch.clamp(perturbed_norm, min=min_val_norm, max=max_val_norm)

    return NormTensor(perturbed_norm, input_norm.state)
