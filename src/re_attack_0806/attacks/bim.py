from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from re_attack_0806.utils.normTensor import *


def bim(input_norm: NormTensor,
        target: Tensor,
        eval_model: nn.Module,
        device: torch.device,
        eps_norm: float | Tensor,
        alpha_norm: float | Tensor,
        num_iter: int,
        *,
        orig_norm: Optional[NormTensor] = None) -> NormTensor:
    """汎用 BIM 実装（正規化空間）。

    - input_norm: 攻撃開始点（正規化済みテンソル）。
    - orig_norm: クリップの基準（None の場合は input_norm を基準とする＝新規攻撃）。
    - target: ラベルテンソル（スカラーや [B] を受け付ける）。
    - eval_model: eval() 済みのモデル（関数内部で状態を変更しない）。

    これにより attack / reattack の両方の用途を同じ実装で扱える。
    """
    input_norm_inner = input_norm.tensor.clone().detach().to(device)
    perturbed_norm = input_norm_inner.clone().detach()
    if orig_norm is None:
        orig_norm_inner = input_norm.tensor.clone().detach()
    else:
        orig_norm_inner = orig_norm.tensor.clone().detach().to(device)

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
        # 最終的に摂動の大きさをepsilonで制限する
        delta = perturbed_norm - orig_norm_inner
        delta = torch.max(torch.min(delta, eps_norm), -eps_norm)
        perturbed_norm = (orig_norm_inner + delta).detach()
    return NormTensor(perturbed_norm, input_norm.state)

def _to_norm_params(epsilon: float, alpha: float, std_t: Tensor) -> tuple[Tensor, Tensor]:
    """ヘルパ: ピクセル空間の epsilon/alpha を正規化空間のテンソルに変換する。

    戻り値は (eps_norm, alpha_norm) で、std_t と同じ形 [1,C,1,1] のテンソル。
    """
    # std_t はテンソルなのでブロードキャストされる
    eps_norm = torch.tensor(epsilon, device=std_t.device, dtype=std_t.dtype) / std_t
    alpha_norm = torch.tensor(alpha, device=std_t.device, dtype=std_t.dtype) / std_t
    return eps_norm, alpha_norm
