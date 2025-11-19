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
        eps: float,
        alpha: float,
        num_iter: int,
        mean_t: Tensor,
        std_t: Tensor,
        *,
        # 画像の値域クリップ用パラメータ
        # 画像（正規化後）の最小値・最大値は，0から1に収まるため
        min_val: float = 0.0, 
        max_val: float = 1.0, 
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

    eps_norm = eps / std_t
    alpha_norm = alpha / std_t

    min_val_norm = (min_val - mean_t) / std_t
    max_val_norm = (max_val - mean_t) / std_t

    # ターゲット整形
    if target is not None:
        target = target.to(device)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        target = target.long()

    loss_fn = nn.CrossEntropyLoss()

    # eps_norm / alpha_norm をテンソル化してブロードキャスト可能にする

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
        delta = torch.clamp(delta, -eps_norm, eps_norm)
        perturbed_norm = (orig_norm_inner + delta).detach()
        # --- 追加: 画像の値域クリップ (Domain Constraint) ---
        # ※ NormTensorが正規化されている場合、min/max_valも正規化後の値を渡す必要がある
        perturbed_norm = torch.clamp(perturbed_norm, min=min_val_norm, max=max_val_norm)
        # ------------------------------------------------
    return NormTensor(perturbed_norm, input_norm.state)

def _to_norm_params(epsilon: float, alpha: float, std_t: Tensor) -> tuple[Tensor, Tensor]:
    """ヘルパ: ピクセル空間の epsilon/alpha を正規化空間のテンソルに変換する。

    戻り値は (eps_norm, alpha_norm) で、std_t と同じ形 [1,C,1,1] のテンソル。
    """
    # std_t はテンソルなのでブロードキャストされる
    eps_norm = torch.tensor(epsilon, device=std_t.device, dtype=std_t.dtype) / std_t
    alpha_norm = torch.tensor(alpha, device=std_t.device, dtype=std_t.dtype) / std_t
    return eps_norm, alpha_norm
