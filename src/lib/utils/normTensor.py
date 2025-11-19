# -*- coding: utf-8 -*-
"""テンソルとその正規化状態を保持するユーティリティ（Rustのファントム型風）。

このモジュールは以下を提供します：
- `TensorWithState[S]`: テンソルと状態を持つ不変データコンテナ。
- 状態マーカー `Denormalized` / `Normalized` とシングルトン `DENORMALIZED` / `NORMALIZED`。
- 型エイリアス `DenormTensor` / `NormTensor`。

メソッド `normalize` / `denormalize` は元のインスタンスを変更せず新しいインスタンスを返します。

注意: 正規化/逆正規化の実処理は同パッケージの `normalize_tensor` / `denormalize_tensor` を利用します。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Final

import torch


class _BaseState:
    """状態マーカーの基底クラス（型パラメータ用）。"""
    pass


class Denormalized(_BaseState):
    """非正規化（ピクセル空間）を表すマーカー。"""
    pass


class Normalized(_BaseState):
    """正規化済（mean/std が適用された空間）を表すマーカー。"""
    pass


# 実行時に使うシングルトンマーカー
DENORMALIZED: Final[Denormalized] = Denormalized()
NORMALIZED: Final[Normalized] = Normalized()


S = TypeVar("S", bound=_BaseState)


@dataclass(frozen=True)
class TensorWithState(Generic[S]):
    """テンソルと状態を保持する単純なデータクラス。

    このクラスはデータのみを保持します。正規化やデバイス移動などの
    ヘルパ関数は持ちません（純粋なデータ定義として使ってください）。

    Attributes
    - tensor: `torch.Tensor`（形状は [B,C,H,W] を想定）
    - state: `Denormalized` または `Normalized` のインスタンス
    """

    tensor: torch.Tensor
    state: S


# 型エイリアス（静的解析やドキュメント用）
NormTensor = TensorWithState[Normalized]
DenormTensor = TensorWithState[Denormalized]


__all__ = [
    "TensorWithState",
    "Denormalized",
    "Normalized",
    "DENORMALIZED",
    "NORMALIZED",
    "NormTensor",
    "DenormTensor",
]
