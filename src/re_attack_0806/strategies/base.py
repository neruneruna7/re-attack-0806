"""攻撃ストラテジーの抽象基底クラス"""
from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.types import Number

from re_attack_0806.utils.normTensor import TensorWithState
from re_attack_0806.utils.config import AttackParams, DatasetNorm


class AttackStrategy(ABC):
    """攻撃手法の抽象基底クラス
    
    すべての攻撃ストラテジーはこのクラスを継承し、execute メソッドを実装する必要があります。
    これにより、攻撃手法の実装を統一的に扱うことができます。
    """
    
    @abstractmethod
    def execute(
        self,
        data_ts_norm: TensorWithState,
        target_ts: Tensor,
        model: nn.Module,
        device: torch.device,
        dataset_norm: DatasetNorm,
        attack_params: AttackParams
    ) -> TensorWithState:
        """攻撃を実行して、摂動を加えたテンソルを返す
        
        Args:
            data_ts_norm: 正規化済みの入力データ
            target_ts: ターゲットラベル
            model: 攻撃対象のモデル
            device: 計算デバイス
            dataset_norm: データセット正規化情報
            attack_params: 攻撃パラメータ
            
        Returns:
            摂動を加えた正規化済みテンソル
        """
        pass
