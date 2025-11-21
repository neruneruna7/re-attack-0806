"""BIM攻撃のストラテジー実装"""
import torch
import torch.nn as nn
from torch import Tensor

from re_attack_0806.strategies.base import AttackStrategy
from re_attack_0806.utils.normTensor import TensorWithState
from re_attack_0806.utils.config import AttackParams, BIMAttackParam, DatasetNorm
from re_attack_0806.attacks import bim


class BIMAttackStrategy(AttackStrategy):
    """BIM（Basic Iterative Method）攻撃のストラテジー
    
    この攻撃手法は、FGSM攻撃を複数回反復することで、
    より効果的な敵対的サンプルを生成します。
    """
    
    def execute(
        self,
        data_ts_norm: TensorWithState,
        target_ts: Tensor,
        model: nn.Module,
        device: torch.device,
        dataset_norm: DatasetNorm,
        attack_params: AttackParams
    ) -> TensorWithState:
        """BIM攻撃を実行
        
        Args:
            data_ts_norm: 正規化済みの入力データ
            target_ts: ターゲットラベル
            model: 攻撃対象のモデル
            device: 計算デバイス
            dataset_norm: データセット正規化情報
            attack_params: BIM攻撃パラメータ
            
        Returns:
            摂動を加えた正規化済みテンソル
            
        Raises:
            AssertionError: attack_params が BIMAttackParam でない場合
        """
        assert isinstance(attack_params, BIMAttackParam), "BIM攻撃にはBIMAttackParamが必要です"
        
        perturbed = bim.bim(
            data_ts_norm,
            target_ts,
            model,
            device,
            attack_params.epsilon,
            attack_params.alpha,
            attack_params.iters,
            dataset_norm.mean,
            dataset_norm.std
        )
        
        return perturbed
