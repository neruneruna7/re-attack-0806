"""FGSM攻撃のストラテジー実装"""
import torch
import torch.nn as nn
from torch import Tensor

from re_attack_0806.strategies.base import AttackStrategy
from re_attack_0806.utils.normTensor import TensorWithState
from re_attack_0806.utils.config import AttackParams, FGSMAttackParam, DatasetNorm
from re_attack_0806.attacks import fgsm


class FGSMAttackStrategy(AttackStrategy):
    """FGSM（Fast Gradient Sign Method）攻撃のストラテジー
    
    この攻撃手法は、勾配の符号を利用して効率的に敵対的サンプルを生成します。
    計算コストが低く、基本的な攻撃手法として広く使用されています。
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
        """FGSM攻撃を実行
        
        Args:
            data_ts_norm: 正規化済みの入力データ
            target_ts: ターゲットラベル
            model: 攻撃対象のモデル
            device: 計算デバイス
            dataset_norm: データセット正規化情報
            attack_params: FGSM攻撃パラメータ
            
        Returns:
            摂動を加えた正規化済みテンソル
            
        Raises:
            AssertionError: attack_params が FGSMAttackParam でない場合
        """
        assert isinstance(attack_params, FGSMAttackParam), "FGSM攻撃にはFGSMAttackParamが必要です"
        
        perturbed = fgsm.fgsm(
            data_ts_norm,
            target_ts,
            model,
            device,
            attack_params.epsilon,
            dataset_norm.mean,
            dataset_norm.std
        )
        
        return perturbed
