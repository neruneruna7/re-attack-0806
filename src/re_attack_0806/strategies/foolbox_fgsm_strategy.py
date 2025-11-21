"""Foolbox FGSM攻撃のストラテジー実装"""
import torch
import torch.nn as nn
from torch import Tensor
import foolbox

from re_attack_0806.strategies.base import AttackStrategy
from re_attack_0806.strategies.foolbox_utils import FoolboxModelCreator
from re_attack_0806.utils.normTensor import TensorWithState, DENORMALIZED
from re_attack_0806.utils.config import AttackParams, FGSMAttackParam, DatasetNorm


class FoolboxFGSMAttackStrategy(AttackStrategy):
    """Foolbox FGSM攻撃のストラテジー
    
    Foolboxライブラリを使用したFGSM攻撃の実装です。
    標準的な実装を使用することで、再現性と信頼性を確保します。
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
        """Foolbox FGSM攻撃を実行
        
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
            RuntimeError: Foolbox攻撃が失敗した場合
        """
        assert isinstance(attack_params, FGSMAttackParam), "FoolboxFGSM攻撃にはFGSMAttackParamが必要です"
        
        # Foolboxモデルの初期化（共通ユーティリティを使用）
        fmodel = FoolboxModelCreator.create_foolbox_model(model, dataset_norm, device)
        
        # 攻撃の実行
        foolbox_attack = foolbox.attacks.FGSM()
        
        # 非正規化データを取得
        data_denorm = dataset_norm.denormalize(data_ts_norm)
        
        try:
            raw, clipped, is_adv = foolbox_attack(
                fmodel,
                data_denorm.tensor,
                target_ts,
                epsilons=attack_params.epsilon
            )
            
            # 結果の処理
            perturbed_denorm = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
            perturbed_denorm_ts = TensorWithState(perturbed_denorm, DENORMALIZED)
            perturbed = dataset_norm.normalize(perturbed_denorm_ts)
            
            return perturbed
            
        except Exception as e:
            raise RuntimeError(f"Foolbox FGSM攻撃が失敗しました: {e}") from e
