"""Foolbox BIM攻撃のストラテジー実装"""
import torch
import torch.nn as nn
from torch import Tensor
import foolbox

from re_attack_0806.strategies.base import AttackStrategy
from re_attack_0806.strategies.foolbox_utils import FoolboxModelCreator
from re_attack_0806.utils.normTensor import TensorWithState, DENORMALIZED
from re_attack_0806.utils.config import AttackParams, BIMAttackParam, DatasetNorm


class FoolboxBIMAttackStrategy(AttackStrategy):
    """Foolbox BIM攻撃のストラテジー
    
    Foolboxライブラリを使用したBIM攻撃の実装です。
    Foolboxは敵対的攻撃の標準的なライブラリであり、
    多くの攻撃手法を統一的なインターフェースで提供します。
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
        """Foolbox BIM攻撃を実行
        
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
            RuntimeError: Foolbox攻撃が失敗した場合
        """
        assert isinstance(attack_params, BIMAttackParam), "FoolboxBIM攻撃にはBIMAttackParamが必要です"
        
        # Foolboxモデルの初期化（共通ユーティリティを使用）
        fmodel = FoolboxModelCreator.create_foolbox_model(model, dataset_norm, device)
        
        # 攻撃の実行
        foolbox_attack = foolbox.attacks.LinfBasicIterativeAttack(
            steps=attack_params.iters,
            abs_stepsize=attack_params.alpha
        )
        
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
            raise RuntimeError(f"Foolbox BIM攻撃が失敗しました: {e}") from e
