"""攻撃ストラテジーファクトリー"""
from typing import Dict, Type

from re_attack_0806.strategies.base import AttackStrategy
from re_attack_0806.strategies.bim_strategy import BIMAttackStrategy
from re_attack_0806.strategies.fgsm_strategy import FGSMAttackStrategy
from re_attack_0806.strategies.foolbox_bim_strategy import FoolboxBIMAttackStrategy
from re_attack_0806.strategies.foolbox_fgsm_strategy import FoolboxFGSMAttackStrategy
from re_attack_0806.utils.config import AttackKind


class AttackStrategyFactory:
    """攻撃ストラテジーを生成するファクトリークラス
    
    AttackKindに基づいて適切な攻撃ストラテジーインスタンスを生成します。
    ディクショナリマッピングを使用することで、新しい攻撃手法の追加が容易になります。
    """
    
    # 攻撃種別からストラテジークラスへのマッピング
    _STRATEGY_MAP: Dict[AttackKind, Type[AttackStrategy]] = {
        AttackKind.BIM: BIMAttackStrategy,
        AttackKind.LINF_BIM: BIMAttackStrategy,
        AttackKind.FGSM: FGSMAttackStrategy,
        AttackKind.FOOLBOX_BIM: FoolboxBIMAttackStrategy,
        AttackKind.FOOLBOX_FGSM: FoolboxFGSMAttackStrategy,
    }
    
    @classmethod
    def create(cls, attack_kind: AttackKind) -> AttackStrategy:
        """攻撃ストラテジーを生成
        
        Args:
            attack_kind: 攻撃の種類
            
        Returns:
            対応する攻撃ストラテジーのインスタンス
            
        Raises:
            ValueError: サポートされていない攻撃種別の場合
        """
        strategy_class = cls._STRATEGY_MAP.get(attack_kind)
        
        if strategy_class is None:
            raise ValueError(f"サポートされていない攻撃種別です: {attack_kind}")
        
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, attack_kind: AttackKind, strategy_class: Type[AttackStrategy]):
        """新しい攻撃ストラテジーを登録
        
        このメソッドを使用することで、実行時に新しい攻撃手法を追加できます。
        
        Args:
            attack_kind: 攻撃の種類
            strategy_class: 攻撃ストラテジークラス
        """
        cls._STRATEGY_MAP[attack_kind] = strategy_class

