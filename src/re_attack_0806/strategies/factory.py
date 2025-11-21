"""攻撃ストラテジーファクトリー"""
from re_attack_0806.strategies.base import AttackStrategy
from re_attack_0806.strategies.bim_strategy import BIMAttackStrategy
from re_attack_0806.strategies.fgsm_strategy import FGSMAttackStrategy
from re_attack_0806.strategies.foolbox_bim_strategy import FoolboxBIMAttackStrategy
from re_attack_0806.strategies.foolbox_fgsm_strategy import FoolboxFGSMAttackStrategy
from re_attack_0806.utils.config import AttackKind


class AttackStrategyFactory:
    """攻撃ストラテジーを生成するファクトリークラス
    
    AttackKindに基づいて適切な攻撃ストラテジーインスタンスを生成します。
    このファクトリーパターンにより、新しい攻撃手法の追加が容易になります。
    """
    
    @staticmethod
    def create(attack_kind: AttackKind) -> AttackStrategy:
        """攻撃ストラテジーを生成
        
        Args:
            attack_kind: 攻撃の種類
            
        Returns:
            対応する攻撃ストラテジーのインスタンス
            
        Raises:
            ValueError: サポートされていない攻撃種別の場合
        """
        if attack_kind == AttackKind.BIM or attack_kind == AttackKind.LINF_BIM:
            return BIMAttackStrategy()
        elif attack_kind == AttackKind.FGSM:
            return FGSMAttackStrategy()
        elif attack_kind == AttackKind.FOOLBOX_BIM:
            return FoolboxBIMAttackStrategy()
        elif attack_kind == AttackKind.FOOLBOX_FGSM:
            return FoolboxFGSMAttackStrategy()
        else:
            raise ValueError(f"サポートされていない攻撃種別です: {attack_kind}")
