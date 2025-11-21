"""Foolbox関連の共通ユーティリティ"""
import torch
import torch.nn as nn
from torch import Tensor
import foolbox

from re_attack_0806.utils.config import DatasetNorm
from re_attack_0806.utils.normTensor import TensorWithState, DENORMALIZED


class FoolboxModelCreator:
    """Foolboxモデルを作成するためのユーティリティクラス
    
    このクラスは、FoolboxのPyTorchModelを作成する共通処理を提供します。
    DRY原則に従い、重複コードを排除しています。
    """
    
    @staticmethod
    def create_foolbox_model(
        model: nn.Module,
        dataset_norm: DatasetNorm,
        device: torch.device
    ) -> foolbox.PyTorchModel:
        """Foolboxモデルを作成
        
        Args:
            model: PyTorchモデル
            dataset_norm: データセット正規化情報
            device: 計算デバイス
            
        Returns:
            Foolboxモデル
        """
        mean_list = dataset_norm.mean.squeeze().tolist()
        std_list = dataset_norm.std.squeeze().tolist()
        preprocessing = dict(mean=mean_list, std=std_list, axis=-3)
        bounds = (0.0, 1.0)
        
        return foolbox.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device
        )
    
    @staticmethod
    def extract_perturbed_tensor(foolbox_result: tuple) -> Tensor:
        """Foolbox攻撃結果から摂動を加えたテンソルを抽出
        
        Foolboxの攻撃メソッドは (raw, clipped, is_adv) のタプルを返します。
        clippedはテンソルまたはテンソルのリスト/タプルの場合があります。
        
        Args:
            foolbox_result: Foolbox攻撃メソッドの戻り値 (raw, clipped, is_adv)
            
        Returns:
            摂動を加えたテンソル
        """
        _, clipped, _ = foolbox_result
        
        if isinstance(clipped, (list, tuple)):
            return clipped[0]
        return clipped
