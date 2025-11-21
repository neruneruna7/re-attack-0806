"""Foolbox関連の共通ユーティリティ"""
import torch
import torch.nn as nn
import foolbox

from re_attack_0806.utils.config import DatasetNorm


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
