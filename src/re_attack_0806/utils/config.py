from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypeAlias, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch import Tensor

# 型別名
from enum import Enum
import argparse
import re_attack_0806
from copy import deepcopy

from re_attack_0806.models import MorimotoCifar10, MorimotoMnist, Ploof
from re_attack_0806.utils.normTensor import * 


class DatasetKind(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    IMAGE_NET = "imagenet"

class ModelKind(str, Enum):
    MORIMOTO_MNIST = "morimoto-mnist"
    MORIMOTO_CIFAR10 = "morimoto-cifar10"
    INCEPTION_V3 = "inception-v3"
    PLOOF = "ploof"

class AttackKind(str, Enum):
    BIM = "bim"
    FGSM = "fgsm"
    FOOLBOX_FGSM = "foolbox-fgsm"
    FOOLBOX_BIM = "foolbox-bim"
    LINF_BIM = "linf-bim"

@dataclass
class FGSMAttackParam:
    epsilon: float
    batch_size: int

@dataclass
class BIMAttackParam:
    epsilon: float
    alpha: float
    iters: int
    batch_size: int

AttackParams = Union[FGSMAttackParam, BIMAttackParam]


from typing import Type


class AttackParamKind(Enum):
    """攻撃ごとのパラメータ型を列挙する Enum。

    各メンバの値は (value_str, ParamClass) のタプルで、`.default()` で初期値インスタンスを取得できます。
    """
    FGSM = ("fgsm", FGSMAttackParam)
    BIM = ("bim", BIMAttackParam)
    FOOLBOX_BIM = ("foolbox_bim", BIMAttackParam)

    @property
    def param_class(self) -> Type:
        return self.value[1]


# AttackKind -> AttackParamKind のマッピング（必要に応じて利用）
ATTACK_KIND_TO_PARAM: dict[AttackKind, AttackParamKind] = {
    AttackKind.FGSM: AttackParamKind.FGSM,
    AttackKind.BIM: AttackParamKind.BIM,
    AttackKind.FOOLBOX_FGSM: AttackParamKind.FGSM,
    AttackKind.FOOLBOX_BIM: AttackParamKind.FOOLBOX_BIM,
    AttackKind.LINF_BIM: AttackParamKind.BIM,
}

class DatasetNorm:
    """データセット種別に基づく正規化/非正規化を行うユーティリティクラス。

    引数 dataset_kind は `DatasetKind`（またはその value 文字列）を受け取る。
    内部で mean/std を決定し、テンソルの正規化/非正規化を行う。
    """
    def __init__(self, dataset_kind: DatasetKind, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        # dataset_kind が Enum の場合は value を使う（循環 import を避けるため、柔軟に受け入れる）
        self.device = device
        self.dtype = dtype

        if dataset_kind == DatasetKind.MNIST:
            mean = [0.1307]
            std = [0.3081]
        elif dataset_kind == DatasetKind.CIFAR10:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
        elif dataset_kind == DatasetKind.IMAGE_NET:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"unsupported dataset kind: {dataset_kind}")

        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype).view(1, -1, 1, 1)
        self.mean_list = mean
        self.std_list = std

    def normalize(self, x: DenormTensor) -> NormTensor:
        """非正規化テンソルを受け取り、正規化テンソルを返す。"""
        x_tensor = x.tensor.to(self.device)
        tensor = (x_tensor - self.mean) / self.std
        return TensorWithState(tensor, NORMALIZED)

    def denormalize(self, x: NormTensor) -> DenormTensor:
        """正規化テンソルを受け取り、非正規化テンソルを返す。"""
        x_tensor = x.tensor.to(self.device)
        tensor = x_tensor * self.std + self.mean
        return TensorWithState(tensor, DENORMALIZED)

class ModelFactory:
    @staticmethod
    def create(kind: ModelKind, device: torch.device) -> nn.Module:
        if kind == ModelKind.MORIMOTO_MNIST:
            return MorimotoMnist.MnistNet().to(device)
        if kind == ModelKind.MORIMOTO_CIFAR10:
            return MorimotoCifar10.Cifar10Net().to(device)
        if kind == ModelKind.PLOOF:
            # MNIST前提になってしまってる．このファクトリーの仕組みも問題がある．
            # 改善が必要
            return Ploof.PloofNet(10).to(device)
        if kind == ModelKind.INCEPTION_V3:
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=True, aux_logits=True)
            return model.to(device)
        raise ValueError(f"unsupported model kind: {kind}")

class DataFactory:
    @staticmethod
    def loader(kind: DatasetKind, batch_size: int, train: bool, shuffle: bool) -> torch.utils.data.DataLoader:
        if kind == DatasetKind.MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ds = datasets.MNIST('../data', train=train, download=True, transform=transform)
        elif kind == DatasetKind.CIFAR10:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ds = datasets.CIFAR10('../data', train=train, download=True, transform=transform)
        elif kind == DatasetKind.IMAGE_NET:
            transform = transforms.Compose([
            # 1. 短辺を342にリサイズ (299より少し大きくしておくのが一般的です)
                #    ※PyTorch公式のInception v3の推奨は Resize(299) -> CenterCrop(299) ですが、
                #      ここではより一般的に精度が出やすい Resize(342) -> CenterCrop(299) を推奨します。
                transforms.Resize(299), 
                # 2. 中央の299x299を切り出す
                transforms.CenterCrop(299), 
                transforms.ToTensor(),
            ])
            # torchvision.datasets.ImageNet は devkit（ILSVRC2012_devkit_t12.tar.gz）や
            # 画像アーカイブが所定の場所にないと RuntimeError を投げることがある。
            # その場合はローカルに展開済みの val ディレクトリを ImageFolder で読み込むフォールバックを行う。
            try:
                ds = datasets.ImageFolder('./data/image_net/ILSVRC2012_img_val', transform=transform)
            except RuntimeError as e:
                # 典型的なエラーメッセージ例:
                # "The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted."
                print(f"datasets.ImageNet failed: {e}")
                alt_dir = '../data/image_net/ILSVRC2012_img_val'
                if os.path.exists(alt_dir) and os.path.isdir(alt_dir):
                    print(f"Falling back to ImageFolder at {alt_dir}")
                    ds = datasets.ImageFolder(alt_dir, transform=transform)
                else:
                    raise RuntimeError(
                        "ImageNet dataset not found or devkit missing.\n"
                        "Please download the ILSVRC2012 data and devkit and place them under ../data/image_net,\n"
                        "or prepare a validation folder at ../data/image_net/ILSVRC2012_img_val and retry.\n"
                        "See README or ImageNet official site for instructions to obtain ILSVRC2012 archives."
                    )

        else:
            raise ValueError(f"unsupported dataset kind: {kind}")
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# --- Preprocessing ---

class PreprocessingKind(str, Enum):
    """実行する前処理の種類。"""
    NONE = "none"
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_BLUR = "median_blur"
    # 今後、JPEG圧縮などを追加可能
    # JPEG_COMPRESSION = "jpeg_compression"

@dataclass
class GaussianBlurParams:
    """ガウシアンブラーのパラメータ。"""
    kernel_size: int
    sigma: float

@dataclass
class MedianBlurParams:
    """メディアンブラーのパラメータ。"""
    kernel_size: int

# 前処理パラメータの Union 型
PreprocessingParams = Union[GaussianBlurParams, MedianBlurParams, None]
