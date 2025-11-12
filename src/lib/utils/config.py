from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
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
from enum import Enum
import argparse
import lib
from copy import deepcopy

from lib.models import MorimotoCifar10, MorimotoMnist, Ploof


class DatasetKind(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    IMAGE_NET = "imagenet"
    INCEPTION_V3 = "inception_v3"


class ModelKind(str, Enum):
    MORIMOTO_MNIST = "mnist"
    MORIMOTO_CIFAR10 = "cifar10"
    INCEPTION_V3 = "inception_v3"
    PLOOF = "ploof"


class AttackKind(str, Enum):
    BIM = "bim"
    FGSM = "fgsm"
    FOOLBOX_BIM = "foolbox_bim"




class DatasetNorm:
    """データセット種別に基づく正規化/非正規化を行うユーティリティクラス。

    引数 dataset_kind は `DatasetKind`（またはその value 文字列）を受け取る。
    内部で mean/std を決定し、テンソルの正規化/非正規化を行う。
    """
    def __init__(self, dataset_kind: object, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        # dataset_kind が Enum の場合は value を使う（循環 import を避けるため、柔軟に受け入れる）
        kind = getattr(dataset_kind, 'value', dataset_kind)
        # Enum や文字列どちらでも受け取るため、文字列化して小文字化する
        kind = str(kind).lower()
        self.device = device
        self.dtype = dtype

        if kind in ('mnist', 'mnist'.upper()):
            mean = [0.1307]
            std = [0.3081]
        elif kind in ('cifar10', 'cifar10'.upper(), 'cifar'):
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
        elif kind in ('imagenet', 'image_net', 'image-net'):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            # デフォルトは CIFAR 風（3 チャンネル）
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]

        self.mean = torch.tensor(mean, device=self.device, dtype=self.dtype).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=self.device, dtype=self.dtype).view(1, -1, 1, 1)

    def normalize(self, x: Tensor) -> Tensor:
        """正規化を行う。入力は [B,C,H,W] を想定（必要なら .to(self.device) を呼ぶ）。"""
        x = x.to(self.device)
        return (x - self.mean) / self.std

    def denormalize(self, x: Tensor) -> Tensor:
        """非正規化（元のピクセル空間へ戻す）。"""
        x = x.to(self.device)
        return x * self.std + self.mean



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
            model = inception_v3(pretrained=False, aux_logits=False)
            return model.to(device)
        raise ValueError(f"unsupported model kind: {kind}")


class DataFactory:
    @staticmethod
    def loader(kind: DatasetKind, batch_size: int):
        if kind == DatasetKind.MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            ds = datasets.MNIST('../data', train=False, download=True, transform=transform)
        elif kind == DatasetKind.CIFAR10:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
            ])
            ds = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
        elif kind == DatasetKind.IMAGE_NET:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])
            # torchvision.datasets.ImageNet は devkit（ILSVRC2012_devkit_t12.tar.gz）や
            # 画像アーカイブが所定の場所にないと RuntimeError を投げることがある。
            # その場合はローカルに展開済みの val ディレクトリを ImageFolder で読み込むフォールバックを行う。
            try:
                ds = datasets.ImageNet('../data/image_net', split='val', transform=transform)
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
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
