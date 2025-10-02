# 論文，再攻撃を用いた敵対的サンプルの矯正の試み 追試と，fgsm再攻撃実装の正当性を確認するための実装
from typing import Tuple
import torch
from torch import nn
import numpy as np
from torch import Tensor

class PloofNet(nn.Module):
    def __init__(self):
        super(PloofNet, self).__init__()
        self.model_name = "ploof"

    def forward(self, x: Tensor) -> Tensor:
        x = x % 2

        return x

