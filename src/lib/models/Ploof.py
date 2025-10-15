# 論文，再攻撃を用いた敵対的サンプルの矯正の試み 追試と，fgsm再攻撃実装の正当性を確認するための実装
from typing import Tuple
import torch
from torch import nn
import numpy as np
from torch import Tensor

class PloofNet(nn.Module):
    def __init__(self, output_features: int):
        super(PloofNet, self).__init__()
        self.model_name = "ploof"
        self.output_features = output_features

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.output_features)  # 出力を10クラスに合わせる

    def forward(self, x: Tensor) -> Tensor:
        print("input", x.shape)
        x = torch.sin(x)
        x = x + x
        x = x * x
        x = torch.tanh(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # 最終出力をクラス数と合わせる必要がある？

        return x

