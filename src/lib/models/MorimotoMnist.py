# 論文，再攻撃を用いた敵対的サンプルの矯正の試み 追試と，fgsm再攻撃実装の正当性を確認するための実装
from typing import Tuple
import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.model_name = "morimoto_mnist"

        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.act2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 1, 1)
        self.act3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.act4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        # print("conv1", x.shape)

        x = self.conv2(x)
        x = self.act2(x)
        # print("conv2", x.shape)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.act3(x)
        # print("conv3", x.shape)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        # print("flatten", x.shape)
        x = self.fc1(x)
        x = self.act4(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        # output = nn.functional.log_softmax(x, dim=1)
        return x

