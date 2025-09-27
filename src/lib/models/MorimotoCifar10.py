# 論文，再攻撃を用いた敵対的サンプルの矯正の試み 追試と，fgsm再攻撃実装の正当性を確認するための実装
from typing import Tuple
import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.model_name = "morimoto_cifar10"

        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.act1 = nn.ELU()

        self.bn1 = nn.BatchNorm2d(64)
        self.act2 = nn.ELU()

        self.pool = nn.MaxPool2d(2)

        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.act3 = nn.ELU()

        self.bn2 = nn.BatchNorm2d(128)
        self.act4 = nn.ELU()

        self.maxpool2 = nn.MaxPool2d(2)

        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.act5 = nn.ELU()

        self.bn3 = nn.BatchNorm2d(256)
        self.act6 = nn.ELU()

        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.act7 = nn.ELU()

        self.dropout4 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(1024, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.bn1(x)
        x = self.act2(x)
        x = self.pool(x)

        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.act3(x)
        x = self.bn2(x)
        x = self.act4(x)

        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.act5(x)
        x = self.bn3(x)
        x = self.act6(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act7(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

