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

        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act2 = nn.ELU()

        self.pool = nn.MaxPool2d(2)

        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.act3 = nn.ELU()

        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act4 = nn.ELU()

        self.maxpool2 = nn.MaxPool2d(2)

        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(128, 256, 3, 1)
        self.act5 = nn.ELU()

        self.conv6 = nn.Conv2d(256, 256, 3, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act6 = nn.ELU()

        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(256, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.act7 = nn.ELU()

        self.dropout4 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(1024, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        # print("conv1", x.shape)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act2(x)
        # print("bn1", x.shape)

        x = self.pool(x)
        # print("pool", x.shape)

        x = self.dropout1(x)
        # print("dropout1", x.shape)

        x = self.conv3(x)
        x = self.act3(x)
        # print("conv2", x.shape)

        x = self.conv4(x)
        x = self.bn2(x)
        x = self.act4(x)
        # print("bn2", x.shape)

        x = self.maxpool2(x)
        x = self.dropout2(x)
        # print("dropout2", x.shape)

        x = self.conv5(x)
        x = self.act5(x)
        # print("conv3", x.shape)

        x = self.conv6(x)
        x = self.bn3(x)
        x = self.act6(x)
        # print("bn3", x.shape)

        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        # print("flatten", x.shape)

        x = self.fc1(x)
        x = self.act7(x)
        # print("fc1", x.shape)

        x = self.dropout4(x)
        x = self.fc2(x)
        # print("fc2", x.shape)
        # print("/n/n/n/n")

        # panic

        return x

