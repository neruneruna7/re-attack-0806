from typing import Tuple
import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # print("forward")
        # print("1", x.shape)
        x = self.conv1(x)
        # print("conv1", x.shape)
        x = nn.functional.relu(x)
        # print("conv1_relu", x.shape)

        x = self.conv2(x)
        # print("conv2", x.shape)

        x = nn.functional.relu(x)
        # print("conv2_relu", x.shape)

        x = nn.functional.max_pool2d(x, 2)
        # print("max_pool", x.shape)

        x = self.dropout1(x)
        # print("dropout1", x.shape)

        x = torch.flatten(x, 1)
        # print("flatten", x.shape)

        x = self.fc1(x)
        # print("fc1", x.shape)

        x = nn.functional.relu(x)
        # print("fc1_relu", x.shape)

        x = self.dropout2(x)
        # print("dropout2", x.shape)

        x = self.fc2(x)
        # print("fc2", x.shape)
        # print("")


        output = nn.functional.log_softmax(x, dim=1)
        return output

