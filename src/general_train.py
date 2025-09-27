# モデルをトレーニングする汎用コード
from typing import Any, Tuple
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
import lib

from lib import utils
from lib.models import MorimotoMnist, MorimotoCifar10



def train(data_loader, model: nn.Module, loss_fn, optimizer, device):
    model.train()
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model: nn.Module, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_roop(model: nn.Module, loss_fn, optimizer, train_loader, test_loader, device, epochs: int):

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        # test(test_loader, model, loss_fn, device)
    print("Done!")

def main():
    batch_size = 128
    epochs = 99
    save_dir = "./weight"

    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True)
    
    device = utils.get_device()
    
    print(f"Using {device} device")


    # model = MorimotoMnist.MnistNet().to(device)
    model = MorimotoCifar10.Cifar10Net().to(device)
    print(model)

    lr=1e-2
    print(f"Learning rate: {lr}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学習ループを実行
    train_roop(model, loss_fn, optimizer, train_loader, test_loader, device, epochs)

    test(test_loader, model, loss_fn, device)


    # モデルの保存
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model.model_name}.pth"))
    print(f"Saved initial model to {os.path.join(save_dir, f'{model.model_name}.pth')}")


if __name__ == "__main__":
    main()