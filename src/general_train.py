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
import re_attack_0806

from re_attack_0806 import utils
from re_attack_0806.utils.config import DataFactory, DatasetKind
from re_attack_0806.models import MorimotoMnist, MorimotoCifar10

class TrainParam:
    def __init__(self, epochs: int = 10, batch_size: int = 128, lr: float = 1e-2, save_weight_dir: str = "./weight"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_dir = save_weight_dir


def train(data_loader, model: nn.Module, loss_fn, optimizer, device):
    model.train()
    size = len(data_loader.dataset)
    last_loss = 0.0
    num_batches = 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        last_loss = loss_value
        num_batches += 1

        if batch % 100 == 0:
            current = batch * len(X)
    # epoch の最終バッチの loss を返す（バッチがない場合は 0.0 を返す）
    return last_loss

def test(dataloader, model: nn.Module, loss_fn, device):
    """テストデータで評価し、(accuracy_percent, avg_loss) を返す。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            batch_size = X.size(0)
            total_loss += loss_fn(pred, y).item() * batch_size
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += batch_size

    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    accuracy_percent = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
    return accuracy_percent, avg_loss


def train_roop(model: nn.Module, loss_fn, optimizer, train_loader, test_loader, device, epochs: int):
    """学習ループ。各 epoch の最終 loss とテスト指標を収集して返す。

    Returns:
        data: list of tuples (train_final_loss, test_accuracy_percent, test_avg_loss)
    """
    data: list[tuple[float, float, float]] = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_final_loss = train(train_loader, model, loss_fn, optimizer, device)
        print(f"final loss: {epoch_final_loss:.6f}")
        accuracy, test_loss = test(test_loader, model, loss_fn, device)
        print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

        data.append((epoch_final_loss, accuracy, test_loss))

    print("Done!")
    return data


def plot_training_progress(data: list[tuple[float, float, float]], out_path: str = './training_plot.png') -> None:
    """与えられた学習結果データからプロットを作成して保存する。

    Args:
        data: list of tuples (train_final_loss, test_accuracy_percent, test_avg_loss)
        out_path: 出力 PNG ファイルパス
    """
    if not data:
        print("No training data to plot.")
        return

    epochs = len(data)
    epochs_x = list(range(1, epochs + 1))
    epoch_losses = [d[0] for d in data]
    test_accuracies = [d[1] for d in data]
    test_losses = [d[2] for d in data]

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
        ax1.plot(epochs_x, epoch_losses, marker='o', label='epoch_final_loss')
        ax1.plot(epochs_x, test_losses, marker='x', label='test_avg_loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()

        ax2.plot(epochs_x, test_accuracies, marker='s', color='tab:orange', label='test_accuracy(%)')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy (%)')
        ax2.legend()

        fig.suptitle('Training Progress')
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved training plot to {out_path}")
    except Exception as e:
        print(f"Failed to create training plot: {e}")

def main():
    save_dir = "./weight"

    mnist_train_param = TrainParam(epochs=99, batch_size=128, lr=1e-4, save_weight_dir=save_dir)
    general_train_param = mnist_train_param

    # cifar10_train_param = TrainParam(epochs=99, batch_size=128, lr=1e-2, save_weight_dir=save_dir)
    # general_train_param = cifar10_train_param

    # DataFactory を使ってデータローダを生成（config.py に定義された変換と正規化を利用）
    train_loader = DataFactory.loader(DatasetKind.MNIST, train=True, batch_size=mnist_train_param.batch_size)
    test_loader = DataFactory.loader(DatasetKind.MNIST, train=False, batch_size=mnist_train_param.batch_size)
    
    device = utils.get_device()
    
    print(f"Using {device} device")


    model = MorimotoMnist.MnistNet().to(device)
    # model = MorimotoCifar10.Cifar10Net().to(device)
    print(model)



    print(f"Learning rate: {general_train_param.lr}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=general_train_param.lr)

    # 学習ループを実行
    results = train_roop(model, loss_fn, optimizer, train_loader, test_loader, device, general_train_param.epochs)
    # 結果に基づいてプロットを出力
    plot_path = os.path.join("./assets", "training_plot.png")
    try:
        plot_training_progress(results, out_path=plot_path)
    except Exception as e:
        print(f"Failed to plot training progress: {e}")


    # モデルの保存
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model.model_name}.pth"))
    print(f"Saved initial model to {os.path.join(save_dir, f'{model.model_name}.pth')}")


if __name__ == "__main__":
    main()