from typing import Tuple
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

from lib import attacks, utils, lenet

# from lib.models.lenet import Net

# Set random seed for reproducibility
torch.manual_seed(42)
out_dir = "data/attacked_images"


def main():
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [0.9]
    pretrained_model = "data/lenet_mnist_model.pth"

    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
    device = utils.get_device()
    print(f"Using {device} device")

    # Initialize the network
    model = lenet.Net().to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location=device, weights_only=True))

    accuracies = []
    examples = []

    label_taple = []

    # # テスト用のサンプルを1つ取得
    # images, labels = next(iter(test_loader))
    # images, labels = images.to(device), labels.to(device)

    print("RUN!")
    for i, (image, label) in enumerate(test_loader):
        image, label = image.to(device), label.to(device)
        # print(f"i: {i}")
        # Run test for each epsilon
        for eps in epsilons:
            # acc, ex = test(model, device, test_loader, eps)
            x_adv_prime = attacks.bim_reattack(model, image, label, device, epsilon=eps, alpha=0.05, num_iter=10)
            # 元画像と再攻撃後の予測を比較
            with torch.no_grad():
                pred_orig = model(image).argmax(dim=1)
                pred_adv = model(x_adv_prime).argmax(dim=1)
                # print(f"元のラベル: {label.item()}")
                # print(f"元画像の予測: {pred_orig.item()}")
                # print(f"再攻撃後の予測: {pred_adv.item()}")
                label_taple.append((label.item(), pred_orig.item(), pred_adv.item()))
                # accuracies.append(acc)
                # examples.append(ex)
    
    # 正解率の計算
    # 元のラベルに対して，再攻撃後の予測が一致している割合
    # ただし，元画像の予測の時点で正しいものに限る
    # 元画像の予測の時点で間違っているものは，別途カウントする
    correct_count = 0
    total_count = 0
    wrong_pred_count = 0
    for (true_label, pred_orig, pred_adv) in label_taple:
        if true_label == pred_orig:
            total_count += 1
            if true_label == pred_adv:
                correct_count += 1
        else:
            wrong_pred_count += 1

    final_acc = correct_count / total_count if total_count > 0 else 0
    print(f"再攻撃後の正解率: {final_acc} ({correct_count} / {total_count})")
    print(f"元画像の予測が間違っていた数: {wrong_pred_count}")
    


if __name__ == "__main__":
    main()