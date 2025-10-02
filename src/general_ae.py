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

from lib.models import MorimotoMnist, MorimotoCifar10
from lib import attacks, utils


def main():
    epsilon = 0.3
    alpha = 0.05
    n = 10
    model_save_dir = "./weight"

    image_save_dir = "data/attacked_images"

    device = utils.get_device()

    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)
    
    model = MorimotoMnist.MnistNet().to(device)

    # Load the pretrained model
    model_path = os.path.join(model_save_dir, f'{model.model_name}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    print("RUN")
    adv_examples = []
    count = 0
    for i, (data, target) in enumerate(test_loader):
        # if i >= 10:  # 最初の100個のテストデータに対して攻撃を行う
        #     break
        # if count >= 1000:  # 正しく分類されたものを10個見つけたら終了
        #     break
        data: Tensor  = data.to(device) 
        target: Tensor = target.to(device)
        data.requires_grad = True
        
        clean_output: Tensor = model(data)
        clean_pred = clean_output.max(1, keepdim=True)[1] # get the index of the max log-probability


        loss = F.cross_entropy(clean_output, target)
        model.zero_grad()
        loss.backward()

        grad = data.grad
        if grad is None:
            raise ValueError("grad is None")
        data_grad = grad.data

        data_denorm = utils.denorm(data_grad, device)

        # perturbed_data = attacks.fgsm_attack(data_denorm, epsilon, data_grad)
        perturbed_data = attacks.bim_attack(data_denorm, epsilon, alpha, n, data_grad, model, device)

        # 最初の AE による予測（これが final_pred）
        perturbed_data_normalized: Tensor = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        perturbed_output: Tensor = model(perturbed_data_normalized)

        perturbed_pred = perturbed_output.max(1, keepdim=True)[1]  # shape [B,1]

        # 平均摂動量を計算して表示
        mean_perturb = attacks.mean_perturbation(data, perturbed_data)

        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append( (clean_pred.item(), perturbed_pred.item(), target.item(), adv_ex, mean_perturb) )

        if clean_pred.item() == target.item():
            count += 1


    # final_acc = correct/float(len(test_loader))
    # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # 攻撃成功率を計算する
    # fgsm前のデータで正しく分類されていたもののうち、fgsm後に誤分類されたものの割合
    fail = 0
    total = 0
    mean_mean_perturb = 0
    for clean, adv, target, ex, mean_perturb in adv_examples:
        # print(f"clean: {clean}, adv: {adv}, target: {target},\n mean_perturb: {mean_perturb}")

        if clean != target:
            continue
        total += 1
        mean_mean_perturb += mean_perturb
        if adv != target:
            fail += 1
    
    mean_perturb = mean_mean_perturb / total if total > 0 else 0.0

    if total == 0:
        attack_acc = 0.0
    else:
        attack_acc = fail / total
    print(f"Epsilon: {epsilon}\tAttack Success Rate = {attack_acc} = {fail} / {total}")
    print(f"Mean Perturbation = {mean_perturb}")

    return attack_acc, adv_examples

if __name__ == "__main__":
    main()