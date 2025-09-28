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

# restores the tensors to their original scale
def denorm(batch, device, mean=[0.1307], std=[0.3081]) -> Tensor:
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    # print("batch_shape", batch.shape) 
    # #バッチは４次元

    # print("mean_shape", mean.shape)
    # # 1次元
    # print(mean.view(1, -1, 1, 1).shape)
    # # 4次元

    # print("std_shape", std.shape)
    # # 1次元
    # print(std.view(1, -1, 1, 1).shape)
    # # 4次元
    # print("")

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def save_tensor_as_image(tensor: Tensor, path: str):
    """
    tensor: [B,C,H,W] or [C,H,W] or [H,W], values in 0..1 (非正規化)
    path: output png path
    """
    t = tensor.clone().detach().cpu()
    # squeeze batch/channel dims if needed
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3 and t.size(0) == 1:
        arr = t.squeeze(0).numpy()
    elif t.dim() == 3 and t.size(0) == 3:
        # convert CHW -> HWC
        arr = t.permute(1, 2, 0).numpy()
    elif t.dim() == 2:
        arr = t.numpy()
    else:
        arr = t.numpy()

    # clip and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    # create parent dir
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = Image.fromarray(arr_u8)
    img.save(path)
    info = f"saved image {path}"
    print(info)

def main():
    epsilon = 0.9
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
    for i, (data, target) in enumerate(test_loader):
        # if i >= 100:  # 最初の100個のテストデータに対して攻撃を行う
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

        data_denorm = denorm(data_grad, device)

        perturbed_data = attacks.fgsm_attack(data_denorm, epsilon, data_grad)

        # 最初の FGSM による予測（これが final_pred）
        perturbed_data_normalized: Tensor = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        perturbed_output: Tensor = model(perturbed_data_normalized)

        perturbed_pred = perturbed_output.max(1, keepdim=True)[1]  # shape [B,1]

        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append( (clean_pred.item(), perturbed_pred.item(), target.item(), adv_ex) )

    # final_acc = correct/float(len(test_loader))
    # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # 攻撃成功率を計算する
    # fgsm前のデータで正しく分類されていたもののうち、fgsm後に誤分類されたものの割合
    fail = 0
    total = 0
    for clean, adv, target, ex in adv_examples:
        if clean != target:
            continue
        total += 1
        if adv != target:
            fail += 1

    if total == 0:
        attack_acc = 0.0
    else:
        attack_acc = fail / total
    print(f"Epsilon: {epsilon}\tAttack Success Rate = {attack_acc} = {fail} / {total}")

    return attack_acc, adv_examples

if __name__ == "__main__":
    main()