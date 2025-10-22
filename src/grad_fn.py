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

import foolbox


from lib.models import MorimotoMnist, MorimotoCifar10, Ploof
from lib import attacks, utils


def main():
    n = 10
    model_save_dir = "./weight"

    image_save_dir = "data/attacked_images"

    device = utils.get_device()

    test_loader_MNIST = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=False)
    
    test_loader = test_loader_MNIST

    model = MorimotoMnist.MnistNet().to(device)
    # model = MorimotoCifar10.Cifar10Net().to(device)
    # model = Ploof.PloofNet(10).to(device)

    fmodel = foolbox.PyTorchModel(model)

    # # Load the pretrained model
    # model_path = os.path.join(model_save_dir, f'{model.model_name}.pth')
    # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    print("RUN")

    grads = []
    for i, (data, target) in enumerate(test_loader):
        if i >= 1:  # つかうのは最初の1つだけ
            break

        for i in range(n):
            data_i: Tensor  = data.to(device) 
            target_i: Tensor = target.to(device)
            data_i.requires_grad = True
            
            clean_output: Tensor = model(data_i)


            loss = F.cross_entropy(clean_output, target_i)
            model.zero_grad()
            loss.backward()

            grad = data_i.grad
            if grad is None:
                raise ValueError("grad is None")
            data_grad = grad.data

            grads.append(data_grad.detach().cpu().numpy())
    
    # gradsをnumpy arrayに変換
    grads = np.array(grads)  # shape (n, 1, 28, 28)

    # gradsの数値をprint
    for i, g in enumerate(grads):
        print(f"grad {i}: min {g.min()}, max {g.max()}, mean {g.mean()}, std {g.std()}")
        print(g)
        print("-----")
    

    print("END")

if __name__ == "__main__":
    main()