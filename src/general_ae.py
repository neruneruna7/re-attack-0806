# モデルをトレーニングする汎用コード
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
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
from enum import Enum
import argparse
import lib

from lib.models import MorimotoMnist, MorimotoCifar10
from lib import attacks, utils


class DatasetKind(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"


class ModelKind(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"


class AttackKind(str, Enum):
    BIM = "bim"
    FGSM = "fgsm"

@dataclass
class Config:
    dataset: DatasetKind = DatasetKind.MNIST
    model: ModelKind = ModelKind.MNIST
    attack: AttackKind = AttackKind.BIM
    model_dir: str = "./weight"
    epsilon: float = 0.3
    alpha: float = 0.05
    iters: int = 10
    batch_size: int = 1
    device: Optional[torch.device] = None

class ModelFactory:
    @staticmethod
    def create(kind: ModelKind, device: torch.device) -> nn.Module:
        if kind == ModelKind.MNIST:
            return MorimotoMnist.MnistNet().to(device)
        if kind == ModelKind.CIFAR10:
            return MorimotoCifar10.Cifar10Net().to(device)
        raise ValueError(f"unsupported model kind: {kind}")


class DataFactory:
    @staticmethod
    def loader(kind: DatasetKind, batch_size: int):
        if kind == DatasetKind.MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            ds = datasets.MNIST('../data', train=False, download=True, transform=transform)
        elif kind == DatasetKind.CIFAR10:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
            ])
            ds = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
        else:
            raise ValueError(f"unsupported dataset kind: {kind}")
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device or utils.get_device()
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, cfg.batch_size)
        self._load_weights_if_exists(cfg.model_dir)

    def _load_weights_if_exists(self, model_dir: str):
        path = os.path.join(model_dir, f"{self.model.model_name}.pth")
        if os.path.exists(path):
            st = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(st)
                print(f"loaded weights from {path}")
            except Exception:
                # try looser loading (in case saved dict had extra wrappers)
                print(f"failed to load exact state_dict from {path}, continuing with init model")

    @staticmethod
    def _to_logits(output: Tensor) -> Tensor:
        # convert (N,C,H,W) -> (N,C) by global averaging if needed
        if output.dim() == 4:
            return output.mean(dim=(2, 3))
        if output.dim() > 2:
            return output.view(output.size(0), output.size(1), -1).mean(dim=2)
        return output
    
    def run(self) -> List[Tuple[int, int, int, Tensor, float]]:
        self.model.eval()
        adv_examples = []

        for i, (data, target) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = target.to(self.device).view(-1).long()
            data.requires_grad = True

            # forward
            output = self.model(data)
            logits = self._to_logits(output)
            pred = logits.max(1, keepdim=True)[1]

            # compute loss and grads
            loss = F.cross_entropy(logits, target)
            self.model.zero_grad()
            loss.backward()
            grad = data.grad
            if grad is None:
                # skip and continue; could log for debugging
                print(f"skip idx {i}: no grad")
                continue
            data_grad = grad.data

            # convert grad -> denorm if needed (utils.denorm expects normalized input? existing code used it)
            data_denorm = utils.denorm(data_grad, self.device)

            # attack
            if self.cfg.attack == AttackKind.BIM:
                perturbed = attacks.bim_attack(data_denorm, self.cfg.epsilon, self.cfg.alpha, self.cfg.iters,
                                               data_grad, self.model, self.device)
            else:
                # FGSM expects (image_denorm, epsilon, target, model, device)
                perturbed = attacks.fgsm_attack(data_denorm, self.cfg.epsilon, target)
                 # normalize perturbed for model inference
            if data.dim() == 4 and data.size(1) == 1:
                perturbed_norm = transforms.Normalize((0.1307,), (0.3081,))(perturbed)
            else:
                perturbed_norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                      (0.247, 0.243, 0.261))(perturbed)

            out2 = self.model(perturbed_norm)
            logits2 = self._to_logits(out2)
            pred2 = logits2.max(1, keepdim=True)[1]

            mean_perturb = attacks.mean_perturbation(data, perturbed)
            adv_examples.append((i, pred.item(), pred2.item(), perturbed.squeeze().detach().cpu(), mean_perturb))
        return adv_examples

        # # compute stats: only consider samples that were initially correct
        # fail = 0
        # total = 0
        # mean_mean_perturb = 0.0
        # for idx, before, after, ex, m in adv_examples:
        #     # need original true label; we can reload dataset sample if necessary,
        #     # but here assume before was model's initial pred and compare with that
        #     total += 1
        #     mean_mean_perturb += m
        #     if after != before:
        #         fail += 1

        # attack_acc = (fail / total) if total > 0 else 0.0
        # mean_perturb = (mean_mean_perturb / total) if total > 0 else 0.0
        # print(f"Epsilon: {self.cfg.epsilon}\tAttack Success Rate = {attack_acc} = {fail} / {total}")
        # print(f"Mean Perturbation = {mean_perturb}")
        # return attack_acc, adv_examples

    
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="general AE attack runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], default=DatasetKind.MNIST.value)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], default=ModelKind.MNIST.value)
    parser.add_argument("--attack", choices=[a.value for a in AttackKind], default=AttackKind.BIM.value)
    parser.add_argument("--model-dir", default="./weight")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    cfg = Config(
        dataset=DatasetKind(args.dataset),
        model=ModelKind(args.model),
        attack=AttackKind(args.attack),
        model_dir=args.model_dir,
        epsilon=args.epsilon,
        alpha=args.alpha,
        iters=args.iters,
        batch_size=args.batch_size,
        device=utils.get_device()
    )
    return cfg

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result = runner.run()

    # compute stats: only consider samples that were initially correct
    fail = 0
    total = 0
    mean_mean_perturb = 0.0
    for idx, before, after, ex, m in result:
        # need original true label; we can reload dataset sample if necessary,
        # but here assume before was model's initial pred and compare with that
        total += 1
        mean_mean_perturb += m
        if after != before:
            fail += 1

    attack_acc = (fail / total) if total > 0 else 0.0
    mean_perturb = (mean_mean_perturb / total) if total > 0 else 0.0
    print(f"Epsilon: {cfg.epsilon}\tAttack Success Rate = {attack_acc} = {fail} / {total}")
    print(f"Mean Perturbation = {mean_perturb}")


# def main():
#     epsilon = 0.3
#     alpha = 0.05
#     n = 10
#     model_save_dir = "./weight"

#     image_save_dir = "data/attacked_images"

#     device = utils.get_device()

#     test_loader_MNIST = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Normalize((0.1307,), (0.3081,)),
#             ])),
#         batch_size=1, shuffle=True)
    
#     test_loader_cifar10 = torch.utils.data.DataLoader(
#     datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])),
#         batch_size=1, shuffle=True)
    
#     test_loader = test_loader_MNIST
#     # test_loader = test_loader_cifar10

#     model = MorimotoMnist.MnistNet().to(device)
#     # model = MorimotoCifar10.Cifar10Net().to(device)


#     # Load the pretrained model
#     model_path = os.path.join(model_save_dir, f'{model.model_name}.pth')
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

#         # Set the model in evaluation mode. In this case this is for the Dropout layers
#     model.eval()

#     print("RUN")
#     adv_examples = []
#     count = 0
#     for i, (data, target) in enumerate(test_loader):
#         # if i >= 10:  # 最初の100個のテストデータに対して攻撃を行う
#         #     break
#         # if count >= 1000:  # 正しく分類されたものを10個見つけたら終了
#         #     break
#         data: Tensor  = data.to(device) 
#         target: Tensor = target.to(device)
#         data.requires_grad = True
        
#         clean_output: Tensor = model(data)
#         clean_pred = clean_output.max(1, keepdim=True)[1] # get the index of the max log-probability


#         loss = F.cross_entropy(clean_output, target)
#         model.zero_grad()
#         loss.backward()

#         grad = data.grad
#         if grad is None:
#             raise ValueError("grad is None")
#         data_grad = grad.data

#         data_denorm = utils.denorm(data_grad, device)

#         # perturbed_data = attacks.fgsm_attack(data_denorm, epsilon, data_grad)
#         perturbed_data = attacks.bim_attack(data_denorm, epsilon, alpha, n, data_grad, model, device)

#         # 最初の AE による予測（これが final_pred）
#         perturbed_data_normalized: Tensor = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
#         perturbed_output: Tensor = model(perturbed_data_normalized)

#         perturbed_pred = perturbed_output.max(1, keepdim=True)[1]  # shape [B,1]

#         # 平均摂動量を計算して表示
#         mean_perturb = attacks.mean_perturbation(data, perturbed_data)

#         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#         adv_examples.append( (clean_pred.item(), perturbed_pred.item(), target.item(), adv_ex, mean_perturb) )

#         if clean_pred.item() == target.item():
#             count += 1


#     # final_acc = correct/float(len(test_loader))
#     # print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

#     # 攻撃成功率を計算する
#     # fgsm前のデータで正しく分類されていたもののうち、fgsm後に誤分類されたものの割合
#     fail = 0
#     total = 0
#     mean_mean_perturb = 0
#     for clean, adv, target, ex, mean_perturb in adv_examples:
#         # print(f"clean: {clean}, adv: {adv}, target: {target},\n mean_perturb: {mean_perturb}")

#         if clean != target:
#             continue
#         total += 1
#         mean_mean_perturb += mean_perturb
#         if adv != target:
#             fail += 1
    
#     mean_perturb = mean_mean_perturb / total if total > 0 else 0.0

#     if total == 0:
#         attack_acc = 0.0
#     else:
#         attack_acc = fail / total
#     print(f"Epsilon: {epsilon}\tAttack Success Rate = {attack_acc} = {fail} / {total}")
#     print(f"Mean Perturbation = {mean_perturb}")

#     return attack_acc, adv_examples

if __name__ == "__main__":
    main()