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
from copy import deepcopy

from lib.models import MorimotoMnist, MorimotoCifar10, Ploof
from lib import attacks, utils
from lib.attacks import bim, fgsm
from lib import attacks____
# use attacks package under lib (implements fgsm and bim)

import foolbox

class DatasetKind(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    IMAGE_NET = "imagenet"
    INCEPTION_V3 = "inception_v3"


class ModelKind(str, Enum):
    MORIMOTO_MNIST = "mnist"
    MORIMOTO_CIFAR10 = "cifar10"
    INCEPTION_V3 = "inception_v3"
    PLOOF = "ploof"


class AttackKind(str, Enum):
    BIM = "bim"
    FGSM = "fgsm"
    FOOLBOX_BIM = "foolbox_bim"

class PresetKind(str, Enum):
    MORIMOTO_MNIST_BIM = "morimoto_mnist_bim"
    SAMPLE_PLOOF = "sample_ploof"
    BIM_ADVO = "bim_advo"
    DEFAULT = "default"

# Preset mapping: each preset sets fields on Config (uses enum values where appropriate)
PRESETS = {
    PresetKind.DEFAULT: {
        # no-op
    },
    PresetKind.MORIMOTO_MNIST_BIM: {
        "dataset": DatasetKind.MNIST,
        "model": ModelKind.MORIMOTO_MNIST,
        "attack": AttackKind.BIM,
        "epsilon": 0.3,
        "alpha": 0.05,
        "n": 10,
        "batch_size": 1,
    },
    PresetKind.SAMPLE_PLOOF: {
        "dataset": DatasetKind.MNIST,
        "model": ModelKind.PLOOF,
        "attack": AttackKind.FGSM,
    },
    PresetKind.BIM_ADVO : {
        "dataset": DatasetKind.IMAGE_NET,
        "model": ModelKind.INCEPTION_V3,
        "attack": AttackKind.BIM,
        "epsilon": 0.3,
        "alpha": 0.05,
        "n": 10,
        "batch_size": 1,
    }
}


@dataclass
class Config:
    dataset: DatasetKind = DatasetKind.MNIST
    model: ModelKind = ModelKind.MORIMOTO_MNIST
    attack: AttackKind = AttackKind.BIM
    model_dir: str = "./weight"
    epsilon: float = 0.3
    alpha: float = 0.05
    iters: int = 10
    n: int = 10
    batch_size: int = 1
    device: Optional[torch.device] = None

def apply_preset(cfg: Config, paper: PresetKind) -> Config:
    """Return new Config with preset fields applied (shallow copy)."""
    preset = PRESETS.get(paper, {})
    new_cfg = deepcopy(cfg)
    for k, v in preset.items():
        if hasattr(new_cfg, k):
            setattr(new_cfg, k, v)
    return new_cfg

class ModelFactory:
    @staticmethod
    def create(kind: ModelKind, device: torch.device) -> nn.Module:
        if kind == ModelKind.MORIMOTO_MNIST:
            return MorimotoMnist.MnistNet().to(device)
        if kind == ModelKind.MORIMOTO_CIFAR10:
            return MorimotoCifar10.Cifar10Net().to(device)
        if kind == ModelKind.PLOOF:
            # MNIST前提になってしまってる．このファクトリーの仕組みも問題がある．
            # 改善が必要
            return Ploof.PloofNet(10).to(device)
        if kind == ModelKind.INCEPTION_V3:
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=False, aux_logits=False)
            return model.to(device)
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
        elif kind == DatasetKind.IMAGE_NET:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])
            ds = datasets.ImageNet('../data/image_net', split='val', transform=transform)
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
        print(f"Running attack {self.cfg.attack} on model {self.cfg.model} with cfg: {self.cfg}")
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
            import foolbox

            if self.cfg.attack == AttackKind.BIM:
                # Use bim_attack from lib.attacks package (internal implementation)
                perturbed = bim.bim_attack(data_denorm, self.cfg.epsilon, self.cfg.alpha, self.cfg.n,
                                           target, self.model, self.device)
            elif self.cfg.attack == AttackKind.FOOLBOX_BIM:
                # Use Foolbox implementation of Linf Basic Iterative Attack
                # Prepare preprocessing depending on channel count
                if data.dim() == 4 and data.size(1) == 1:
                    preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)
                else:
                    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261], axis=-3)
                bounds = (0.0, 1.0)
                try:
                    # pylanceの警告を抑制する． # type: ignore[reportPrivateImportUsage] を使う．
                    fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
                    attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.n, abs_stepsize=self.cfg.alpha) # type: ignore[reportPrivateImportUsage]
                    # ここまでpylanceの警告を抑制する．
                    raw, clipped, is_adv = attack(fmodel, data_denorm, target, epsilons=self.cfg.epsilon)
                    # clipped may be a single tensor or a list/array depending on foolbox version
                    perturbed = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
                except Exception as e:
                    print(f"Foolbox BIM attack failed: {e}")
                # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
                # preprocessing = dict(mean=[0.1307], std=[0.3081])
                # bounds = (-float("inf"), float("inf"))
                # fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing,  device=self.device)
                # attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.n, abs_stepsize=self.cfg.alpha)
                # raw, clipped, is_adv = attack(fmodel, data_denorm, target,  epsilons=0.03)
                # # print("row data:", raw)
                # # print("clipped:", clipped)
                # # print(is_adv)
                # perturbed = clipped
            elif self.cfg.attack == AttackKind.FGSM:
                # FGSM expects (image_denorm, epsilon, target, model, device)
                # perturbed = attacks.fgsm_attack(data_denorm, self.cfg.epsilon, target)
                 # normalize perturbed for model inference
                perturbed = fgsm.fgsm_attack(
                    data_denorm, self.cfg.epsilon, data_grad, self.model, self.device
                )
            else:
                raise ValueError(f"unsupported attack kind: {self.cfg.attack}")
    
            if data.dim() == 4 and data.size(1) == 1:
                perturbed_norm = transforms.Normalize((0.1307,), (0.3081,))(perturbed)
            else:
                perturbed_norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                      (0.247, 0.243, 0.261))(perturbed)

            out2 = self.model(perturbed_norm)
            logits2 = self._to_logits(out2)
            pred2 = logits2.max(1, keepdim=True)[1]

            mean_perturb = attacks____.mean_perturbation(data, perturbed)
            adv_examples.append((i, pred.item(), pred2.item(), perturbed.squeeze().detach().cpu(), mean_perturb))
        return adv_examples
    
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="general AE attack runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], default=DatasetKind.MNIST.value)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], default=ModelKind.MORIMOTO_MNIST.value)
    parser.add_argument("--attack", choices=[a.value for a in AttackKind], default=AttackKind.BIM.value)
    parser.add_argument("--preset", choices=[p.value for p in PresetKind], default=None,
                        help="apply preset parameters for a reference paper")
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

    if args.preset:
        cfg = apply_preset(cfg, PresetKind(args.preset))
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

if __name__ == "__main__":
    main()