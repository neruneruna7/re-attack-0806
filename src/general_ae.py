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
import re_attack_0806
from copy import deepcopy

from re_attack_0806.models import MorimotoMnist, MorimotoCifar10, Ploof
from re_attack_0806 import attacks, utils
from re_attack_0806.attacks import bim, fgsm
from re_attack_0806 import attacks____
# lib 以下の attacks パッケージを使用（fgsm と bim を実装）

import foolbox

from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm
from re_attack_0806.utils.normTensor import *

@dataclass
class Config:
    dataset: DatasetKind = DatasetKind.MNIST
    model: ModelKind = ModelKind.MORIMOTO_MNIST
    attack: AttackKind = AttackKind.BIM
    model_dir: str = "./weight"
    epsilon: float = 0.3
    alpha: float = 0.05
    n: int = 10
    batch_size: int = 1
    device: Optional[torch.device] = None
    dataset_norm: DatasetNorm = DatasetNorm(DatasetKind.MNIST, torch.device("cpu"))

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device or utils.get_device()
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, train=False, batch_size=cfg.batch_size)
        self._load_weights_if_exists(cfg.model_dir)

    def _load_weights_if_exists(self, model_dir: str):
        path = os.path.join(model_dir, f"{self.model.model_name}.pth")
        if os.path.exists(path):
            st = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(st)
                print(f"loaded weights from {path}")
            except Exception:
                # より緩い読み込みを試みる（保存された state_dict にラッパー等が含まれる場合）
                print(f"failed to load exact state_dict from {path}, continuing with init model")

    @staticmethod
    def _to_logits(output: Tensor) -> Tensor:
    # （必要であれば）(N,C,H,W) を全体平均して (N,C) に変換する
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
            data = TensorWithState(data, NORMALIZED)
            # data = self.cfg.dataset_norm.normalize(data)
            target: Tensor = target.to(self.device).view(-1).long()
            data.tensor.requires_grad = True

            # 順伝播
            output = self.model(data.tensor)
            logits = self._to_logits(output)
            pred = logits.max(1, keepdim=True)[1]

            # 損失と勾配を計算
            loss = F.cross_entropy(logits, target)
            self.model.zero_grad()
            loss.backward()
            grad = data.tensor.grad
            if grad is None:
                # スキップして続行（デバッグ用にログを残しても良い）
                print(f"skip idx {i}: no grad")
                continue
            data_grad = grad.data
            data_grad = TensorWithState(data_grad, NORMALIZED)

            # 必要なら勾配を逆正規化（utils.denorm の期待値に合わせる）
            # data_denorm = utils.denorm(data_grad, self.device)
            # data_denorm = self.cfg.dataset_norm.denormalize(data_grad)
    
            # 攻撃実行

            if self.cfg.attack == AttackKind.BIM:
                # lib.attacks の内部実装である bim_attack を使用
                # perturbed = bim.bim_attack(data_denorm, self.cfg.epsilon, self.cfg.alpha, self.cfg.n,
                #                            target, self.model, self.device)
                # print(f"data_denorm shape: {data_grad.tensor.shape}")
                perturbed = bim.bim(
                    data_grad,
                    target,
                    self.model,
                    self.device,
                    self.cfg.epsilon,
                    self.cfg.alpha,
                    self.cfg.n,
                )
                # print(f"perturbed shape: {perturbed.tensor.shape}")
            # elif self.cfg.attack == AttackKind.FOOLBOX_BIM:
            #     # Foolbox による Linf Basic Iterative Attack を使用
            #     # チャンネル数に応じた前処理パラメータを準備
            #     if data.dim() == 4 and data.size(1) == 1:
            #         preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)
            #     else:
            #         preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261], axis=-3)
            #     bounds = (0.0, 1.0)
            #     try:
            #         # pylance の警告を抑制する（# type: ignore を付与）
            #         fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
            #         attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.n, abs_stepsize=self.cfg.alpha) # type: ignore[reportPrivateImportUsage]
            #         # ここまでpylanceの警告を抑制する．
            #         raw, clipped, is_adv = attack(fmodel, data_denorm, target, epsilons=self.cfg.epsilon)
            #         # clipped は foolbox のバージョンにより単体のテンソルかリストになることがある
            #         perturbed = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
            #     except Exception as e:
            #         print(f"Foolbox BIM attack failed: {e}")
            #     # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
            #     # preprocessing = dict(mean=[0.1307], std=[0.3081])
            #     # bounds = (-float("inf"), float("inf"))
            #     # fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing,  device=self.device)
            #     # attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.n, abs_stepsize=self.cfg.alpha)
            #     # raw, clipped, is_adv = attack(fmodel, data_denorm, target,  epsilons=0.03)
            #     # # print("row data:", raw)
            #     # # print("clipped:", clipped)
            #     # # print(is_adv)
            #     # perturbed = clipped
            elif self.cfg.attack == AttackKind.FGSM:
                # FGSM は (image_denorm, epsilon, target, model, device) を想定
                perturbed = fgsm.fgsm(data_grad, self.cfg.epsilon, target, self.model, self.device)
                # モデル推論のために perturbed を正規化
                # perturbed = fgsm.fgsm_attack(
                #     data_denorm, self.cfg.epsilon, data_grad, self.model, self.device
                # )
            else:
                raise ValueError(f"unsupported attack kind: {self.cfg.attack}")
    
            # # perturbed_norm = self.cfg.dataset_norm.normalize(perturbed)
            # print(f"original data average: {data.tensor.mean().item()}")
            # # print(f"perturbed_norm average: {perturbed_norm.mean().item()}")
            # print(f"perturbed average: {perturbed.tensor.mean().item()}")

            average_perturbation = utils.l2_norm_perturbation(
                self.cfg.dataset_norm.denormalize(data), 
                self.cfg.dataset_norm.denormalize(perturbed))

            out2 = self.model(perturbed.tensor)
            logits2 = self._to_logits(out2)
            pred2 = logits2.max(1, keepdim=True)[1]

            # mean_perturb = attacks____.mean_perturbation(data, perturbed)
            # print(f"average_perturbation: {average_perturbation}")


            # print(f"mean_perturb {mean_perturb}")
            adv_examples.append((i, pred.item(), pred2.item(), perturbed.tensor.squeeze().detach().cpu(), average_perturbation.item()))
        return adv_examples
    
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="general AE attack runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], default=DatasetKind.MNIST.value)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], default=ModelKind.MORIMOTO_MNIST.value)
    parser.add_argument("--attack", choices=[a.value for a in AttackKind], default=AttackKind.BIM.value)
    parser.add_argument("--model-dir", default="./weight")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=10, help="number of iterations for BIM")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    cfg = Config(
        dataset=DatasetKind(args.dataset),
        model=ModelKind(args.model),
        attack=AttackKind(args.attack),
        model_dir=args.model_dir,
        epsilon=args.epsilon,
        alpha=args.alpha,
        n=args.n,
        batch_size=args.batch_size,
        device=utils.get_device(),
        dataset_norm=DatasetNorm(DatasetKind(args.dataset), utils.get_device())
    )
    return cfg

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result = runner.run()

    # 統計を計算: 最初に正しく分類されていたサンプルのみを考慮
    fail = 0
    total = 0
    mean_mean_perturb = 0.0
    for idx, before, after, ex, m in result:
    # 元の正解ラベルが必要; 必要ならデータセットから再読み込みしても良い
    # ここでは以前のモデルの初期予測を用いて比較することを想定
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