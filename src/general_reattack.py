# 再攻撃を実行し、その結果を評価・保存する汎用コード
from dataclasses import dataclass
from torch.types import Number
from typing import Any, Iterator, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from PIL import Image
from torch import Tensor
from enum import Enum
import argparse
import re_attack_0806
from copy import deepcopy

from re_attack_0806 import attacks, utils
from re_attack_0806.attacks import bim, fgsm
from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm
from re_attack_0806.utils.normTensor import *

from tqdm import tqdm

@dataclass
class Config:
    dataset: DatasetKind = DatasetKind.MNIST
    model: ModelKind = ModelKind.MORIMOTO_MNIST
    model_dir: str = "./weight"
    batch_size: int = 16
    device: Optional[torch.device] = None
    dataset_norm: DatasetNorm = DatasetNorm(DatasetKind.MNIST, torch.device("cpu"))
    output_dir: str = "reattacked_data"
    num_samples: int = -1

    # Initial Attack Config
    attack_kind: AttackKind = AttackKind.BIM
    attack_eps: float = 0.3
    attack_alpha: float = 0.05
    attack_n: int = 10

    # Re-Attack Config
    reattack_kind: AttackKind = AttackKind.BIM
    reattack_eps: float = 0.3
    reattack_alpha: float = 0.05
    reattack_n: int = 10


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device or utils.get_device()
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, train=False, batch_size=cfg.batch_size)
        self._load_weights_if_exists(cfg.model_dir)

    def _load_weights_if_exists(self, model_dir: str):
        model_name = getattr(self.model, "model_name", None)
        if model_name is None:
            print(f"warning: model has no 'model_name' attribute (model class: {self.model.__class__.__name__}), skipping weight load")
            return
        path = os.path.join(model_dir, f"{self.model.model_name}.pth")
        if os.path.exists(path):
            st = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(st)
                print(f"loaded weights from {path}")
            except Exception:
                print(f"failed to load exact state_dict from {path}, continuing with init model")

    @staticmethod
    def _to_logits(output: Tensor) -> Tensor:
        if output.dim() == 4:
            return output.mean(dim=(2, 3))
        if output.dim() > 2:
            return output.view(output.size(0), output.size(1), -1).mean(dim=2)
        return output

    def _perform_attack(self, data: TensorWithState, target: Tensor, kind: AttackKind, **kwargs: Any) -> TensorWithState:
        eps = kwargs.get('eps', self.cfg.attack_eps) # Fallback to attack_eps if not provided
        alpha = kwargs.get('alpha', self.cfg.attack_alpha) # Fallback to attack_alpha if not provided
        n = kwargs.get('n', self.cfg.attack_n) # Fallback to attack_n if not provided

        if kind == AttackKind.BIM:
            return bim.bim(data, target, self.model, self.device, eps, alpha, n, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std)
        elif kind == AttackKind.FGSM:
            return fgsm.fgsm(data, eps, target, self.model, self.device)
        elif kind == AttackKind.FOOLBOX_BIM:
            # Dynamically set preprocessing based on dataset
            if self.cfg.dataset == DatasetKind.MNIST:
                preprocessing = dict(
                    mean=self.cfg.dataset_norm.mean.item(), # MNIST is grayscale, often single channel
                    std=self.cfg.dataset_norm.std.item(),   # MNIST is grayscale, often single channel
                )
            else: # CIFAR-10 etc. (3-channel images)
                preprocessing = dict(
                    mean=self.cfg.dataset_norm.mean.tolist(),
                    std=self.cfg.dataset_norm.std.tolist(),
                    axis=-3 # Channel dimension for (C,H,W)
                )
            bounds = (0.0, 1.0) # Assumes normalized images are 0-1 range

            try:
                fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
                attack = foolbox.attacks.LinfBasicIterativeAttack(steps=n, abs_stepsize=alpha) # type: ignore[reportPrivateImportUsage]
                
                # Foolbox expects normalized images as input
                raw, clipped, is_adv = attack(fmodel, data.tensor, target, epsilons=eps)
                __perturbed = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
                return TensorWithState(__perturbed, NORMALIZED)
            except Exception as e:
                print(f"Foolbox BIM attack ({kind.value}) failed: {e}. Returning original data.")
                return data # Return original data if attack fails
        else:
            raise ValueError(f"unsupported attack kind: {kind}")

    def run(self) -> Iterator[Tuple]:
        print(f"Running Re-Attack with config: {self.cfg}")
        self.model.eval()

        global_idx = 0
        for data, target in tqdm(self.test_loader):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                break
            
            current_batch_size = data.shape[0]
            clean_data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
            clean_data_ts_norm = self.cfg.dataset_norm.normalize(clean_data_ts)
            target_ts: Tensor = target.to(self.device).view(-1).long()
            
            # --- Predictions for clean data ---
            logits_clean = self._to_logits(self.model(clean_data_ts_norm.tensor))
            pred_clean = logits_clean.max(1, keepdim=True)[1]

            # --- Initial Attack ---
            attacked_data_ts_norm = self._perform_attack(
                clean_data_ts_norm, 
                target_ts, 
                self.cfg.attack_kind,
                eps=self.cfg.attack_eps,
                alpha=self.cfg.attack_alpha,
                n=self.cfg.attack_n
            )
            logits_attacked = self._to_logits(self.model(attacked_data_ts_norm.tensor))
            pred_attacked = logits_attacked.max(1, keepdim=True)[1]

            # --- Re-Attack ---
            # The target for re-attack is the prediction of the attacked image
            reattack_target = pred_attacked.view(-1)
            reattacked_data_ts_norm = self._perform_attack(
                attacked_data_ts_norm, 
                reattack_target, 
                self.cfg.reattack_kind,
                eps=self.cfg.reattack_eps,
                alpha=self.cfg.reattack_alpha,
                n=self.cfg.reattack_n
            )
            logits_reattacked = self._to_logits(self.model(reattacked_data_ts_norm.tensor))
            pred_reattacked = logits_reattacked.max(1, keepdim=True)[1]

            # --- Denormalize and calculate L2 norms ---
            clean_data_denorm = self.cfg.dataset_norm.denormalize(clean_data_ts_norm).tensor
            attacked_data_denorm = self.cfg.dataset_norm.denormalize(attacked_data_ts_norm).tensor
            reattacked_data_denorm = self.cfg.dataset_norm.denormalize(reattacked_data_ts_norm).tensor

            l2_clean_vs_attacked = torch.linalg.norm((clean_data_denorm - attacked_data_denorm).view(current_batch_size, -1), ord=2, dim=1)
            l2_attacked_vs_reattacked = torch.linalg.norm((attacked_data_denorm - reattacked_data_denorm).view(current_batch_size, -1), ord=2, dim=1)
            l2_clean_vs_reattacked = torch.linalg.norm((clean_data_denorm - reattacked_data_denorm).view(current_batch_size, -1), ord=2, dim=1)

            # --- Yield results for each sample in the batch ---
            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                    return

                reattacked_sample_denorm_ts = TensorWithState(reattacked_data_denorm[j].detach().cpu(), DENORMALIZED)

                yield (
                    global_idx,
                    target_ts[j].item(),
                    pred_clean.view(-1)[j].item(),
                    pred_attacked.view(-1)[j].item(),
                    pred_reattacked.view(-1)[j].item(),
                    l2_clean_vs_attacked[j].item(),
                    l2_attacked_vs_reattacked[j].item(),
                    l2_clean_vs_reattacked[j].item(),
                    reattacked_sample_denorm_ts
                )
                global_idx += 1

def fraction_float(s: str) -> float:
    if "/" in s:
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            raise argparse.ArgumentTypeError(f'"{s}" is not a valid fraction.')
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{s}" is not a valid float.')

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="General Re-Attack Runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], default=DatasetKind.MNIST.value)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], default=ModelKind.MORIMOTO_MNIST.value)
    parser.add_argument("--model-dir", default="./weight")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="reattacked_data")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to process. -1 for all samples.")

    # Initial Attack args
    parser.add_argument("--attack-kind", choices=[a.value for a in AttackKind if a in [AttackKind.BIM, AttackKind.FGSM]], default=AttackKind.BIM.value)
    parser.add_argument("--attack-eps", type=fraction_float, default=0.3)
    parser.add_argument("--attack-alpha", type=fraction_float, default=0.05)
    parser.add_argument("--attack-n", type=int, default=10)
    
    # Re-Attack args
    parser.add_argument("--reattack-kind", choices=[a.value for a in AttackKind if a in [AttackKind.BIM, AttackKind.FGSM]], default=AttackKind.BIM.value)
    parser.add_argument("--reattack-eps", type=fraction_float, default=0.3)
    parser.add_argument("--reattack-alpha", type=fraction_float, default=0.05)
    parser.add_argument("--reattack-n", type=int, default=10)

    args = parser.parse_args()
    return Config(
        dataset=DatasetKind(args.dataset),
        model=ModelKind(args.model),
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        device=utils.get_device(),
        dataset_norm=DatasetNorm(DatasetKind(args.dataset), utils.get_device()),
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        attack_kind=AttackKind(args.attack_kind),
        attack_eps=args.attack_eps,
        attack_alpha=args.attack_alpha,
        attack_n=args.attack_n,
        reattack_kind=AttackKind(args.reattack_kind),
        reattack_eps=args.reattack_eps,
        reattack_alpha=args.reattack_alpha,
        reattack_n=args.reattack_n,
    )

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result_generator = runner.run()

    # --- Prepare directories and CSV ---
    attack_name = f"{cfg.attack_kind.value}_eps{cfg.attack_eps:.3f}"
    reattack_name = f"{cfg.reattack_kind.value}_eps{cfg.reattack_eps:.3f}"
    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.model.value, attack_name, reattack_name)
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "reattack_results.csv")
    csv_header = [
        "index", "target_label", "pred_clean", "pred_attacked", "pred_reattacked",
        "l2_clean_vs_attacked", "l2_attacked_vs_reattacked", "l2_clean_vs_reattacked",
        "reattacked_image_path"
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(csv_header)

    # --- Process results ---
    stats = {
        'total': 0,
        'clean_correct': 0,
        'attack_successful': 0,
        'reattack_successful_to_clean': 0,
        'reattack_successful_to_any': 0,
    }

    for result_tuple in result_generator:
        (idx, target, pred_c, pred_a, pred_r, l2_c_a, l2_a_r, l2_c_r, image_ts) = result_tuple
        
        stats['total'] += 1
        is_clean_correct = (pred_c == target)
        if is_clean_correct:
            stats['clean_correct'] += 1
            if pred_a != pred_c:
                stats['attack_successful'] += 1
                if pred_r == pred_c:
                    stats['reattack_successful_to_clean'] += 1
                if pred_r != pred_a:
                    stats['reattack_successful_to_any'] += 1

        # Save image
        image_path = os.path.join(output_folder, f"idx_{idx}_target{target}_r{pred_r}.png")
        utils.save_tensor_as_image(image_ts.tensor, image_path)
        
        # Write to CSV
        csv_row = [idx, target, pred_c, pred_a, pred_r, l2_c_a, l2_a_r, l2_c_r, image_path]
        with open(csv_filename, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow(csv_row)

    # --- Print Summary ---
    total = stats['total']
    clean_correct = stats['clean_correct']
    attack_successful = stats['attack_successful']
    reattack_to_clean = stats['reattack_successful_to_clean']
    
    print("\n=== Re-Attack Summary ===")
    print(f"Processed {total} samples.")
    print(f"Clean Accuracy: {clean_correct / total:.4f} ({clean_correct}/{total})")
    if clean_correct > 0:
        print(f"Initial Attack Success Rate: {attack_successful / clean_correct:.4f} ({attack_successful}/{clean_correct})")
        if attack_successful > 0:
            print(f"Re-Attack Success Rate (to Clean): {reattack_to_clean / attack_successful:.4f} ({reattack_to_clean}/{attack_successful})")
    print(f"Results saved to {csv_filename}")
    print("==========================")


if __name__ == "__main__":
    main()
