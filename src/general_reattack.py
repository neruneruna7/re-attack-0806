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

from enum import Enum

class TargetLabelSource(Enum):
    ORIGINAL_PRED = "original_pred"  # 元のモデルの予測ラベル
    TRUE_LABEL = "true_label"        # 真のラベル
    FIXED_LABEL = "fixed_label"      # 固定値のラベル

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

    # Re-attack specific parameters
    attacked_image_dir: str = "data/attacked_images"
    reattack_epsilon: float = 0.05
    reattack_alpha: float = 0.01
    reattack_n: int = 1
    target_label_source: TargetLabelSource = TargetLabelSource.ORIGINAL_PRED
    fixed_target_label: Optional[int] = None
    re_attack_kind: AttackKind = AttackKind.FGSM

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device or utils.get_device()
        self.model = ModelFactory.create(cfg.model, self.device)
        # 既存の test_loader は再攻撃では使わないため削除またはコメントアウト
        # self.test_loader = DataFactory.loader(cfg.dataset, train=False, batch_size=cfg.batch_size)

        # 攻撃済み画像をロードするための処理
        self.attacked_images_info = self._get_attacked_image_paths()
        
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

    def _get_attacked_image_paths(self) -> List[Tuple[str, float, int, int]]:
        """
        指定されたディレクトリから攻撃済み画像のパスと関連情報を取得する
        戻り値: (ファイルパス, 元のイプシロン, 画像インデックス, 真のラベル) のリスト
        """
        image_info_list = []
        # attacked_image_dir は "data/attacked_images" のような形式を想定
        # そのサブディレクトリは "eps_0.100" のような形式を想定
        # さらにその中に "attacked_idx_0_label_5.pt" のようなファイルがあることを想定
        
        # まず、ベースディレクトリ内のイプシロンごとのサブディレクトリを走査
        for eps_dir_name in os.listdir(self.cfg.attacked_image_dir):
            eps_dir_path = os.path.join(self.cfg.attacked_image_dir, eps_dir_name)
            if not os.path.isdir(eps_dir_path):
                continue
            
            try:
                # フォルダ名からイプシロン値を取得 (例: "eps_0.100" -> 0.1)
                eps_str = eps_dir_name.split('_')[-1]
                original_epsilon = float(eps_str)
            except ValueError:
                print(f"Skipping directory with invalid epsilon format: {eps_dir_name}")
                continue

            # サブディレクトリ内の画像ファイルを走査
            for file_name in os.listdir(eps_dir_path):
                # .pt または .pth ファイルのみを対象とする
                if file_name.endswith(".pt") or file_name.endswith(".pth"):
                    file_path = os.path.join(eps_dir_path, file_name)
                    try:
                        # ファイル名から index と true_label を取得
                        # utils.from_filename は "attacked_12_label_3.png" のようなファイル名を想定しているため
                        # ".pt" を削除したファイル名で渡す必要がある
                        idx, true_label = utils.from_filename(file_name.replace(".pt", "").replace(".pth", ""))
                        image_info_list.append((file_path, original_epsilon, idx, true_label))
                    except ValueError as e:
                        print(f"Skipping file with invalid filename format: {file_name}. Error: {e}")
                        continue
        return image_info_list

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
        print(f"Running re-attack {self.cfg.re_attack_kind} on model {self.cfg.model} with cfg: {self.cfg}")
        self.model.eval()
        reattack_examples = []

        for file_path, original_epsilon, original_idx, true_label in self.attacked_images_info:
            # 攻撃済み画像をロード
            # _get_attacked_image_paths でファイルパスを取得済みなので直接ロード
            img_denorm: Tensor = torch.load(file_path, map_location=self.device)
            # 画像がバッチ次元を持つか確認し、必要であれば追加
            if img_denorm.dim() == 3:
                img_denorm = img_denorm.unsqueeze(0) # (C, H, W) -> (1, C, H, W)

            # denorm -> norm
            # データタイプを cfg.dataset_norm.normalize の期待値に合わせる
            data_raw = TensorWithState(img_denorm, DENORMALIZED)
            data = self.cfg.dataset_norm.normalize(data_raw).to(self.device)
            data.tensor.requires_grad = True

            # ターゲットラベルの決定
            with torch.no_grad():
                original_output = self.model(data.tensor)
                original_logits = self._to_logits(original_output)
                original_pred = original_logits.max(1, keepdim=True)[1].item()

            target_label: Tensor
            if self.cfg.target_label_source == TargetLabelSource.ORIGINAL_PRED:
                target_label = torch.tensor([original_pred], device=self.device)
            elif self.cfg.target_label_source == TargetLabelSource.TRUE_LABEL:
                target_label = torch.tensor([true_label], device=self.device)
            elif self.cfg.target_label_source == TargetLabelSource.FIXED_LABEL:
                if self.cfg.fixed_target_label is None:
                    raise ValueError("fixed_target_label must be set when target_label_source is FIXED_LABEL")
                target_label = torch.tensor([self.cfg.fixed_target_label], device=self.device)
            else:
                raise ValueError(f"unsupported target_label_source: {self.cfg.target_label_source}")

            target_label = target_label.view(-1).long()


            # 勾配計算 (再攻撃のための勾配)
            # BIMやFGSMなどの勾配ベースの攻撃では、勾配を計算する必要がある
            # ここでは再攻撃対象となるモデルの出力に対する損失を計算し、その勾配を得る
            output_for_grad = self.model(data.tensor)
            logits_for_grad = self._to_logits(output_for_grad)
            loss_for_grad = F.cross_entropy(logits_for_grad, target_label)
            self.model.zero_grad()
            loss_for_grad.backward()
            grad = data.tensor.grad
            if grad is None:
                print(f"skip file {file_path}: no grad")
                continue
            data_grad = TensorWithState(grad.data, NORMALIZED)

            # 再攻撃実行
            perturbed_data: TensorWithState
            if self.cfg.re_attack_kind == AttackKind.BIM:
                # BIM は (data_grad, target, model, device, epsilon, alpha, n) を想定
                perturbed_data = bim.bim(
                    data_grad,
                    target_label,
                    self.model,
                    self.device,
                    self.cfg.reattack_epsilon,
                    self.cfg.reattack_alpha,
                    self.cfg.reattack_n,
                )
            elif self.cfg.re_attack_kind == AttackKind.FGSM:
                # FGSM は (data_grad, epsilon, target, model, device) を想定
                perturbed_data = fgsm.fgsm(
                    data_grad,
                    self.cfg.reattack_epsilon, # 再攻撃用のイプシロン
                    target_label,
                    self.model,
                    self.device,
                )
            else:
                raise ValueError(f"unsupported re-attack kind: {self.cfg.re_attack_kind}")

            # 再攻撃後の予測
            out_after_reattack = self.model(perturbed_data.tensor)
            logits_after_reattack = self._to_logits(out_after_reattack)
            pred_after_reattack = logits_after_reattack.max(1, keepdim=True)[1].item()

            average_perturbation = utils.l2_norm_perturbation(
                self.cfg.dataset_norm.denormalize(data), # 元の攻撃済み画像（非正規化）
                self.cfg.dataset_norm.denormalize(perturbed_data) # 再攻撃後の画像（非正規化）
            )

            reattack_examples.append((original_idx, original_pred, pred_after_reattack, perturbed_data.tensor.squeeze().detach().cpu(), average_perturbation.item()))

        return reattack_examples
    
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="general AE attack runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], default=DatasetKind.MNIST.value)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], default=ModelKind.MORIMOTO_MNIST.value)
    # attack は元の攻撃の種類なので、ここでは使用しないかもしれないが、互換性のため残す
    parser.add_argument("--attack", choices=[a.value for a in AttackKind], default=AttackKind.BIM.value)
    parser.add_argument("--model-dir", default="./weight")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=10, help="number of iterations for BIM")
    parser.add_argument("--batch-size", type=int, default=1)

    # Re-attack specific arguments
    parser.add_argument("--attacked-image-dir", type=str, default="data/attacked_images",
                        help="Directory containing previously attacked images")
    parser.add_argument("--reattack-epsilon", type=float, default=0.05,
                        help="Epsilon for re-attack")
    parser.add_argument("--reattack-alpha", type=float, default=0.01,
                        help="Alpha for re-attack (BIM only)")
    parser.add_argument("--reattack-n", type=int, default=1,
                        help="Number of iterations for re-attack (BIM only)")
    parser.add_argument("--target-label-source", choices=[tls.value for tls in TargetLabelSource],
                        default=TargetLabelSource.ORIGINAL_PRED.value,
                        help="Source for target labels in re-attack")
    parser.add_argument("--fixed-target-label", type=int, default=None,
                        help="Fixed target label when target-label-source is FIXED_LABEL")
    parser.add_argument("--re-attack-kind", choices=[ak.value for ak in AttackKind],
                        default=AttackKind.FGSM.value,
                        help="Kind of attack for re-attack (e.g., FGSM, BIM)")

    args = parser.parse_args()
    cfg = Config(
        dataset=DatasetKind(args.dataset),
        model=ModelKind(args.model),
        attack=AttackKind(args.attack), # この値は再攻撃では直接使われないが、Configの互換性のため残す
        model_dir=args.model_dir,
        epsilon=args.epsilon, # この値も再攻撃では直接使われないが、Configの互換性のため残す
        alpha=args.alpha,     # この値も再攻撃では直接使われないが、Configの互換性のため残す
        n=args.n,             # この値も再攻撃では直接使われないが、Configの互換性のため残す
        batch_size=args.batch_size, # この値も再攻撃では直接使われないが、Configの互換性のため残す
        device=utils.get_device(),
        dataset_norm=DatasetNorm(DatasetKind(args.dataset), utils.get_device()),

        # Re-attack specific parameters
        attacked_image_dir=args.attacked_image_dir,
        reattack_epsilon=args.reattack_epsilon,
        reattack_alpha=args.reattack_alpha,
        reattack_n=args.reattack_n,
        target_label_source=TargetLabelSource(args.target_label_source),
        fixed_target_label=args.fixed_target_label,
        re_attack_kind=AttackKind(args.re_attack_kind)
    )
    return cfg

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result = runner.run()

    # 統計を計算:
    total_images = len(result)
    successful_reattacks = 0
    mean_perturbation = 0.0

    if total_images == 0:
        print("No images found for re-attack.")
        return

    for original_idx, original_pred, pred_after_reattack, reattacked_image, perturbation in result:
        mean_perturbation += perturbation
        # 再攻撃前の予測と再攻撃後の予測が異なる場合、再攻撃成功とみなす
        if original_pred != pred_after_reattack:
            successful_reattacks += 1
            # print(f"Index: {original_idx}, Original Pred: {original_pred}, Re-attacked Pred: {pred_after_reattack}, Perturbation: {perturbation:.4f}")

    reattack_success_rate = (successful_reattacks / total_images) if total_images > 0 else 0.0
    mean_perturbation_avg = (mean_perturbation / total_images) if total_images > 0 else 0.0

    print(f"--- Re-attack Summary ---")
    print(f"Re-attack Kind: {cfg.re_attack_kind.value}")
    print(f"Target Model: {cfg.model.value}")
    print(f"Total Images Processed: {total_images}")
    print(f"Successful Re-attacks: {successful_reattacks}")
    print(f"Re-attack Success Rate = {reattack_success_rate:.4f} = {successful_reattacks} / {total_images}")
    print(f"Average Perturbation (L2 norm): {mean_perturbation_avg:.4f}")

    # FIXED_LABELの場合の追加情報
    if cfg.target_label_source == TargetLabelSource.FIXED_LABEL:
        print(f"Target Label Source: FIXED_LABEL ({cfg.fixed_target_label})")

if __name__ == "__main__":
    main()