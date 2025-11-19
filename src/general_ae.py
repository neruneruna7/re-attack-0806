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
import csv
from PIL import Image
from torch import Tensor
from enum import Enum
import argparse
import re_attack_0806
from copy import deepcopy

from re_attack_0806 import attacks, utils
from re_attack_0806.attacks import bim, fgsm, linfbim
# lib 以下の attacks パッケージを使用（fgsm と bim を実装）

import foolbox

from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm
from re_attack_0806.utils.normTensor import *

from tqdm import tqdm

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
    save_attacked_images: bool = True # 攻撃後の画像を保存するかどうかのフラグ
    output_dir: str = "attacked_data" # 保存先ディレクトリ

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device or utils.get_device()
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, train=False, batch_size=cfg.batch_size)
        self._load_weights_if_exists(cfg.model_dir)

    def _load_weights_if_exists(self, model_dir: str):
        # model に model_name 属性が無ければ何もしないで戻る（Inception3 等の外部モデル対策）
        model_name = getattr(self.model, "model_name", None)
        if model_name is None:
            # デバッグ用にクラス名を通知（必要ならここでフォールバック名を使う実装に差し替え可能）
            print(f"warning: model has no 'model_name' attribute (model class: {self.model.__class__.__name__}), skipping weight load")
            return None
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
    
    def run(self) -> List[Tuple[int, int, int, int, TensorWithState, float]]:
        print(f"Running attack {self.cfg.attack} on model {self.cfg.model} with cfg: {self.cfg}")
        self.model.eval()
        adv_examples = []

        for i, (data, target) in tqdm(enumerate(self.test_loader)):
            # if i >= 100:  # 最初の100サンプルのみを攻撃
            #     break
            data = data.to(self.device)
            data = TensorWithState(data, DENORMALIZED)
            data = self.cfg.dataset_norm.normalize(data)

            target: Tensor = target.to(self.device).view(-1).long()
            

            # 順伝播
            output = self.model(data.tensor)
            logits = self._to_logits(output)
            pred = logits.max(1, keepdim=True)[1]

            print(f"\n[Debug] Index: {i}")
            print(f"  Dataset Label (target): {target.item()}")
            print(f"  Model Prediction (pred): {pred.item()}")

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
                    data,
                    target,
                    self.model,
                    self.device,
                    self.cfg.epsilon,
                    self.cfg.alpha,
                    self.cfg.n,
                    self.cfg.dataset_norm.mean,
                    self.cfg.dataset_norm.std,
                )
                # print(f"perturbed shape: {perturbed.tensor.shape}")
            elif self.cfg.attack == AttackKind.FOOLBOX_BIM:
                # Foolbox による Linf Basic Iterative Attack を使用
                preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
                bounds = (0.0, 1.0)
                try:
                    # pylance の警告を抑制する（# type: ignore を付与）
                    fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
                    print("fmodel created")
                    attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.n, abs_stepsize=self.cfg.alpha) # type: ignore[reportPrivateImportUsage]
                    print("attack created")
                    # ここまでpylanceの警告を抑制する．
                    # denormData = self.cfg.dataset_norm.denormalize(data)
                    raw, clipped, is_adv = attack(fmodel, data.tensor, target, epsilons=self.cfg.epsilon)
                    print("attack executed")
                    # clipped は foolbox のバージョンにより単体のテンソルかリストになることがある
                    __perturbed = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]


                    perturbed = TensorWithState(__perturbed, NORMALIZED)
                except Exception as e:
                    print(f"Foolbox BIM attack failed: {e}")
                    continue
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
            # elif self.cfg.attack == AttackKind.LINF_BIM:
            #     # Linf BIM を自前実装で実行
            #     perturbed = linfbim.linf_bim_attack_with_normalization(
            #         self.model, da
            #     )
            elif self.cfg.attack == AttackKind.FGSM:
                # FGSM は (image_denorm, epsilon, target, model, device) を想定
                perturbed = fgsm.fgsm(data, self.cfg.epsilon, target, self.model, self.device)
            else:
                raise ValueError(f"unsupported attack kind: {self.cfg.attack}")
    
            # # perturbed_norm = self.cfg.dataset_norm.normalize(perturbed)
            # print(f"original data average: {data.tensor.mean().item()}")
            # # print(f"perturbed_norm average: {perturbed_norm.mean().item()}")
            # print(f"perturbed average: {perturbed.tensor.mean().item()}")

            average_perturbation = utils.l2_norm_perturbation(
                self.cfg.dataset_norm.denormalize(data), 
                self.cfg.dataset_norm.denormalize(perturbed))
            # average_perturbation = utils.l2_norm_perturbation(data, perturbed)

            out2 = self.model(perturbed.tensor)
            logits2 = self._to_logits(out2)
            pred2 = logits2.max(1, keepdim=True)[1]

            # mean_perturb = attacks____.mean_perturbation(data, perturbed)
            # print(f"average_perturbation: {average_perturbation}")


            # print(f"mean_perturb {mean_perturb}")
            denorm_perturbed = self.cfg.dataset_norm.denormalize(perturbed)
            denorm_perturbed = TensorWithState(denorm_perturbed.tensor.squeeze().detach().cpu(), DENORMALIZED)
            adv_examples.append((i, target.item(), pred.item(), pred2.item(), denorm_perturbed, average_perturbation.item()))
        return adv_examples
    
def fraction_float(s: str) -> float:
    """
    argparseで分数形式の文字列（例: "8/255"）をfloatに変換するための型関数。
    """
    if "/" in s:
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            raise argparse.ArgumentTypeError(f'"{s}" は不正な分数形式です。')
    try:
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{s}" は不正なfloat値です。')


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="general AE attack runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], default=DatasetKind.MNIST.value)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], default=ModelKind.MORIMOTO_MNIST.value)
    parser.add_argument("--attack", choices=[a.value for a in AttackKind], default=AttackKind.BIM.value)
    parser.add_argument("--model-dir", default="./weight")
    parser.add_argument("--epsilon", type=fraction_float, default=0.3)
    parser.add_argument("--alpha", type=fraction_float, default=0.05)
    parser.add_argument("--n", type=int, default=10, help="number of iterations for BIM")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-attacked-images", action="store_true", default=True,
                        help="Save attacked images to specified output directory")
    parser.add_argument("--output-dir", type=str, default="attacked_data",
                        help="Directory to save attacked images")
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
        dataset_norm=DatasetNorm(DatasetKind(args.dataset), utils.get_device()),
        save_attacked_images=args.save_attacked_images,
        output_dir=args.output_dir
    )
    return cfg

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result = runner.run()

    # 統計を計算: 最初に正しく分類されていたサンプルのみを考慮
    fail = 0
    clean_acc_total = 0
    total = 0
    mean_mean_perturb = 0.0

    # CSV保存の準備
    dataset_name = cfg.dataset.value
    attack_name = cfg.attack.value
    output_folder = os.path.join(cfg.output_dir, dataset_name, attack_name, f"eps_{cfg.epsilon:.3f}")
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "attack_results.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["index", "target_label", "prediction_before_attack", "prediction_after_attack", "l2_perturbation", "image_filepath"])

    for idx, target, before, after, ex, m in result:
        total += 1
        # 元の正解ラベルが必要; 必要ならデータセットから再読み込みしても良い
        # ここでは以前のモデルの初期予測を用いて比較することを想定
        if before == target:
            clean_acc_total += 1
            mean_mean_perturb += m
            if after != before:
                fail += 1

        output_filename = "" # 画像ファイルパスの初期化
        # 攻撃後の画像を保存する処理
        if cfg.save_attacked_images:
            # ex は perturbed.tensor.squeeze().detach().cpu() なので非正規化画像
            # ファイル名を構築
            # 例: data/attacked_images/mnist/bim/eps_0.300/idx_0_label_5.png
            output_filename = os.path.join(output_folder, f"idx_{idx}_label_{after}.png")
            
            # utils.save_tensor_as_image を使用
            # save_tensor_as_image はテンソルを0-1にクリップしてuint8に変換
            # ex は既に非正規化されたテンソルなのでそのまま渡せる
            # ただし、保存はPNGで行うため、テンソルの状態は問わない
            # utils.save_tensor_as_image は `Tensor` を受け取るので `ex` をそのまま渡す
            utils.save_tensor_as_image(ex.tensor, output_filename)
            # print(f"Saved attacked image to {output_filename}")
        
        # CSVに結果を追記
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([idx, target, before, after, m, output_filename])


    attack_acc = (fail / clean_acc_total) if clean_acc_total > 0 else 0.0
    mean_perturb = (mean_mean_perturb / clean_acc_total) if clean_acc_total > 0 else 0.0
    print("\n=== Attack Summary ===")
    print(f"Epsilon: {cfg.epsilon}\t攻撃成功率= {attack_acc} = {fail} / {clean_acc_total}")
    print(f"攻撃成功率には，元の画像を正しく分類したサンプルのみを考慮している．")
    print(f"平均摂動量(L2ノルム) = {mean_perturb}")
    print(f"クリーン画像を正しく分類したサンプル数: {clean_acc_total}")
    print(f"処理したサンプル総数: {total}")
    print(f"結果は {csv_filename} に保存されました。")

    print("\n=== End of Attack Summary ===")

if __name__ == "__main__":
    main()