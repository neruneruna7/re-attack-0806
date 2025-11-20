# モデルをトレーニングする汎用コード
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
import concurrent.futures # パフォーマンス改善のために追加

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
    batch_size: int = 64 # パフォーマンス改善のためデフォルト値を変更
    device: Optional[torch.device] = None
    dataset_norm: DatasetNorm = DatasetNorm(DatasetKind.MNIST, torch.device("cpu"))
    save_attacked_images: bool = True # 攻撃後の画像を保存するかどうかのフラグ
    output_dir: str = "attacked_data" # 保存先ディレクトリ
    num_samples: int = -1 # 処理するサンプル数 (-1は全サンプル)

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
    
    def run(self) -> Iterator[Tuple[int, Number, Number, Number, TensorWithState, Number]]:
        print(f"Running attack {self.cfg.attack} on model {self.cfg.model} with cfg: {self.cfg}")
        self.model.eval()

        # パフォーマンス改善：foolboxモデルの初期化をループ外に移動
        fmodel = None
        foolbox_attack = None
        if self.cfg.attack == AttackKind.FOOLBOX_BIM:
            preprocessing = dict(mean=self.cfg.dataset_norm.mean_list, std=self.cfg.dataset_norm.std_list, axis=-3)
            bounds = (0.0, 1.0)
            fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
            foolbox_attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.n, abs_stepsize=self.cfg.alpha) # type: ignore[reportPrivateImportUsage]


        global_idx = 0
        for i, (data, target) in tqdm(enumerate(self.test_loader)):
            # # デバッグ用に処理するサンプル数を100に制限
            # if global_idx >= 100:
            #     break
            
            current_batch_size = data.shape[0]

            # `data` テンソルを TensorWithState にラップして正規化
            data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
            data_ts_norm = self.cfg.dataset_norm.normalize(data_ts)

            target_ts: Tensor = target.to(self.device).view(-1).long()
            
            # 順伝播（攻撃前）
            output = self.model(data_ts_norm.tensor)
            logits = self._to_logits(output)
            pred = logits.max(1, keepdim=True)[1]

            # 攻撃実行
            if self.cfg.attack == AttackKind.BIM:
                perturbed = bim.bim(
                    data_ts_norm,
                    target_ts,
                    self.model,
                    self.device,
                    self.cfg.epsilon,
                    self.cfg.alpha,
                    self.cfg.n,
                    self.cfg.dataset_norm.mean,
                    self.cfg.dataset_norm.std,
                )
            elif self.cfg.attack == AttackKind.FOOLBOX_BIM:
                try:
                    # パフォーマンス改善：初期化済みのモデルと攻撃を使用
                    assert fmodel is not None and foolbox_attack is not None, "Foolbox model and attack should have been initialized"
                    
                    # foolboxには[0,1]の範囲の非正規化データを渡す
                    # foolbox内部でpreprocessingに指定した方法で正規化される
                    raw, clipped, is_adv = foolbox_attack(fmodel, data_ts.tensor, target_ts, epsilons=self.cfg.epsilon)
                    
                    # foolboxの出力(clipped)は非正規化データなので、後続の処理のために正規化する
                    __perturbed_denorm = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
                    perturbed_denorm_ts = TensorWithState(__perturbed_denorm, DENORMALIZED)
                    perturbed = self.cfg.dataset_norm.normalize(perturbed_denorm_ts)
                except Exception as e:
                    print(f"Foolbox BIM attack failed: {e}")
                    continue
            elif self.cfg.attack == AttackKind.FGSM:
                perturbed = fgsm.fgsm(data_ts_norm, self.cfg.epsilon, target_ts, self.model, self.device)
            else:
                raise ValueError(f"unsupported attack kind: {self.cfg.attack}")

            # 順伝播（攻撃後）
            out2 = self.model(perturbed.tensor)
            logits2 = self._to_logits(out2)
            pred2 = logits2.max(1, keepdim=True)[1]

            # 摂動の計算（バッチ対応）
            denormalized_data = self.cfg.dataset_norm.denormalize(data_ts_norm)
            denormalized_perturbed = self.cfg.dataset_norm.denormalize(perturbed)
            
            # バッチ内の各サンプルのL2ノルムを計算
            perturbation_batch = denormalized_perturbed.tensor - denormalized_data.tensor
            l2_perturbations = torch.linalg.norm(perturbation_batch.view(current_batch_size, -1), ord=2, dim=1)

            # バッチ内の各サンプルについて結果をyield
            for j in range(current_batch_size):
                # num_samplesが指定されている場合、制限を超えたらジェネレータを終了
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                    return
                
                perturbed_sample_denorm = TensorWithState(denormalized_perturbed.tensor[j].detach().cpu(), DENORMALIZED)
                
                yield (
                    global_idx,
                    target_ts[j].item(),
                    pred.view(-1)[j].item(),
                    pred2.view(-1)[j].item(),
                    perturbed_sample_denorm,
                    l2_perturbations[j].item()
                )
                global_idx += 1
    
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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing data. Larger is faster on GPU.") # パフォーマンス改善
    parser.add_argument("--save-attacked-images", action="store_true", default=True,
                        help="Save attacked images to specified output directory")
    parser.add_argument("--output-dir", type=str, default="attacked_data",
                        help="Directory to save attacked images")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to process. -1 for all samples.")
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
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    return cfg

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result_generator = runner.run()

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
    
    # パフォーマンス改善: ThreadPoolExecutorとCSV writerをループの外で初期化
    with concurrent.futures.ThreadPoolExecutor() as executor, \
         open(csv_filename, 'w', newline='') as csvfile:
        
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["index", "target_label", "prediction_before_attack", "prediction_after_attack", "l2_perturbation", "image_filepath"])

        for idx, target, before, after, ex, m in result_generator:
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
                output_filename = os.path.join(output_folder, f"idx_{idx}_label_{after}.png")
                # パフォーマンス改善: 画像保存を非同期で実行
                executor.submit(utils.save_tensor_as_image, ex.tensor, output_filename)
            
            # CSVに結果を追記
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
