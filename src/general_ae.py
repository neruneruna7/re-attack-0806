# モデルをトレーニングする汎用コード
from dataclasses import dataclass, asdict
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
import sys
from PIL import Image
from torch import Tensor
from enum import Enum
import argparse
import re_attack_0806
from copy import deepcopy
import concurrent.futures

from re_attack_0806 import attacks, utils
from re_attack_0806.attacks import bim, fgsm, linfbim
import foolbox

# AttackParamsをインポート
from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm, AttackParams, FGSMAttackParam, BIMAttackParam
from re_attack_0806.utils.normTensor import *

from tqdm import tqdm

# --- 定数定義 ---
DEFAULT_MODEL_DIR = "./weight"
DEFAULT_OUTPUT_DIR = "attacked_data"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_SAMPLES = -1
DEFAULT_SHUFFLE_DATALOADER = False # 新しく追加


@dataclass
class Config:
    # 必須パラメータ
    dataset: DatasetKind
    model: ModelKind
    attack: AttackKind
    attack_params: AttackParams  # 攻撃パラメータを構造化
    
    # デフォルト値を持つパラメータ
    model_dir: str
    output_dir: str
    batch_size: int
    num_samples: int
    save_attacked_images: bool
    shuffle_dataloader: bool # 新しく追加

    # デバイスと正規化情報（内部で設定）
    device: torch.device
    dataset_norm: DatasetNorm

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, train=False, batch_size=cfg.batch_size, shuffle=cfg.shuffle_dataloader)
        self._load_weights_if_exists(cfg.model_dir)
        self.model.eval()

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
    
    def run(self) -> Iterator[Tuple[int, Number, Number, Number, TensorWithState, Number]]:
        print(f"Running attack {self.cfg.attack} on model {self.cfg.model} with cfg: {self.cfg}")

        fmodel = None
        foolbox_attack = None
        if self.cfg.attack in [AttackKind.FOOLBOX_BIM, AttackKind.FOOLBOX_FGSM]:
            mean_list = self.cfg.dataset_norm.mean.squeeze().tolist()
            std_list = self.cfg.dataset_norm.std.squeeze().tolist()
            preprocessing = dict(mean=mean_list, std=std_list, axis=-3)
            bounds = (0.0, 1.0)
            fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
            if self.cfg.attack == AttackKind.FOOLBOX_BIM:
                assert isinstance(self.cfg.attack_params, BIMAttackParam), "Invalid params for FoolboxBIM"
                foolbox_attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.attack_params.iters, abs_stepsize=self.cfg.attack_params.alpha) # type: ignore[reportPrivateImportUsage]
            elif self.cfg.attack == AttackKind.FOOLBOX_FGSM:
                assert isinstance(self.cfg.attack_params, FGSMAttackParam), "Invalid params for FoolboxFGSM"
                foolbox_attack = foolbox.attacks.FGSM()
        elif self.cfg.attack == AttackKind.FOOLBOX_FGSM:
            assert isinstance(self.cfg.attack_params, FGSMAttackParam), "Invalid params for FoolboxFGSM"
            mean_list = self.cfg.dataset_norm.mean.squeeze().tolist()
            std_list = self.cfg.dataset_norm.std.squeeze().tolist()
            preprocessing = dict(mean=mean_list, std=std_list, axis=-3)
            bounds = (0.0, 1.0)
            fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
            foolbox_attack = foolbox.attacks.FGSM(random_start=False) # type: ignore[reportPrivateImportUsage]

        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Attacking"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                break
            
            current_batch_size = data.shape[0]
            data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
            data_ts_norm = self.cfg.dataset_norm.normalize(data_ts)
            target_ts: Tensor = target.to(self.device).view(-1).long()
            
            output = self.model(data_ts_norm.tensor)
            logits = self._to_logits(output)
            pred = logits.max(1, keepdim=True)[1]

            if self.cfg.attack == AttackKind.BIM:
                assert isinstance(self.cfg.attack_params, BIMAttackParam), "Invalid params for BIM"
                params = self.cfg.attack_params
                perturbed = bim.bim(
                    data_ts_norm, target, self.model, self.device, params.epsilon,
                    params.alpha, params.iters, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std
                )
            elif self.cfg.attack in [AttackKind.FOOLBOX_BIM, AttackKind.FOOLBOX_FGSM]:
                assert fmodel is not None and foolbox_attack is not None, "Foolbox components not initialized"
                try:
                    raw, clipped, is_adv = foolbox_attack(fmodel, data_ts.tensor, target_ts, epsilons=self.cfg.attack_params.epsilon)
                    __perturbed_denorm = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
                    perturbed_denorm_ts = TensorWithState(__perturbed_denorm, DENORMALIZED)
                    perturbed = self.cfg.dataset_norm.normalize(perturbed_denorm_ts)
                except Exception as e:
                    print(f"Foolbox {self.cfg.attack.value} attack failed: {e}")
                    continue
            elif self.cfg.attack == AttackKind.FGSM:
                assert isinstance(self.cfg.attack_params, FGSMAttackParam), "Invalid params for FGSM"
                perturbed = fgsm.fgsm(data_ts_norm, target_ts, self.model, self.device, self.cfg.attack_params.epsilon, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std)
            else:
                raise ValueError(f"unsupported attack kind: {self.cfg.attack}")

            out2 = self.model(perturbed.tensor)
            logits2 = self._to_logits(out2)
            pred2 = logits2.max(1, keepdim=True)[1]

            denormalized_data = self.cfg.dataset_norm.denormalize(data_ts_norm)
            denormalized_perturbed = self.cfg.dataset_norm.denormalize(perturbed)
            
            perturbation_batch = denormalized_perturbed.tensor - denormalized_data.tensor
            l2_perturbations = torch.linalg.norm(perturbation_batch.view(current_batch_size, -1), ord=2, dim=1)

            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                    return
                
                perturbed_sample_denorm = TensorWithState(denormalized_perturbed.tensor[j].detach().cpu(), DENORMALIZED)
                
                yield (
                    global_idx, target_ts[j].item(), pred.view(-1)[j].item(), pred2.view(-1)[j].item(),
                    perturbed_sample_denorm, l2_perturbations[j].item()
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
    parser = argparse.ArgumentParser(description="general AE attack runner")
    # 必須パラメータ
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], required=True, help="Dataset to use.")
    parser.add_argument("--model", choices=[m.value for m in ModelKind], required=True, help="Model to use.")
    parser.add_argument("--attack", choices=[a.value for a in AttackKind], required=True, help="Attack method to use.")
    
    # デフォルト値を持つパラメータ
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help=f"Directory for model weights (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory to save attacked images (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of samples to process. -1 for all. (default: -1)")
    parser.add_argument('--save-images', action=argparse.BooleanOptionalAction, default=False, help='Whether to save attacked images (default: False).')
    parser.add_argument("--shuffle-dataloader", action="store_true", default=DEFAULT_SHUFFLE_DATALOADER, help=f"Shuffle the dataloader (default: {DEFAULT_SHUFFLE_DATALOADER}).")

    # 攻撃ごとの任意パラメータ
    parser.add_argument("--epsilon", type=fraction_float, default=None, help="Epsilon for attack.")
    parser.add_argument("--alpha", type=fraction_float, default=None, help="Alpha for iterative attacks.")
    parser.add_argument("--n", type=int, default=None, help="Number of iterations for iterative attacks.")
    
    args = parser.parse_args()

    # attack_kindに基づいてAttackParamsを生成・検証
    attack_kind = AttackKind(args.attack)
    attack_params: AttackParams

    if attack_kind in [AttackKind.BIM, AttackKind.FOOLBOX_BIM, AttackKind.LINF_BIM]:
        if args.epsilon is None or args.alpha is None or args.n is None:
            parser.error(f"{attack_kind.value} attack requires --epsilon, --alpha, and --n.")
        attack_params = BIMAttackParam(epsilon=args.epsilon, alpha=args.alpha, iters=args.n, batch_size=args.batch_size)
    
    elif attack_kind in [AttackKind.FGSM, AttackKind.FOOLBOX_FGSM]:
        if args.epsilon is None:
            parser.error(f"{attack_kind.value} attack requires --epsilon.")
        attack_params = FGSMAttackParam(epsilon=args.epsilon, batch_size=args.batch_size)

    else:
        raise NotImplementedError(f"Parameter validation for {attack_kind.value} is not implemented.")

    dataset = DatasetKind(args.dataset)
    device = utils.get_device()
    
    return Config(
        dataset=dataset,
        model=ModelKind(args.model),
        attack=attack_kind,
        attack_params=attack_params,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        save_attacked_images=args.save_images,
        shuffle_dataloader=args.shuffle_dataloader,
        device=device,
        dataset_norm=DatasetNorm(dataset, device),
    )

def main():
    cfg = parse_args()
    # parse_args内で検証されるため、ここでの検証は不要
    
    runner = Runner(cfg)
    result_generator = runner.run()

    fail, clean_acc_total, total, mean_mean_perturb = 0, 0, 0, 0.0

    # attack_paramsからepsilonを取得
    eps_str = f"{cfg.attack_params.epsilon:.3f}"
    
    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.attack.value, f"eps_{eps_str}")
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "attack_results.csv")
    
    with concurrent.futures.ThreadPoolExecutor() as executor, open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["index", "target_label", "prediction_before_attack", "prediction_after_attack", "l2_perturbation", "image_filepath"])

        for idx, target, before, after, ex, m in result_generator:
            total += 1
            if before == target:
                clean_acc_total += 1
                mean_mean_perturb += m
                if after != before:
                    fail += 1

            output_filename = ""
            if cfg.save_attacked_images:
                output_filename = os.path.join(output_folder, f"idx_{idx}_label_{after}.png")
                executor.submit(utils.save_tensor_as_image, ex.tensor, output_filename)
            
            csv_writer.writerow([idx, target, before, after, m, output_filename])

    attack_acc = (fail / clean_acc_total) if clean_acc_total > 0 else 0.0
    mean_perturb = (mean_mean_perturb / clean_acc_total) if clean_acc_total > 0 else 0.0

    def format_params(params: AttackParams) -> str:
        # dataclassを辞書に変換し、'batch_size'を除外
        params_dict = asdict(params)
        params_dict.pop('batch_size', None)
        # 値を整形して、見やすい文字列を生成
        return ", ".join([f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}" for k, v in params_dict.items()])

    attack_params_formatted = format_params(cfg.attack_params)

    print("\n=== Attack Summary ===")
    print("[実験設定]")
    print(f"  データセット: {cfg.dataset.value}")
    print(f"  モデル: {cfg.model.value}")
    print(f"  攻撃: {cfg.attack.value} ({attack_params_formatted})")
    
    print("\n[結果]")
    print(f"  処理サンプル総数: {total}")
    print(f"  クリーン画像を正しく分類したサンプル数: {clean_acc_total}")
    print(f"  攻撃成功率: {attack_acc:.4f} = {fail} / {clean_acc_total}")
    print("  攻撃成功率には，元の画像を正しく分類したサンプルのみを考慮している．")
    print(f"  平均摂動量(L2ノルム): {mean_perturb:.4f}")
    print(f"結果は {csv_filename} に保存されました。")
    print("======================")

if __name__ == "__main__":
    main()