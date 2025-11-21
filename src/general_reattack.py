# 再攻撃を実行し、その結果を評価・保存する汎用コード
from dataclasses import dataclass, field
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
import sys # parser.errorを使うため
from PIL import Image
from torch import Tensor
from enum import Enum
import argparse
import re_attack_0806
from copy import deepcopy
import concurrent.futures # パフォーマンス改善のために追加

from re_attack_0806 import attacks, utils
from re_attack_0806.attacks import bim, fgsm
# AttackParams関連をインポート
from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm, FGSMAttackParam, BIMAttackParam, AttackParams
from re_attack_0806.utils.normTensor import *

import foolbox
from tqdm import tqdm

# --- 定数定義 ---
DEFAULT_MODEL_DIR = "./weight"
DEFAULT_OUTPUT_DIR = "reattacked_data"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_SAMPLES = -1
DEFAULT_SHUFFLE_DATALOADER = False # 新しく追加


@dataclass
class Config:
    # 必須パラメータ
    dataset: DatasetKind
    model: ModelKind
    attack_kind: AttackKind
    attack_params: AttackParams
    reattack_kind: AttackKind
    reattack_params: AttackParams
    
    # デフォルト値を持つパラメータ
    model_dir: str
    batch_size: int
    output_dir: str
    num_samples: int
    save_images: bool
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
        self.model.eval() # 評価モードに設定

        # パフォーマンス改善: foolbox関連オブジェクトを一度だけ初期化
        self.fmodel = None
        self.attack_obj = None
        self.reattack_obj = None

        if cfg.attack_kind == AttackKind.FOOLBOX_BIM or cfg.reattack_kind == AttackKind.FOOLBOX_BIM:
            mean_list = self.cfg.dataset_norm.mean.squeeze().tolist()
            std_list = self.cfg.dataset_norm.std.squeeze().tolist()
            preprocessing = dict(mean=mean_list, std=std_list, axis=-3)
            # type: ignore[reportPrivateImportUsage] はfoolboxのPyTorchModelがプライベートな型を使用していることによる警告を抑制
            self.fmodel = foolbox.PyTorchModel(self.model, bounds=(0.0, 1.0), preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]

        if cfg.attack_kind == AttackKind.FOOLBOX_BIM:
            assert isinstance(cfg.attack_params, BIMAttackParam), "Invalid attack_params for FoolboxBIM"
            self.attack_obj = foolbox.attacks.LinfBasicIterativeAttack(steps=cfg.attack_params.iters, abs_stepsize=cfg.attack_params.alpha) # type: ignore[reportPrivateImportUsage]
        
        if cfg.reattack_kind == AttackKind.FOOLBOX_BIM:
            assert isinstance(cfg.reattack_params, BIMAttackParam), "Invalid reattack_params for FoolboxBIM"
            self.reattack_obj = foolbox.attacks.LinfBasicIterativeAttack(steps=cfg.reattack_params.iters, abs_stepsize=cfg.reattack_params.alpha) # type: ignore[reportPrivateImportUsage]


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

    def _perform_attack(self, data_norm: TensorWithState, target: Tensor, kind: AttackKind, params: AttackParams) -> TensorWithState:
        # このメソッドは常に正規化済みのデータを入力として受け取り、正規化済みのデータを返す
        if kind == AttackKind.BIM:
            assert isinstance(params, BIMAttackParam), "Invalid params for BIM"
            return bim.bim(data_norm, target, self.model, self.device, params.epsilon, params.alpha, params.iters, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std)
        
        elif kind == AttackKind.FGSM:
            assert isinstance(params, FGSMAttackParam), "Invalid params for FGSM"
            eps_norm = params.epsilon / self.cfg.dataset_norm.std.mean() # stdが複数の値を持つ場合を考慮
            return fgsm.fgsm(data_norm, eps_norm, target, self.model, self.device)
            
        elif kind == AttackKind.FOOLBOX_BIM:
            assert isinstance(params, BIMAttackParam), "Invalid params for FoolboxBIM"
            
            # attack_objまたはreattack_objを適切に選択
            if kind == self.cfg.attack_kind:
                attack_obj = self.attack_obj
            elif kind == self.cfg.reattack_kind:
                attack_obj = self.reattack_obj
            else:
                raise ValueError("Attack kind not recognized for foolbox object selection.")

            if self.fmodel is None or attack_obj is None:
                raise RuntimeError("Foolbox model or attack object is not initialized.")

            # バグ修正: foolboxには非正規化データを渡す
            data_denorm = self.cfg.dataset_norm.denormalize(data_norm)
            
            try:
                _, clipped, _ = attack_obj(self.fmodel, data_denorm.tensor, target, epsilons=params.epsilon)
                
                # バグ修正: foolboxの出力(非正規化)を後続処理のため正規化する
                __perturbed_denorm = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
                perturbed_denorm_ts = TensorWithState(__perturbed_denorm, DENORMALIZED)
                return self.cfg.dataset_norm.normalize(perturbed_denorm_ts)

            except Exception as e:
                print(f"Foolbox BIM attack failed: {e}. Returning original data.")
                return data_norm # 失敗した場合は元の正規化データを返す
        else:
            raise ValueError(f"unsupported attack kind: {kind}")

    def run(self) -> Iterator[Tuple]:
        print(f"Running Re-Attack with config: {self.cfg}")

        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Re-Attacking"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                break
            
            current_batch_size = data.shape[0]
            clean_data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
            clean_data_ts_norm = self.cfg.dataset_norm.normalize(clean_data_ts)
            target_ts: Tensor = target.to(self.device).view(-1).long()
            
            logits_clean = self._to_logits(self.model(clean_data_ts_norm.tensor))
            pred_clean = logits_clean.max(1, keepdim=True)[1]

            # _perform_attackは常に正規化データを返すので、データの状態管理がシンプルになる
            attacked_data_ts_norm = self._perform_attack(clean_data_ts_norm, target_ts, self.cfg.attack_kind, self.cfg.attack_params)
            logits_attacked = self._to_logits(self.model(attacked_data_ts_norm.tensor))
            pred_attacked = logits_attacked.max(1, keepdim=True)[1]

            reattack_target = pred_attacked.view(-1)
            reattacked_data_ts_norm = self._perform_attack(attacked_data_ts_norm, reattack_target, self.cfg.reattack_kind, self.cfg.reattack_params)
            logits_reattacked = self._to_logits(self.model(reattacked_data_ts_norm.tensor))
            pred_reattacked = logits_reattacked.max(1, keepdim=True)[1]

            # --- 以降、評価と結果のyield ---
            clean_data_denorm = self.cfg.dataset_norm.denormalize(clean_data_ts_norm).tensor
            attacked_data_denorm = self.cfg.dataset_norm.denormalize(attacked_data_ts_norm).tensor
            reattacked_data_denorm = self.cfg.dataset_norm.denormalize(reattacked_data_ts_norm).tensor

            l2_clean_vs_attacked = torch.linalg.norm((clean_data_denorm - attacked_data_denorm).view(current_batch_size, -1), ord=2, dim=1)
            l2_attacked_vs_reattacked = torch.linalg.norm((attacked_data_denorm - reattacked_data_denorm).view(current_batch_size, -1), ord=2, dim=1)
            l2_clean_vs_reattacked = torch.linalg.norm((clean_data_denorm - reattacked_data_denorm).view(current_batch_size, -1), ord=2, dim=1)

            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                    return

                reattacked_sample_denorm_ts = TensorWithState(reattacked_data_denorm[j].detach().cpu(), DENORMALIZED)

                yield (
                    global_idx, target_ts[j].item(), pred_clean.view(-1)[j].item(), pred_attacked.view(-1)[j].item(),
                    pred_reattacked.view(-1)[j].item(), l2_clean_vs_attacked[j].item(), l2_attacked_vs_reattacked[j].item(),
                    l2_clean_vs_reattacked[j].item(), reattacked_sample_denorm_ts
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
    # 必須パラメータ
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], required=True, help="Dataset to use.")
    parser.add_argument("--model", choices=[m.value for m in ModelKind], required=True, help="Model to use.")
    parser.add_argument("--attack-kind", choices=[k.value for k in AttackKind], required=True, help="Initial attack method to use.")
    parser.add_argument("--reattack-kind", choices=[k.value for k in AttackKind], required=True, help="Re-attack method to use.")

    # デフォルト値を持つパラメータ
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help=f"Directory for model weights (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory to save reattacked images (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of samples to process. -1 for all. (default: -1)")
    parser.add_argument('--save-images', action=argparse.BooleanOptionalAction, default=True, help='Save attacked images (default). Use --no-save-images to disable.')
    parser.add_argument("--shuffle-dataloader", action="store_true", default=DEFAULT_SHUFFLE_DATALOADER, help=f"Shuffle the dataloader (default: {DEFAULT_SHUFFLE_DATALOADER}).") # ここに追加

    # Attack Configs (必須パラメータはここでNoneとし、後に検証)
    parser.add_argument("--attack-eps", type=fraction_float, default=None, help="Epsilon for initial attack.")
    parser.add_argument("--attack-alpha", type=fraction_float, default=None, help="Alpha for iterative initial attacks.")
    parser.add_argument("--attack-n", type=int, default=None, help="Number of iterations for iterative initial attacks.")
    
    # Re-Attack Configs (必須パラメータはここでNoneとし、後に検証)
    parser.add_argument("--reattack-eps", type=fraction_float, default=None, help="Epsilon for re-attack.")
    parser.add_argument("--reattack-alpha", type=fraction_float, default=None, help="Alpha for iterative re-attacks.")
    parser.add_argument("--reattack-n", type=int, default=None, help="Number of iterations for iterative re-attacks.")

    args = parser.parse_args()

    # Create AttackParams
    attack_kind = AttackKind(args.attack_kind)
    attack_params: AttackParams
    if attack_kind == AttackKind.FGSM:
        if args.attack_eps is None: parser.error("FGSM initial attack requires --attack-eps.")
        attack_params = FGSMAttackParam(epsilon=args.attack_eps, batch_size=args.batch_size)
    elif attack_kind in [AttackKind.BIM, AttackKind.FOOLBOX_BIM]:
        if args.attack_eps is None or args.attack_alpha is None or args.attack_n is None:
            parser.error(f"{attack_kind.value} initial attack requires --attack-eps, --attack-alpha, and --attack-n.")
        attack_params = BIMAttackParam(epsilon=args.attack_eps, alpha=args.attack_alpha, iters=args.attack_n, batch_size=args.batch_size)
    else:
        raise ValueError(f"Unsupported initial attack kind for params: {attack_kind}")

    # Create Re-attackParams
    reattack_kind = AttackKind(args.reattack_kind)
    reattack_params: AttackParams
    if reattack_kind == AttackKind.FGSM:
        if args.reattack_eps is None: parser.error("FGSM re-attack requires --reattack-eps.")
        reattack_params = FGSMAttackParam(epsilon=args.reattack_eps, batch_size=args.batch_size)
    elif reattack_kind in [AttackKind.BIM, AttackKind.FOOLBOX_BIM]:
        if args.reattack_eps is None or args.reattack_alpha is None or args.reattack_n is None:
            parser.error(f"{reattack_kind.value} re-attack requires --reattack-eps, --reattack-alpha, and --reattack-n.")
        reattack_params = BIMAttackParam(epsilon=args.reattack_eps, alpha=args.reattack_alpha, iters=args.reattack_n, batch_size=args.batch_size)
    else:
        raise ValueError(f"Unsupported re-attack kind for params: {reattack_kind}")

    dataset = DatasetKind(args.dataset)
    device = utils.get_device()

    return Config(
        dataset=dataset,
        model=ModelKind(args.model),
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        save_images=args.save_images,
        shuffle_dataloader=args.shuffle_dataloader,
        device=device,
        dataset_norm=DatasetNorm(dataset, device),
        attack_kind=attack_kind,
        attack_params=attack_params,
        reattack_kind=reattack_kind,
        reattack_params=reattack_params
    )

def main():
    cfg = parse_args()
    # parse_args内で検証されるため、ここでの検証は不要
    
    runner = Runner(cfg)
    result_generator = runner.run()

    # ----- 統計情報の初期化とパス生成 -----
    stats = {'total': 0, 'clean_correct': 0, 'attack_successful': 0, 'reattack_successful': 0, 'reattack_successful_to_clean': 0}
    
    # epsilonをattack_paramsから取得
    attack_eps_str = f"{cfg.attack_params.epsilon:.3f}"
    reattack_eps_str = f"{cfg.reattack_params.epsilon:.3f}"

    attack_params_str = f"{cfg.attack_kind.value}_eps{attack_eps_str}"
    reattack_params_str = f"{cfg.reattack_kind.value}_eps{reattack_eps_str}"
    
    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.model.value, attack_params_str, reattack_params_str)
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "reattack_results.csv")
    csv_header = [
        "index", "target_label", "pred_clean", "pred_attacked", "pred_reattacked",
        "l2_clean_vs_attacked", "l2_attacked_vs_reattacked", "l2_clean_vs_reattacked", "reattacked_image_path"
    ]
    
    # パフォーマンス改善: ThreadPoolExecutorとCSV writerをループの外で初期化
    with concurrent.futures.ThreadPoolExecutor() as executor, open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        
        futures = {}

        for result_tuple in result_generator:
            (idx, target, pred_c, pred_a, pred_r, l2_c_a, l2_a_r, l2_c_r, image_ts) = result_tuple
            
            stats['total'] += 1
            if pred_c == target:
                stats['clean_correct'] += 1
                if pred_a != pred_c:
                    stats['attack_successful'] += 1
                    # 再攻撃が成功し、元の予測から変わった場合 (ターゲット化された再攻撃の成功)
                    if pred_r != pred_a:
                         stats['reattack_successful'] += 1
                    # 再攻撃が成功し、元のクリーンな予測に戻った場合
                    if pred_r == pred_c:
                        stats['reattack_successful_to_clean'] += 1

            image_path = ""
            if cfg.save_images:
                image_path = os.path.join(output_folder, f"idx_{idx}_target{target}_r{pred_r}.png")
                # 画像保存を非同期で実行
                future = executor.submit(utils.save_tensor_as_image, image_ts.tensor, image_path)
                futures[future] = idx

            csv_row = [idx, target, pred_c, pred_a, pred_r, l2_c_a, l2_a_r, l2_c_r, image_path]
            writer.writerow(csv_row)

        # 非同期処理の完了を待ち、エラーがあれば表示
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Image saving for index {idx} generated an exception: {exc}")


    total, clean_correct, attack_successful, reattack_successful, reattack_to_clean = \
        stats['total'], stats['clean_correct'], stats['attack_successful'], stats['reattack_successful'], stats['reattack_successful_to_clean']
    
    print("\n=== 再攻撃概要 ===")
    print(f"処理サンプル総数: {total}")
    print(f"クリーン精度: {clean_correct / total:.4f} ({clean_correct}/{total})")
    if clean_correct > 0:
        print(f"初回攻撃成功率: {attack_successful / clean_correct:.4f} ({attack_successful}/{clean_correct})")
        if attack_successful > 0:
            print(f"再攻撃成功率 (攻撃後から変化): {reattack_successful / attack_successful:.4f} ({reattack_successful}/{attack_successful})")
            print(f"再攻撃成功率 (クリーンに戻ったもの): {reattack_to_clean / attack_successful:.4f} ({reattack_to_clean}/{attack_successful})")
    print(f"結果は {csv_filename} に保存されました。")
    print("=========================")


if __name__ == "__main__":
    main()
