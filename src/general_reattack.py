# 再攻撃を実行し、その結果を評価・保存する汎用コード
from dataclasses import dataclass, field, asdict
from torch.types import Number
from typing import Any, Final, Iterator, Optional, Tuple
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

@dataclass
class ReAttackResult:
    """再攻撃実験の1サンプル分の結果を保持するクラス"""
    index: int
    target_label: int
    pred_clean: int
    pred_attacked: int
    pred_reattacked: int
    l2_clean_vs_attacked: float
    l2_attacked_vs_reattacked: float
    l2_clean_vs_reattacked: float
    reattacked_image_ts: TensorWithState # 画像データ（保存用）


def attack(attack_kind: AttackKind, attack_params: AttackParams, input_data_norm: NormTensor, target_labels: Tensor, eval_model: nn.Module, device: torch.device, mean: Tensor, std: Tensor, config: Config) -> Tuple[utils.TensorWithState[bim.Normalized], Tensor]:
    if attack_kind == AttackKind.BIM:
        assert isinstance(attack_params, BIMAttackParam), "Invalid params for BIM"
        params = attack_params
        attacked_ts_norm  = bim.bim(
                input_data_norm, target_labels, eval_model,device, params.epsilon,
                params.alpha, params.iters, mean, std
            )
    elif attack_kind in [AttackKind.FOOLBOX_BIM, AttackKind.FOOLBOX_FGSM]:
        mean_list = mean.squeeze().tolist()
        std_list = std.squeeze().tolist()
        preprocessing = dict(mean=mean_list, std=std_list, axis=-3)
        bounds = (0.0, 1.0)
        fmodel = foolbox.PyTorchModel(eval_model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
        foolbox_attack = None
        if attack_kind == AttackKind.FOOLBOX_BIM:
            assert isinstance(attack_params, BIMAttackParam), "Invalid params for FoolboxBIM"
            foolbox_attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.attack_params.iters, abs_stepsize=self.cfg.attack_params.alpha) # type: ignore[reportPrivateImportUsage]
        elif attack_kind == AttackKind.FOOLBOX_FGSM:
            assert isinstance(attack_params, FGSMAttackParam), "Invalid params for FoolboxFGSM"
            foolbox_attack = foolbox.attacks.FGSM()
        else:
            raise ValueError(f"unsupported Foolbox attack kind: {attack_kind}")
        try:
            raw, clipped, is_adv = foolbox_attack(fmodel, config.dataset_norm.denormalize(input_data_norm).tensor, target_labels, epsilons=attack_params.epsilon)
            __perturbed_denorm = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
            perturbed_denorm_ts = TensorWithState(__perturbed_denorm, DENORMALIZED)
            attacked_ts_norm = config.dataset_norm.normalize(perturbed_denorm_ts)
        except Exception as e:
            print(f"Foolbox {attack_kind.value} attack failed: {e}")

    elif attack_kind == AttackKind.FGSM:
        assert isinstance(attack_params, FGSMAttackParam), "Invalid params for FGSM"
        attacked_ts_norm = fgsm.fgsm(input_data_norm, target_labels, eval_model, device, attack_params.epsilon, mean, std)
    else:
        raise ValueError(f"unsupported attack kind: {attack_kind}")
        # 攻撃コード終了
    attacked_ts_norm_f: Final[NormTensor] = attacked_ts_norm

        # 攻撃後推論
    out_attacked: Final[Tensor] = eval_model(attacked_ts_norm_f.tensor)
    logits_attacked: Final[Tensor] = _to_logits(out_attacked)

    # probs = F.softmax(logits_attacked, dim=1)
    # conf_max = probs.max(1)[0]
    # # ロジットの最大値、最小値、平均値を表示
    # # 入力テンソルの範囲を確認
    # print(f"Input Tensor Stats - Max: {input_data_norm.tensor.max().item():.4f}, Min: {input_data_norm.tensor.min().item():.4f}")
    # print(f"Attacked Tensor Stats - Max: {attacked_ts_norm_f.tensor.max().item():.4f}, Min: {attacked_ts_norm_f.tensor.min().item():.4f}")
    # print(f"Logits Raw Stats - Max: {logits_attacked.max().item():.4f}, Min: {logits_attacked.min().item():.4f}, Mean: {logits_attacked.mean().item():.4f}")
    # print(f"Logits Shape: {logits_attacked.shape}")
    # print(f"Confidence - Mean: {conf_max.mean().item():.4f}, Max: {conf_max.max().item():.4f}, Min: {conf_max.min().item():.4f}")

    # try:
    #     # 最後のレイヤーを取得（モデル構造に合わせて調整してください）
    #     last_layer = list(eval_model.modules())[-1] 
        
    #     if hasattr(last_layer, 'weight'):
    #         weights = last_layer.weight.data
    #         print(f"Last Layer Weights - Max: {weights.max().item():.4f}, Mean: {weights.mean().item():.4f}, Std: {weights.std().item():.4f}")
    #     else:
    #         print("最後のレイヤーにweight属性がありません。")
    # except Exception as e:
    #     print(f"重みの確認中にエラー: {e}")
    # # フックを使って最終層への入力とバイアスを確認する
    # def inspect_final_layer(module, input, output):
    #     # inputはタプルなので最初の要素を取得
    #     features = input[0]
    #     print("--- Final Layer Inspection ---")
    #     print(f"Features (Input to FC) - Max: {features.max().item():.4f}, Mean: {features.mean().item():.4f}")
        
    #     if module.bias is not None:
    #         print(f"Bias Values - Max: {module.bias.max().item():.4f}, Mean: {module.bias.mean().item():.4f}")
    #     else:
    #         print("Bias: None")
    #     print("------------------------------")

    # 最終層（fcやclassifierなど）にフックを登録
    # モデルの構造に合わせて layer_name を調整してください
    # ResNetなどは 'fc', VGGなどは 'classifier' のことが多いです
    # target_layer = None
    # for name, module in eval_model.named_modules():
    #     # 最後の線形層を自動で見つける簡易ロジック
    #     if isinstance(module, torch.nn.Linear):
    #         target_layer = module
    #         # 最後の層を取得したいのでループを回し続ける（上書きしていく）

    # if target_layer is not None:
    #     handle = target_layer.register_forward_hook(inspect_final_layer)
        
    #     # 一度推論を走らせる（これでフックが呼ばれます）
    #     _ = eval_model(attacked_ts_norm_f.tensor)
        
    #     handle.remove() # フック解除
    # else:
    #     print("Linear Layerが見つかりませんでした。")


    pred_attacked: Final[Tensor] = logits_attacked.max(1, keepdim=True)[1]
    return attacked_ts_norm_f, pred_attacked


@staticmethod
def _to_logits(output: Tensor) -> Tensor:
    # print(f"Output Tensor Shape: {output.shape}, Dim: {output.dim()}")
    if output.dim() == 4:
        return output.mean(dim=(2, 3))
    # 次元は2がやってきていることを確認した．
    if output.dim() > 2:
        return output.view(output.size(0), output.size(1), -1).mean(dim=2)
    return output


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

        if cfg.attack_kind in [AttackKind.FOOLBOX_BIM, AttackKind.FOOLBOX_FGSM] or cfg.reattack_kind in [AttackKind.FOOLBOX_BIM, AttackKind.FOOLBOX_FGSM]:
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

        if cfg.attack_kind == AttackKind.FOOLBOX_FGSM:
            assert isinstance(cfg.attack_params, FGSMAttackParam), "Invalid attack_params for FoolboxFGSM"
            self.attack_obj = foolbox.attacks.FGSM() # type: ignore[reportPrivateImportUsage]
        if cfg.reattack_kind == AttackKind.FOOLBOX_FGSM:
            assert isinstance(cfg.reattack_params, FGSMAttackParam), "Invalid reattack_params for FoolboxFGSM"
            self.reattack_obj = foolbox.attacks.FGSM() # type: ignore[reportPrivateImportUsage]


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

    def run(self) -> Iterator[ReAttackResult]:
        print(f"Running Re-Attack with config: {self.cfg}")

        # model = self.model.eval()

        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Re-Attacking"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                break
            
            current_batch_size, target_ts, pred_clean, pred_attacked, pred_reattacked, reattacked_data_denorm, l2_clean_vs_attacked, l2_attacked_vs_reattacked, l2_clean_vs_reattacked = self.inner_loop(data, target)

            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                    return

                reattacked_sample_denorm_ts = TensorWithState(reattacked_data_denorm[j].detach().cpu(), DENORMALIZED)

                # print(f"Index {global_idx}: Target {target_ts[j].item()}, Clean Pred {pred_clean.view(-1)[j].item()}, Attacked Pred {pred_attacked.view(-1)[j].item()}, Reattacked Pred {pred_reattacked.view(-1)[j].item()}")

                yield ReAttackResult(
                    index=global_idx,
                    target_label=target_ts[j].item().as_integer_ratio()[0],
                    pred_clean=pred_clean.view(-1)[j].item().as_integer_ratio()[0],
                    pred_attacked=pred_attacked.view(-1)[j].item().as_integer_ratio()[0],
                    pred_reattacked=pred_reattacked.view(-1)[j].item().as_integer_ratio()[0],
                    l2_clean_vs_attacked=l2_clean_vs_attacked[j].item(),
                    l2_attacked_vs_reattacked=l2_attacked_vs_reattacked[j].item(),
                    l2_clean_vs_reattacked=l2_clean_vs_reattacked[j].item(),
                    reattacked_image_ts=reattacked_sample_denorm_ts
                )
                global_idx += 1

    def inner_loop(self, data, target):
        eval_model = self.model.eval()
        current_batch_size = data.shape[0]
        clean_data_ts: Final[DenormTensor] = TensorWithState(data.to(self.device), DENORMALIZED)
        clean_data_ts_norm: Final[NormTensor] = self.cfg.dataset_norm.normalize(clean_data_ts)
        target_ts: Final[Tensor] = target.to(self.device).view(-1).long()
            
        # クリーンデータ推論
        output_clean: Final[Tensor] = eval_model(clean_data_ts_norm.tensor)
        # logits_clean: Final[Tensor] = _to_logits(output_clean)
        # pred_clean: Final[Tensor] = logits_clean.max(1, keepdim=True)[1]
        # print(f"clean Logits Raw Stats - Max: {logits_clean.max().item():.4f}, Min: {logits_clean.min().item():.4f}, Mean: {logits_clean.mean().item():.4f}")
        # print(f"clean Logits Shape: {logits_clean.shape}")
        pred_clean: Final[Tensor] = output_clean.max(1, keepdim=True)[1]
        # print(f"clean Logits Raw Stats - Max: {output_clean.max().item():.4f}, Min: {output_clean.min().item():.4f}, Mean: {output_clean.mean().item():.4f}")
        # print(f"clean Logits Shape: {output_clean.shape}")
        


            # _perform_attackは常に正規化データを返すので、データの状態管理がシンプルになる
            # attacked_data_ts_norm = self._perform_attack(clean_data_ts_norm, target_ts, self.cfg.attack_kind, self.cfg.attack_params)
            # 攻撃コード開始

        result = attack(
            self.cfg.attack_kind,
            self.cfg.attack_params,
            clean_data_ts_norm,
            target_ts,
            eval_model,
            self.device,
            self.cfg.dataset_norm.mean,
            self.cfg.dataset_norm.std,
            self.cfg
        )
        attacked_ts_norm, pred_attacked = result
        attacked_ts_norm_f: Final[NormTensor] = attacked_ts_norm




        reattack_target: Final[Tensor] = pred_attacked.view(-1).long()
            # reattacked_data_ts_norm = self._perform_attack(attacked_data_ts_norm, reattack_target, self.cfg.reattack_kind, self.cfg.reattack_params)
            # 再攻撃コード開始

        result_reattack = attack(
            self.cfg.reattack_kind,
            self.cfg.reattack_params,
            attacked_ts_norm_f,
            reattack_target,
            eval_model,
            self.device,
            self.cfg.dataset_norm.mean,
            self.cfg.dataset_norm.std,
            self.cfg
        )
        reattacked_ts_norm, pred_reattacked = result_reattack
        reattacked_ts_norm_f: Final[NormTensor] = reattacked_ts_norm
            

            # --- 以降、評価と結果のyield ---
        clean_data_denorm = self.cfg.dataset_norm.denormalize(clean_data_ts_norm).tensor
        attacked_data_denorm = self.cfg.dataset_norm.denormalize(attacked_ts_norm).tensor
        reattacked_data_denorm = self.cfg.dataset_norm.denormalize(reattacked_ts_norm).tensor

        l2_clean_vs_attacked = torch.linalg.norm((attacked_data_denorm - clean_data_denorm).view(current_batch_size, -1), ord=2, dim=1)
        l2_attacked_vs_reattacked = torch.linalg.norm((reattacked_data_denorm - attacked_data_denorm).view(current_batch_size, -1), ord=2, dim=1)
        l2_clean_vs_reattacked = torch.linalg.norm((reattacked_data_denorm - clean_data_denorm).view(current_batch_size, -1), ord=2, dim=1)
        return current_batch_size,target_ts,pred_clean,pred_attacked,pred_reattacked,reattacked_data_denorm,l2_clean_vs_attacked,l2_attacked_vs_reattacked,l2_clean_vs_reattacked

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
    if attack_kind in [AttackKind.FGSM, AttackKind.FOOLBOX_FGSM]:
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
    if reattack_kind in [AttackKind.FGSM, AttackKind.FOOLBOX_FGSM]:
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


@dataclass
class ExperimentStats:
    """実験全体の統計情報を集計するクラス"""
    total: int = 0
    clean_correct: int = 0
    attack_successful: int = 0
    reattack_successful: int = 0
    reattack_successful_to_clean: int = 0
    reattack_correct_to_original: int = 0

def main():
    cfg = parse_args()
    # parse_args内で検証されるため、ここでの検証は不要
    
    runner = Runner(cfg)
    result_generator = runner.run()

    # ----- 統計情報の初期化とパス生成 -----
    stats = {'total': 0, 'clean_correct': 0, 'attack_successful': 0, 'reattack_successful': 0, 'reattack_successful_to_clean': 0, 'reattack_correct_to_original': 0}
    
    # epsilonをattack_paramsから取得
    attack_eps_str = f"{cfg.attack_params.epsilon:.3f}"
    reattack_eps_str = f"{cfg.reattack_params.epsilon:.3f}"

    attack_params_str = f"{cfg.attack_kind.value}_eps{attack_eps_str}"
    reattack_params_str = f"{cfg.reattack_kind.value}_eps{reattack_eps_str}"
    
    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.model.value, attack_params_str, reattack_params_str)
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "0_reattack_results.csv")
    csv_header = [
        "index", "target_label", "pred_clean", "pred_attacked", "pred_reattacked",
        "l2_clean_vs_attacked", "l2_attacked_vs_reattacked", "l2_clean_vs_reattacked", "reattacked_image_path"
    ]
    
    # パフォーマンス改善: ThreadPoolExecutorとCSV writerをループの外で初期化
    with concurrent.futures.ThreadPoolExecutor() as executor, open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        
        futures = {}

        stats = ExperimentStats()
        for result_tuple in result_generator:
            idx = result_tuple.index
            target = result_tuple.target_label
            pred_c = result_tuple.pred_clean
            pred_a = result_tuple.pred_attacked
            pred_r = result_tuple.pred_reattacked
            l2_c_a = result_tuple.l2_clean_vs_attacked
            l2_a_r = result_tuple.l2_attacked_vs_reattacked
            l2_c_r = result_tuple.l2_clean_vs_reattacked
            image_ts = result_tuple.reattacked_image_ts 

            stats.total += 1

            # 新しい指標: 再攻撃後に元の正解ラベルに一致したか
            if pred_r == target:
                stats.reattack_correct_to_original += 1

            # 既存の指標
            if pred_c == target:
                stats.clean_correct += 1
                if pred_a != pred_c:
                    stats.attack_successful += 1
                    # 再攻撃が成功し、元の予測から変わった場合 (ターゲット化された再攻撃の成功)
                    if pred_r != pred_a:
                         stats.reattack_successful += 1
                    # 再攻撃が成功し、正しいラベルに戻った場合
                    if pred_r == target:
                        stats.reattack_successful_to_clean += 1

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


    
    def format_params(params: AttackParams) -> str:
        # dataclassを辞書に変換し、'batch_size'を除外
        params_dict = asdict(params)
        params_dict.pop('batch_size', None)
        # 値を整形して、見やすい文字列を生成
        return ", ".join([f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}" for k, v in params_dict.items()])

    attack_params_formatted = format_params(cfg.attack_params)
    reattack_params_formatted = format_params(cfg.reattack_params)

    print("\n=== 再攻撃概要 ===")
    print("[実験設定]")
    print(f"  データセット: {cfg.dataset.value}")
    print(f"  モデル: {cfg.model.value}")
    print(f"  初回攻撃: {cfg.attack_kind.value} ({attack_params_formatted})")
    print(f"  再攻撃: {cfg.reattack_kind.value} ({reattack_params_formatted})")
    
    print("\n[結果]")
    print(f"  処理サンプル総数: {stats.total}")
    print(f"  クリーン精度: {stats.clean_correct / stats.total:.4f} ({stats.clean_correct}/{stats.total})")
    if stats.clean_correct > 0:
        print(f"  初回攻撃成功率: {stats.attack_successful / stats.clean_correct:.4f} ({stats.attack_successful}/{stats.clean_correct})")
        if stats.attack_successful > 0:
            print(f"  再攻撃成功率 (攻撃後から変化): {stats.reattack_successful / stats.attack_successful:.4f} ({stats.reattack_successful}/{stats.attack_successful})")
            print(f"  再攻撃成功率 (正しい識別に戻ったもの): {stats.reattack_successful_to_clean / stats.attack_successful:.4f} ({stats.reattack_successful_to_clean}/{stats.attack_successful})")
    print(f"  再攻撃後正解率(クリーン識別成功，攻撃成功を考慮しない): {stats.reattack_correct_to_original / stats.total:.4f} ({stats.reattack_correct_to_original}/{stats.total})")
    print(f"結果は {csv_filename} に保存されました。")
    print("=========================")


if __name__ == "__main__":
    main()
