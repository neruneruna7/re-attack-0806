# src/ensemble_branch_reattack.py

from dataclasses import dataclass, field, asdict
from typing import Final, Iterator, Tuple, List, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import csv
import argparse
import concurrent.futures
import collections
from tqdm import tqdm
from torch import Tensor

# 既存モジュールのインポート (パスが通っている前提)
from re_attack_0806 import utils
from re_attack_0806.attacks import bim
from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm, BIMAttackParam, AttackParams
from re_attack_0806.utils.normTensor import *

# --- 定数定義 ---
DEFAULT_MODEL_DIR = "./weight"
DEFAULT_OUTPUT_DIR = "ensemble_branch_reattacked_data"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_SAMPLES = -1
DEFAULT_SHUFFLE_DATALOADER = False


@dataclass
class Config:
    dataset: DatasetKind
    model: ModelKind
    attack_params: BIMAttackParam   # BIM固定
    reattack_params: BIMAttackParam # BIM固定
    
    model_dir: str
    batch_size: int
    output_dir: str
    num_samples: int
    save_images: bool
    shuffle_dataloader: bool
    
    device: torch.device
    dataset_norm: DatasetNorm

@dataclass
class ReAttackResult:
    index: int
    target_label: int
    pred_clean: int
    pred_attacked: int
    pred_ensemble: int  # アンサンブルによる最終予測
    
    # 指標
    attack_prob_correct: float
    attack_rank_correct: int
    ensemble_prob_correct: float
    ensemble_rank_correct: int
    
    # 保存用画像 (代表としてPlainルートの再攻撃画像などを保持してもよいが、ここでは省略またはPlainを採用)
    reattacked_image_ts: TensorWithState

class DefenseFilter:
    """PyTorch Tensor用の防御フィルタ群"""
    def __init__(self, device: torch.device):
        self.device = device

    def gaussian_blur(self, x: Tensor, kernel_size=3, sigma=1.0) -> Tensor:
        if kernel_size % 2 == 0: kernel_size += 1
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return transform(x)

    def median_filter(self, x: Tensor, kernel_size=3) -> Tensor:
        if kernel_size % 2 == 0: kernel_size += 1
        padding = kernel_size // 2
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        x_unfolded = F.unfold(x_reshaped, kernel_size, padding=padding)
        x_median = x_unfolded.median(dim=1).values
        x_out = x_median.view(b, c, h, w)
        return x_out

    def resize_restore(self, x: Tensor, scale_factor=0.9) -> Tensor:
        _, _, h, w = x.shape
        h_small, w_small = int(h * scale_factor), int(w * scale_factor)
        x_small = F.interpolate(x, size=(h_small, w_small), mode='bilinear', align_corners=False)
        x_restored = F.interpolate(x_small, size=(h, w), mode='bilinear', align_corners=False)
        return x_restored


# 攻撃実行関数 (BIM専用)
def run_bim_attack(attack_params: BIMAttackParam, input_data_norm: NormTensor, target_labels: Tensor, eval_model: nn.Module, device: torch.device, mean: Tensor, std: Tensor) -> Tuple[utils.TensorWithState[bim.Normalized], Tensor, Tensor]:
    
    attacked_ts_norm = bim.bim(
        input_data_norm, target_labels, eval_model, device, 
        attack_params.epsilon, attack_params.alpha, attack_params.iters, 
        mean, std
    )
    
    attacked_ts_norm_f: Final[NormTensor] = attacked_ts_norm
    out_attacked: Final[Tensor] = eval_model(attacked_ts_norm_f.tensor)
    logits_attacked: Final[Tensor] = _to_logits(out_attacked)
    pred_attacked: Final[Tensor] = logits_attacked.max(1, keepdim=True)[1]
    
    return attacked_ts_norm_f, pred_attacked, logits_attacked


@staticmethod
def _to_logits(output: Tensor) -> Tensor:
    if output.dim() == 4:
        return output.mean(dim=(2, 3))
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
        self.model.eval()
        self.filters = DefenseFilter(self.device)

    def _load_weights_if_exists(self, model_dir: str):
            model_name = getattr(self.model, "model_name", None)
            if model_name is None: return
            path = os.path.join(model_dir, f"{model_name}.ckpt")
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    st = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                    self.model.load_state_dict(st)
                    print(f"loaded weights from {path}")
                except Exception as e:
                    print(f"failed to load state_dict from {path}: {e}")

    def run(self) -> Iterator[ReAttackResult]:
        print(f"Running Branched Re-Attack Ensemble with config: {self.cfg}")
        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Branched Re-Attacking"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples: break
            
            results = self.inner_loop(data, target)
            
            (current_batch_size, target_ts, pred_clean, pred_attacked, pred_ensemble,
             prob_a, rank_a, prob_e, rank_e, sample_reattacked_denorm) = results

            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples: return

                reattacked_sample_ts = TensorWithState(sample_reattacked_denorm[j].detach().cpu(), DENORMALIZED)

                yield ReAttackResult(
                    index=global_idx,
                    target_label=target_ts[j].item(),
                    pred_clean=pred_clean.view(-1)[j].item(),
                    pred_attacked=pred_attacked.view(-1)[j].item(),
                    pred_ensemble=pred_ensemble.view(-1)[j].item(),
                    
                    attack_prob_correct=prob_a[j].item(),
                    attack_rank_correct=rank_a[j].item(),
                    ensemble_prob_correct=prob_e[j].item(),
                    ensemble_rank_correct=rank_e[j].item(),
                    
                    reattacked_image_ts=reattacked_sample_ts
                )
                global_idx += 1

    def inner_loop(self, data, target):
        eval_model = self.model.eval()
        current_batch_size = data.shape[0]
        
        # データ準備
        clean_data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
        clean_data_ts_norm = self.cfg.dataset_norm.normalize(clean_data_ts)
        target_ts = target.to(self.device).view(-1).long()
        target_indices = target_ts.view(-1, 1)

        # 1. クリーンデータ推論
        out_clean = eval_model(clean_data_ts_norm.tensor)
        pred_clean = out_clean.max(1, keepdim=True)[1]
        
        # 2. 攻撃 (BIM Attack) - 全ブランチ共通の出発点
        res_attack = run_bim_attack(
            self.cfg.attack_params, clean_data_ts_norm, target_ts, 
            eval_model, self.device, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std
        )
        attacked_ts_norm, pred_attacked, logits_attacked = res_attack
        
        # 攻撃後指標
        probs_attacked = F.softmax(logits_attacked, dim=1)
        prob_correct_attacked = probs_attacked.gather(1, target_indices).view(-1)
        rank_correct_attacked = (logits_attacked > logits_attacked.gather(1, target_indices)).sum(dim=1) + 1

        # 3. 分岐処理 & 再攻撃
        # 各ブランチの定義: (名前, フィルタ関数)
        branches: List[Tuple[str, Callable[[Tensor], Tensor]]] = [
            ("Plain", lambda x: x), # そのまま
            ("Gaussian", lambda x: self.filters.gaussian_blur(x, kernel_size=3, sigma=1.0)),
            ("Median", lambda x: self.filters.median_filter(x, kernel_size=3)),
            ("Resize", lambda x: self.filters.resize_restore(x, scale_factor=0.9))
        ]

        branch_logits_list = []
        plain_reattacked_denorm = None # 画像保存用にPlainルートの結果を保持

        # 再攻撃のターゲットは「攻撃直後の予測ラベル」とする (自己修正の試み)
        reattack_target = pred_attacked.view(-1).long()

        for name, filter_fn in branches:
            # A. 前処理 (Branch Preprocessing)
            # 正規化された状態のままフィルタを適用
            # ※注意: 正規化テンソルに対する空間フィルタ適用は、線形処理(Gaussian, Resize)なら概ね問題ないが
            # 非線形(Median)などは画素値の分布に依存するため、厳密にはDenorm->Filter->Normが正しい場合もある。
            # ここでは計算効率とこれまでの実装との整合性を重視し、NormTensorへの直接適用を行う。
            x_filtered = filter_fn(attacked_ts_norm.tensor)
            x_filtered_state = TensorWithState(x_filtered, NORMALIZED)

            # B. 再攻撃 (Re-Attack)
            # 前処理された状態からスタートして、reattack_targetに向かって最適化(復元)を試みる
            res_reattack = run_bim_attack(
                self.cfg.reattack_params, x_filtered_state, reattack_target, 
                eval_model, self.device, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std
            )
            reattacked_ts_norm_branch, _, logits_branch = res_reattack
            
            branch_logits_list.append(logits_branch)

            if name == "Plain":
                plain_reattacked_denorm = self.cfg.dataset_norm.denormalize(reattacked_ts_norm_branch).tensor

        # 4. アンサンブル統合 (Rank Aggregation)
        # [NumBranches, B, NumClass]
        stacked_logits = torch.stack(branch_logits_list)
        
        # ランク計算: argsortを2回かけて「順位(0-indexed, 小さい方が上位)」に変換
        # descending=True (スコア大=上位) でソート
        ranks = stacked_logits.argsort(dim=2, descending=True).argsort(dim=2)
        
        # 順位の合計 (Borda Count)
        summed_ranks = torch.sum(ranks, dim=0) # [B, NumClass]
        
        # 最終予測: 合計順位が最も小さい(良い)クラス
        pred_ensemble = summed_ranks.argmin(dim=1, keepdim=True)
        
        # --- アンサンブル指標計算 ---
        # ランク精度の計算
        # summed_ranks(小さい方が良い)に基づいて順位付け
        ensemble_rank_order = summed_ranks.argsort(dim=1, descending=False).argsort(dim=1)
        rank_correct_ensemble = ensemble_rank_order.gather(1, target_indices).view(-1) + 1
        
        # 確率の計算 (参考値としてSoft Votingの平均確率を使用)
        stacked_probs = F.softmax(stacked_logits, dim=2)
        mean_probs = torch.mean(stacked_probs, dim=0)
        prob_correct_ensemble = mean_probs.gather(1, target_indices).view(-1)

        return (current_batch_size, target_ts, pred_clean, pred_attacked, pred_ensemble,
                prob_correct_attacked, rank_correct_attacked, 
                prob_correct_ensemble, rank_correct_ensemble,
                plain_reattacked_denorm)


# --- 引数解析 ---
def fraction_float(s: str) -> float:
    if "/" in s:
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            raise argparse.ArgumentTypeError(f'"{s}" is not a valid fraction.')
    try: return float(s)
    except ValueError: raise argparse.ArgumentTypeError(f'"{s}" is not a valid float.')

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Ensemble Branch Re-Attack Runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], required=True)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], required=True)
    
    parser.add_argument("--attack-eps", type=fraction_float, required=True)
    parser.add_argument("--attack-alpha", type=fraction_float, required=True)
    parser.add_argument("--attack-n", type=int, required=True)
    
    parser.add_argument("--reattack-eps", type=fraction_float, required=True)
    parser.add_argument("--reattack-alpha", type=fraction_float, required=True)
    parser.add_argument("--reattack-n", type=int, required=True)

    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--save-images', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-dataloader", action="store_true", default=DEFAULT_SHUFFLE_DATALOADER)

    args = parser.parse_args()

    attack_params = BIMAttackParam(epsilon=args.attack_eps, alpha=args.attack_alpha, iters=args.attack_n, batch_size=args.batch_size)
    reattack_params = BIMAttackParam(epsilon=args.reattack_eps, alpha=args.reattack_alpha, iters=args.reattack_n, batch_size=args.batch_size)

    return Config(
        dataset=DatasetKind(args.dataset), model=ModelKind(args.model), model_dir=args.model_dir,
        batch_size=args.batch_size, output_dir=args.output_dir, num_samples=args.num_samples,
        save_images=args.save_images, shuffle_dataloader=args.shuffle_dataloader,
        device=utils.get_device(), dataset_norm=DatasetNorm(DatasetKind(args.dataset), utils.get_device()),
        attack_params=attack_params, reattack_params=reattack_params
    )

@dataclass
class ExperimentStats:
    total: int = 0
    clean_correct: int = 0
    attack_successful: int = 0
    ensemble_successful_to_clean: int = 0
    ensemble_correct: int = 0

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result_generator = runner.run()

    # 出力パス設定
    attack_str = f"bim_eps{cfg.attack_params.epsilon:.3f}"
    reattack_str = f"bim_eps{cfg.reattack_params.epsilon:.3f}"
    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.model.value, attack_str, reattack_str)
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "branch_ensemble_results.csv")
    csv_header = [
        "index", "target_label", "pred_clean", "pred_attacked", "pred_ensemble",
        "prob_correct_attack", "rank_correct_attack", 
        "prob_correct_ensemble", "rank_correct_ensemble",
        "image_path"
    ]
    
    with concurrent.futures.ThreadPoolExecutor() as executor, open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        futures = {}
        stats = ExperimentStats()

        for res in result_generator:
            stats.total += 1
            if res.pred_ensemble == res.target_label:
                stats.ensemble_correct += 1

            if res.pred_clean == res.target_label:
                stats.clean_correct += 1
                if res.pred_attacked != res.target_label: # 攻撃成功
                    stats.attack_successful += 1
                    if res.pred_ensemble == res.target_label: # アンサンブルで復元
                        stats.ensemble_successful_to_clean += 1

            image_path = ""
            if cfg.save_images:
                image_path = os.path.join(output_folder, f"idx_{res.index}_target{res.target_label}_ens{res.pred_ensemble}.png")
                # Plainルートの再攻撃画像を保存
                future = executor.submit(utils.save_tensor_as_image, res.reattacked_image_ts.tensor, image_path)
                futures[future] = res.index

            writer.writerow([
                res.index, res.target_label, res.pred_clean, res.pred_attacked, res.pred_ensemble,
                res.attack_prob_correct, res.attack_rank_correct,
                res.ensemble_prob_correct, res.ensemble_rank_correct,
                image_path
            ])

        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception: pass

    print("\n=== 分岐型アンサンブル再攻撃結果 (Branch Ensemble) ===")
    print(f"  Total Samples: {stats.total}")
    print(f"  Clean Accuracy: {stats.clean_correct}/{stats.total} ({stats.clean_correct/stats.total:.4f})")
    print(f"  Ensemble Accuracy (All): {stats.ensemble_correct}/{stats.total} ({stats.ensemble_correct/stats.total:.4f})")

    if stats.clean_correct > 0 and stats.attack_successful > 0:
        print(f"  Attack Success Rate: {stats.attack_successful}/{stats.clean_correct} ({stats.attack_successful/stats.clean_correct:.4f})")
        print(f"  Ensemble Recovery (Top-1): {stats.ensemble_successful_to_clean}/{stats.attack_successful} ({stats.ensemble_successful_to_clean/stats.attack_successful:.4f})")
            
    print(f"Saved to {csv_filename}")

if __name__ == "__main__":
    main()