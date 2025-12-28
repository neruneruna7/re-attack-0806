# Re-Attack後にフィルタ処理を加えたアンサンブル推論を行うスクリプト (BIM専用版)
from dataclasses import dataclass, field, asdict
from typing import Final, Iterator, Tuple, List
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

from re_attack_0806 import utils
from re_attack_0806.attacks import bim
from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm, BIMAttackParam, AttackParams
from re_attack_0806.utils.normTensor import *
from tqdm import tqdm

from torch import Tensor

# --- 定数定義 ---
DEFAULT_MODEL_DIR = "./weight"
DEFAULT_OUTPUT_DIR = "ensemble_reattacked_data"
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
    pred_reattacked: int
    pred_ensemble: int  # アンサンブルによる最終予測
    
    l2_clean_vs_attacked: float
    l2_attacked_vs_reattacked: float
    l2_clean_vs_reattacked: float
    reattacked_image_ts: TensorWithState

    # 確率とランク
    attack_prob_correct: float
    attack_rank_correct: int
    reattack_prob_correct: float
    reattack_rank_correct: int
    ensemble_prob_correct: float # アンサンブル後
    ensemble_rank_correct: int   # アンサンブル後


class DefenseFilter:
    """PyTorch Tensor用の防御フィルタ群"""
    def __init__(self, device: torch.device):
        self.device = device

    def gaussian_blur(self, x: Tensor, kernel_size=3, sigma=1.0) -> Tensor:
        # x: [B, C, H, W]
        # kernel_sizeは奇数である必要がある
        if kernel_size % 2 == 0: kernel_size += 1
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return transform(x)

    def median_filter(self, x: Tensor, kernel_size=3) -> Tensor:
        """
        メディアンフィルタの実装 (unfoldを使用)
        x: [B, C, H, W]
        """
        if kernel_size % 2 == 0: kernel_size += 1
        padding = kernel_size // 2
        b, c, h, w = x.shape
        
        # チャネルごとに処理するために [B*C, 1, H, W] に変形
        x_reshaped = x.view(b * c, 1, h, w)
        
        # Unfold: [B*C, kernel_size*kernel_size, H*W]
        x_unfolded = F.unfold(x_reshaped, kernel_size, padding=padding)
        
        # 中央値を取得 (dim=1) -> [B*C, H*W]
        x_median = x_unfolded.median(dim=1).values
        
        # 元の形状に戻す
        x_out = x_median.view(b, c, h, w)
        return x_out

    def resize_restore(self, x: Tensor, scale_factor=0.9) -> Tensor:
        """画像を縮小してから元のサイズに戻す（高周波成分の除去）"""
        _, _, h, w = x.shape
        h_small, w_small = int(h * scale_factor), int(w * scale_factor)
        
        # 縮小 -> 拡大 (Bilinear)
        x_small = F.interpolate(x, size=(h_small, w_small), mode='bilinear', align_corners=False)
        x_restored = F.interpolate(x_small, size=(h, w), mode='bilinear', align_corners=False)
        return x_restored


# 攻撃実行関数 (BIM専用)
def run_bim_attack(attack_params: BIMAttackParam, input_data_norm: NormTensor, target_labels: Tensor, eval_model: nn.Module, device: torch.device, mean: Tensor, std: Tensor) -> Tuple[utils.TensorWithState[bim.Normalized], Tensor, Tensor]:
    
    # BIM実行
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
        
        # フィルタ初期化
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
        print(f"Running Ensemble Re-Attack (BIM only) with config: {self.cfg}")
        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Ensemble Re-Attacking"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples: break
            
            results = self.inner_loop(data, target)
            
            # results展開
            (current_batch_size, target_ts, pred_clean, pred_attacked, pred_reattacked, pred_ensemble,
             reattacked_data_denorm, l2_c_a, l2_a_r, l2_c_r, 
             prob_r, rank_r, prob_a, rank_a, prob_e, rank_e) = results

            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples: return

                reattacked_sample_denorm_ts = TensorWithState(reattacked_data_denorm[j].detach().cpu(), DENORMALIZED)

                yield ReAttackResult(
                    index=global_idx,
                    target_label=target_ts[j].item().as_integer_ratio()[0],
                    pred_clean=pred_clean.view(-1)[j].item().as_integer_ratio()[0],
                    pred_attacked=pred_attacked.view(-1)[j].item().as_integer_ratio()[0],
                    pred_reattacked=pred_reattacked.view(-1)[j].item().as_integer_ratio()[0],
                    pred_ensemble=pred_ensemble.view(-1)[j].item().as_integer_ratio()[0],
                    l2_clean_vs_attacked=l2_c_a[j].item(),
                    l2_attacked_vs_reattacked=l2_a_r[j].item(),
                    l2_clean_vs_reattacked=l2_c_r[j].item(),
                    reattacked_image_ts=reattacked_sample_denorm_ts,
                    attack_prob_correct=prob_a[j].item(),
                    attack_rank_correct=rank_a[j].item(),
                    reattack_prob_correct=prob_r[j].item(),
                    reattack_rank_correct=rank_r[j].item(),
                    ensemble_prob_correct=prob_e[j].item(),
                    ensemble_rank_correct=rank_e[j].item()
                )
                global_idx += 1

    def inner_loop(self, data, target):
        eval_model = self.model.eval()
        current_batch_size = data.shape[0]
        clean_data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
        clean_data_ts_norm = self.cfg.dataset_norm.normalize(clean_data_ts)
        target_ts = target.to(self.device).view(-1).long()
        target_indices = target_ts.view(-1, 1)

        # 1. クリーンデータ推論
        out_clean = eval_model(clean_data_ts_norm.tensor)
        pred_clean = out_clean.max(1, keepdim=True)[1]
        
        # 2. 攻撃 (BIM Attack)
        res_attack = run_bim_attack(
            self.cfg.attack_params, clean_data_ts_norm, target_ts, 
            eval_model, self.device, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std
        )
        attacked_ts_norm, pred_attacked, logits_attacked = res_attack
        
        # 攻撃後の指標計算
        probs_attacked = F.softmax(logits_attacked, dim=1)
        prob_correct_attacked = probs_attacked.gather(1, target_indices).view(-1)
        target_logits_attacked = logits_attacked.gather(1, target_indices)
        rank_correct_attacked = (logits_attacked > target_logits_attacked).sum(dim=1) + 1

        # 3. 再攻撃 (Re-Attack Defense - BIM)
        reattack_target = pred_attacked.view(-1).long()
        res_reattack = run_bim_attack(
            self.cfg.reattack_params, attacked_ts_norm, reattack_target, 
            eval_model, self.device, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std
        )
        reattacked_ts_norm, pred_reattacked, logits_reattacked = res_reattack

        # 再攻撃後の指標計算 (アンサンブル前)
        probs_reattacked = F.softmax(logits_reattacked, dim=1)
        prob_correct = probs_reattacked.gather(1, target_indices).view(-1)
        target_logits = logits_reattacked.gather(1, target_indices)
        rank_correct = (logits_reattacked > target_logits).sum(dim=1) + 1
        
        # 4. アンサンブル推論 (4種のフィルタ処理)
        # reattacked_ts_norm.tensor は正規化済み画像 [B, C, H, W]
        x_orig = reattacked_ts_norm.tensor
        
        with torch.no_grad():
            # A. Original (Re-Attacked)
            logits_1 = logits_reattacked
            
            # B. Gaussian Blur
            x_blur = self.filters.gaussian_blur(x_orig, kernel_size=3, sigma=1.0)
            logits_2 = _to_logits(eval_model(x_blur))
            
            # C. Median Filter
            x_median = self.filters.median_filter(x_orig, kernel_size=3)
            logits_3 = _to_logits(eval_model(x_median))
            
            # D. Resize & Restore (Scale 0.9)
            x_resize = self.filters.resize_restore(x_orig, scale_factor=0.9)
            logits_4 = _to_logits(eval_model(x_resize))
            
            # アンサンブル (ロジットの平均)
            stacked_logits = torch.stack([logits_1, logits_2, logits_3, logits_4]) # [4, B, NumClass]
            mean_logits = torch.mean(stacked_logits, dim=0) # [B, NumClass]
            
            pred_ensemble = mean_logits.max(1, keepdim=True)[1]
            
            # アンサンブル後の指標計算
            probs_ensemble = F.softmax(mean_logits, dim=1)
            prob_correct_ensemble = probs_ensemble.gather(1, target_indices).view(-1)
            target_logits_ens = mean_logits.gather(1, target_indices)
            rank_correct_ensemble = (mean_logits > target_logits_ens).sum(dim=1) + 1


        # L2距離計算などの後処理
        clean_denorm = self.cfg.dataset_norm.denormalize(clean_data_ts_norm).tensor
        attacked_denorm = self.cfg.dataset_norm.denormalize(attacked_ts_norm).tensor
        reattacked_denorm = self.cfg.dataset_norm.denormalize(reattacked_ts_norm).tensor
        
        l2_c_a = torch.linalg.norm((attacked_denorm - clean_denorm).view(current_batch_size, -1), ord=2, dim=1)
        l2_a_r = torch.linalg.norm((reattacked_denorm - attacked_denorm).view(current_batch_size, -1), ord=2, dim=1)
        l2_c_r = torch.linalg.norm((reattacked_denorm - clean_denorm).view(current_batch_size, -1), ord=2, dim=1)
        
        return (current_batch_size, target_ts, pred_clean, pred_attacked, pred_reattacked, pred_ensemble,
                reattacked_denorm, l2_c_a, l2_a_r, l2_c_r, 
                prob_correct, rank_correct, prob_correct_attacked, rank_correct_attacked, 
                prob_correct_ensemble, rank_correct_ensemble)


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
    parser = argparse.ArgumentParser(description="Ensemble Re-Attack Runner (BIM Only)")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], required=True)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], required=True)
    
    # Attack/ReAttackはBIM固定なのでパラメータのみ受け取る
    parser.add_argument("--attack-eps", type=fraction_float, required=True, help="Epsilon for initial BIM attack.")
    parser.add_argument("--attack-alpha", type=fraction_float, required=True, help="Alpha for initial BIM attack.")
    parser.add_argument("--attack-n", type=int, required=True, help="Iterations for initial BIM attack.")
    
    parser.add_argument("--reattack-eps", type=fraction_float, required=True, help="Epsilon for BIM re-attack.")
    parser.add_argument("--reattack-alpha", type=fraction_float, required=True, help="Alpha for BIM re-attack.")
    parser.add_argument("--reattack-n", type=int, required=True, help="Iterations for BIM re-attack.")

    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--save-images', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-dataloader", action="store_true", default=DEFAULT_SHUFFLE_DATALOADER)

    args = parser.parse_args()

    # BIMパラメータのみ構築
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
    reattack_successful: int = 0 # 予測変化
    reattack_successful_to_clean: int = 0 # 正解に戻った
    
    ensemble_successful_to_clean: int = 0 # アンサンブルで正解に戻った
    
    # New: 全サンプルに対するアンサンブル正解数
    ensemble_correct: int = 0

    # 統計用リスト (攻撃成功サンプルのみ)
    prob_correct_attack_list: list = field(default_factory=list)
    rank_correct_attack_list: list = field(default_factory=list)
    
    prob_correct_reattack_list: list = field(default_factory=list)
    rank_correct_reattack_list: list = field(default_factory=list)
    
    prob_correct_ensemble_list: list = field(default_factory=list)
    rank_correct_ensemble_list: list = field(default_factory=list)

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result_generator = runner.run()

    # 出力パス設定
    attack_str = f"bim_eps{cfg.attack_params.epsilon:.3f}"
    reattack_str = f"bim_eps{cfg.reattack_params.epsilon:.3f}"
    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.model.value, attack_str, reattack_str)
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "0_ensemble_results.csv")
    csv_header = [
        "index", "target_label", "pred_clean", "pred_attacked", "pred_reattacked", "pred_ensemble",
        "l2_clean_vs_attacked", "l2_attacked_vs_reattacked", "l2_clean_vs_reattacked", 
        "prob_correct_attack", "rank_correct_attack", 
        "prob_correct_reattack", "rank_correct_reattack", 
        "prob_correct_ensemble", "rank_correct_ensemble",
        "reattacked_image_path"
    ]
    
    with concurrent.futures.ThreadPoolExecutor() as executor, open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        futures = {}
        stats = ExperimentStats()

        for res in result_generator:
            stats.total += 1
            
            # 全サンプルでのアンサンブル正解判定
            if res.pred_ensemble == res.target_label:
                stats.ensemble_correct += 1

            # 統計更新
            if res.pred_clean == res.target_label:
                stats.clean_correct += 1
                if res.pred_attacked != res.target_label: # 攻撃成功
                    stats.attack_successful += 1
                    
                    # リストに追加
                    stats.prob_correct_attack_list.append(res.attack_prob_correct)
                    stats.rank_correct_attack_list.append(res.attack_rank_correct)
                    stats.prob_correct_reattack_list.append(res.reattack_prob_correct)
                    stats.rank_correct_reattack_list.append(res.reattack_rank_correct)
                    stats.prob_correct_ensemble_list.append(res.ensemble_prob_correct)
                    stats.rank_correct_ensemble_list.append(res.ensemble_rank_correct)

                    if res.pred_reattacked != res.pred_attacked:
                        stats.reattack_successful += 1
                    if res.pred_reattacked == res.target_label:
                        stats.reattack_successful_to_clean += 1
                    if res.pred_ensemble == res.target_label: # アンサンブル成功
                        stats.ensemble_successful_to_clean += 1

            image_path = ""
            if cfg.save_images:
                image_path = os.path.join(output_folder, f"idx_{res.index}_target{res.target_label}_ens{res.pred_ensemble}.png")
                future = executor.submit(utils.save_tensor_as_image, res.reattacked_image_ts.tensor, image_path)
                futures[future] = res.index

            writer.writerow([
                res.index, res.target_label, res.pred_clean, res.pred_attacked, res.pred_reattacked, res.pred_ensemble,
                res.l2_clean_vs_attacked, res.l2_attacked_vs_reattacked, res.l2_clean_vs_reattacked,
                res.attack_prob_correct, res.attack_rank_correct,
                res.reattack_prob_correct, res.reattack_rank_correct,
                res.ensemble_prob_correct, res.ensemble_rank_correct,
                image_path
            ])

        # 画像保存待ち
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as e: print(f"Image save error: {e}")

    # 結果表示
    print("\n=== アンサンブル再攻撃結果 (BIM Only) ===")
    print(f"  Total Samples: {stats.total}")
    print(f"  Clean Accuracy: {stats.clean_correct}/{stats.total} ({stats.clean_correct/stats.total:.4f})")
    
    # 全サンプルに対するアンサンブル精度
    print(f"  Ensemble Accuracy (All):    {stats.ensemble_correct}/{stats.total} ({stats.ensemble_correct/stats.total:.4f})")

    if stats.clean_correct > 0:
        print(f"  Attack Success Rate: {stats.attack_successful}/{stats.clean_correct} ({stats.attack_successful/stats.clean_correct:.4f})")
        if stats.attack_successful > 0:
            print(f"  Re-Attack Recovery (Top-1): {stats.reattack_successful_to_clean}/{stats.attack_successful} ({stats.reattack_successful_to_clean/stats.attack_successful:.4f})")
            print(f"  Ensemble Recovery (Top-1):  {stats.ensemble_successful_to_clean}/{stats.attack_successful} ({stats.ensemble_successful_to_clean/stats.attack_successful:.4f})")
            
            # 統計量表示用関数
            def print_stats(name, data):
                if not data: return
                d = np.array(data)
                print(f"  [{name}] Mean: {np.mean(d):.4f}, Median: {np.median(d):.4f}, Max: {np.max(d):.4f}")

            print("\n  [詳細統計 (攻撃成功サンプルのみ)]")
            print_stats("Attack Rank", stats.rank_correct_attack_list)
            print_stats("Re-Attack Rank", stats.rank_correct_reattack_list)
            print_stats("Ensemble Rank", stats.rank_correct_ensemble_list)
            
    print(f"Saved to {csv_filename}")

if __name__ == "__main__":
    main()