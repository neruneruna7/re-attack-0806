# 攻撃後に4分岐で前処理を適用し、それぞれ再攻撃した結果をロジット平均でアンサンブルするスクリプト (BIM専用版)
from dataclasses import dataclass, field
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

from re_attack_0806 import utils
from re_attack_0806.attacks import bim
from re_attack_0806.utils.config import DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm, BIMAttackParam
from re_attack_0806.utils.normTensor import *
from tqdm import tqdm

from torch import Tensor

# --- 定数定義 ---
DEFAULT_MODEL_DIR = "./weight"
DEFAULT_OUTPUT_DIR = "ensemble_preprocess_reattacked_data"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_SAMPLES = -1
DEFAULT_SHUFFLE_DATALOADER = False


@dataclass
class Config:
    dataset: DatasetKind
    model: ModelKind
    attack_params: BIMAttackParam
    reattack_params: BIMAttackParam

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
    pred_gaussian: int
    pred_median: int
    pred_resize_restore: int
    pred_ensemble: int

    attack_prob_correct: float
    attack_rank_correct: int
    gaussian_prob_correct: float
    gaussian_rank_correct: int
    median_prob_correct: float
    median_rank_correct: int
    resize_restore_prob_correct: float
    resize_restore_rank_correct: int
    ensemble_prob_correct: float
    ensemble_rank_correct: int

    reattacked_image_ts: TensorWithState


class DefenseFilter:
    """PyTorch Tensor用の前処理群"""

    def __init__(self, device: torch.device):
        self.device = device

    def gaussian_blur(self, x: Tensor, kernel_size: int = 3, sigma: float = 1.0) -> Tensor:
        if kernel_size % 2 == 0:
            kernel_size += 1
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return transform(x)

    def median_filter(self, x: Tensor, kernel_size: int = 3) -> Tensor:
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        x_unfolded = F.unfold(x_reshaped, kernel_size, padding=padding)
        x_median = x_unfolded.median(dim=1).values
        return x_median.view(b, c, h, w)

    def resize_restore(self, x: Tensor, scale_factor: float = 0.9) -> Tensor:
        _, _, h, w = x.shape
        h_small = max(1, int(h * scale_factor))
        w_small = max(1, int(w * scale_factor))
        x_small = F.interpolate(x, size=(h_small, w_small), mode="bilinear", align_corners=False)
        return F.interpolate(x_small, size=(h, w), mode="bilinear", align_corners=False)


def run_bim_attack(
    attack_params: BIMAttackParam,
    input_data_norm: NormTensor,
    target_labels: Tensor,
    eval_model: nn.Module,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> Tuple[utils.TensorWithState[bim.Normalized], Tensor, Tensor]:
    attacked_ts_norm = bim.bim(
        input_data_norm,
        target_labels,
        eval_model,
        device,
        attack_params.epsilon,
        attack_params.alpha,
        attack_params.iters,
        mean,
        std,
    )

    attacked_ts_norm_f: Final[NormTensor] = attacked_ts_norm
    logits_attacked: Final[Tensor] = _to_logits(eval_model(attacked_ts_norm_f.tensor))
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
        self.test_loader = DataFactory.loader(
            cfg.dataset,
            train=False,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_dataloader,
        )
        self._load_weights_if_exists(cfg.model_dir)
        self.model.eval()
        self.filters = DefenseFilter(self.device)

    def _load_weights_if_exists(self, model_dir: str):
        model_name = getattr(self.model, "model_name", None)
        if model_name is None:
            return
        path = os.path.join(model_dir, f"{model_name}.ckpt")
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                st = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
                self.model.load_state_dict(st)
                print(f"loaded weights from {path}")
            except Exception as e:
                print(f"failed to load state_dict from {path}: {e}")

    def run(self) -> Iterator[ReAttackResult]:
        print(f"Running Ensemble Preprocess -> Re-Attack (BIM only) with config: {self.cfg}")
        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Preprocess Branch Re-Attacking"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                break

            (
                current_batch_size,
                target_ts,
                pred_clean,
                pred_attacked,
                pred_gaussian,
                pred_median,
                pred_resize_restore,
                pred_ensemble,
                prob_a,
                rank_a,
                prob_g,
                rank_g,
                prob_m,
                rank_m,
                prob_r,
                rank_r,
                prob_e,
                rank_e,
                sample_reattacked_denorm,
            ) = self.inner_loop(data, target)

            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                    return

                yield ReAttackResult(
                    index=global_idx,
                    target_label=target_ts[j].item().as_integer_ratio()[0],
                    pred_clean=pred_clean.view(-1)[j].item().as_integer_ratio()[0],
                    pred_attacked=pred_attacked.view(-1)[j].item().as_integer_ratio()[0],
                    pred_gaussian=pred_gaussian.view(-1)[j].item().as_integer_ratio()[0],
                    pred_median=pred_median.view(-1)[j].item().as_integer_ratio()[0],
                    pred_resize_restore=pred_resize_restore.view(-1)[j].item().as_integer_ratio()[0],
                    pred_ensemble=pred_ensemble.view(-1)[j].item().as_integer_ratio()[0],
                    attack_prob_correct=prob_a[j].item(),
                    attack_rank_correct=rank_a[j].item(),
                    gaussian_prob_correct=prob_g[j].item(),
                    gaussian_rank_correct=rank_g[j].item(),
                    median_prob_correct=prob_m[j].item(),
                    median_rank_correct=rank_m[j].item(),
                    resize_restore_prob_correct=prob_r[j].item(),
                    resize_restore_rank_correct=rank_r[j].item(),
                    ensemble_prob_correct=prob_e[j].item(),
                    ensemble_rank_correct=rank_e[j].item(),
                    reattacked_image_ts=TensorWithState(sample_reattacked_denorm[j].detach().cpu(), DENORMALIZED),
                )
                global_idx += 1

    def inner_loop(self, data: Tensor, target: Tensor):
        eval_model = self.model.eval()
        current_batch_size = data.shape[0]

        clean_data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
        clean_data_ts_norm = self.cfg.dataset_norm.normalize(clean_data_ts)
        target_ts = target.to(self.device).view(-1).long()
        target_indices = target_ts.view(-1, 1)

        logits_clean = _to_logits(eval_model(clean_data_ts_norm.tensor))
        pred_clean = logits_clean.max(1, keepdim=True)[1]

        # 初回攻撃: clean 入力に対して BIM を適用し、再攻撃の出発点となる敵対的サンプルを作る。
        attacked_ts_norm, pred_attacked, logits_attacked = run_bim_attack(
            self.cfg.attack_params,
            clean_data_ts_norm,
            target_ts,
            eval_model,
            self.device,
            self.cfg.dataset_norm.mean,
            self.cfg.dataset_norm.std,
        )

        probs_attacked = F.softmax(logits_attacked, dim=1)
        prob_correct_attacked = probs_attacked.gather(1, target_indices).view(-1)
        target_logits_attacked = logits_attacked.gather(1, target_indices)
        rank_correct_attacked = (logits_attacked > target_logits_attacked).sum(dim=1) + 1

        branches: List[Tuple[str, Callable[[Tensor], Tensor]]] = [
            ("gaussian", lambda x: self.filters.gaussian_blur(x, kernel_size=3, sigma=1.0)),
            ("median", lambda x: self.filters.median_filter(x, kernel_size=3)),
            ("resize_restore", lambda x: self.filters.resize_restore(x, scale_factor=0.9)),
        ]

        branch_logits_map: dict[str, Tensor] = {}
        representative_reattacked_denorm = None
        reattack_target = pred_attacked.view(-1).long()

        for name, preprocess_fn in branches:
            # 画像処理: 初回攻撃後の入力に対して各分岐の前処理を適用する。
            # ここでは gaussian / median / resize_restore の 3 分岐を作る。
            # 既存の分岐実装との整合性を優先し、正規化済みテンソルに直接前処理を適用する。
            preprocessed_tensor = preprocess_fn(attacked_ts_norm.tensor)
            preprocessed_ts_norm = TensorWithState(preprocessed_tensor, NORMALIZED)

            # 再攻撃: 各前処理分岐の出力に対して、初回攻撃で得た予測ラベルをターゲットとして
            # 同一条件の BIM を再度適用する。分岐ごとの差は前処理だけである。
            reattacked_ts_norm, _, logits_branch = run_bim_attack(
                self.cfg.reattack_params,
                preprocessed_ts_norm,
                reattack_target,
                eval_model,
                self.device,
                self.cfg.dataset_norm.mean,
                self.cfg.dataset_norm.std,
            )
            branch_logits_map[name] = logits_branch

            if representative_reattacked_denorm is None:
                # 保存画像は先頭の前処理分岐の再攻撃結果を代表として使う。
                representative_reattacked_denorm = self.cfg.dataset_norm.denormalize(reattacked_ts_norm).tensor

        if representative_reattacked_denorm is None:
            raise RuntimeError("Representative branch image was not generated.")

        gaussian_logits = branch_logits_map["gaussian"]
        median_logits = branch_logits_map["median"]
        resize_restore_logits = branch_logits_map["resize_restore"]

        pred_gaussian = gaussian_logits.max(1, keepdim=True)[1]
        pred_median = median_logits.max(1, keepdim=True)[1]
        pred_resize_restore = resize_restore_logits.max(1, keepdim=True)[1]

        probs_gaussian = F.softmax(gaussian_logits, dim=1)
        prob_correct_gaussian = probs_gaussian.gather(1, target_indices).view(-1)
        target_logits_gaussian = gaussian_logits.gather(1, target_indices)
        rank_correct_gaussian = (gaussian_logits > target_logits_gaussian).sum(dim=1) + 1

        probs_median = F.softmax(median_logits, dim=1)
        prob_correct_median = probs_median.gather(1, target_indices).view(-1)
        target_logits_median = median_logits.gather(1, target_indices)
        rank_correct_median = (median_logits > target_logits_median).sum(dim=1) + 1

        probs_resize_restore = F.softmax(resize_restore_logits, dim=1)
        prob_correct_resize_restore = probs_resize_restore.gather(1, target_indices).view(-1)
        target_logits_resize_restore = resize_restore_logits.gather(1, target_indices)
        rank_correct_resize_restore = (resize_restore_logits > target_logits_resize_restore).sum(dim=1) + 1

        # アンサンブル: 3 分岐それぞれの再攻撃後ロジットを積み上げ、平均して最終予測を作る。
        # この平均ロジットに対して softmax / rank を計算し、アンサンブル後の性能を評価する。
        stacked_logits = torch.stack([gaussian_logits, median_logits, resize_restore_logits])
        mean_logits = torch.mean(stacked_logits, dim=0)
        pred_ensemble = mean_logits.max(1, keepdim=True)[1]

        probs_ensemble = F.softmax(mean_logits, dim=1)
        prob_correct_ensemble = probs_ensemble.gather(1, target_indices).view(-1)
        target_logits_ensemble = mean_logits.gather(1, target_indices)
        rank_correct_ensemble = (mean_logits > target_logits_ensemble).sum(dim=1) + 1

        return (
            current_batch_size,
            target_ts,
            pred_clean,
            pred_attacked,
            pred_gaussian,
            pred_median,
            pred_resize_restore,
            pred_ensemble,
            prob_correct_attacked,
            rank_correct_attacked,
            prob_correct_gaussian,
            rank_correct_gaussian,
            prob_correct_median,
            rank_correct_median,
            prob_correct_resize_restore,
            rank_correct_resize_restore,
            prob_correct_ensemble,
            rank_correct_ensemble,
            representative_reattacked_denorm,
        )


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
    parser = argparse.ArgumentParser(description="Ensemble Preprocess -> Re-Attack Runner (BIM Only)")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], required=True)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], required=True)

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
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-dataloader", action="store_true", default=DEFAULT_SHUFFLE_DATALOADER)

    args = parser.parse_args()

    attack_params = BIMAttackParam(
        epsilon=args.attack_eps,
        alpha=args.attack_alpha,
        iters=args.attack_n,
        batch_size=args.batch_size,
    )
    reattack_params = BIMAttackParam(
        epsilon=args.reattack_eps,
        alpha=args.reattack_alpha,
        iters=args.reattack_n,
        batch_size=args.batch_size,
    )

    dataset = DatasetKind(args.dataset)
    device = utils.get_device()

    return Config(
        dataset=dataset,
        model=ModelKind(args.model),
        attack_params=attack_params,
        reattack_params=reattack_params,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        save_images=args.save_images,
        shuffle_dataloader=args.shuffle_dataloader,
        device=device,
        dataset_norm=DatasetNorm(dataset, device),
    )


@dataclass
class ExperimentStats:
    total: int = 0
    clean_correct: int = 0
    attack_successful: int = 0
    gaussian_successful_to_clean: int = 0
    gaussian_correct: int = 0
    median_successful_to_clean: int = 0
    median_correct: int = 0
    resize_restore_successful_to_clean: int = 0
    resize_restore_correct: int = 0
    ensemble_successful_to_clean: int = 0
    ensemble_correct: int = 0

    prob_correct_attack_list: list = field(default_factory=list)
    rank_correct_attack_list: list = field(default_factory=list)
    prob_correct_gaussian_list: list = field(default_factory=list)
    rank_correct_gaussian_list: list = field(default_factory=list)
    prob_correct_median_list: list = field(default_factory=list)
    rank_correct_median_list: list = field(default_factory=list)
    prob_correct_resize_restore_list: list = field(default_factory=list)
    rank_correct_resize_restore_list: list = field(default_factory=list)
    prob_correct_ensemble_list: list = field(default_factory=list)
    rank_correct_ensemble_list: list = field(default_factory=list)


def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result_generator = runner.run()

    attack_str = f"bim_eps{cfg.attack_params.epsilon:.3f}"
    reattack_str = f"bim_eps{cfg.reattack_params.epsilon:.3f}"
    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.model.value, attack_str, reattack_str)
    os.makedirs(output_folder, exist_ok=True)

    csv_filename = os.path.join(output_folder, "preprocess_branch_ensemble_results.csv")
    csv_header = [
        "index",
        "target_label",
        "pred_clean",
        "pred_attacked",
        "pred_gaussian",
        "pred_median",
        "pred_resize_restore",
        "pred_ensemble",
        "prob_correct_attack",
        "rank_correct_attack",
        "prob_correct_gaussian",
        "rank_correct_gaussian",
        "prob_correct_median",
        "rank_correct_median",
        "prob_correct_resize_restore",
        "rank_correct_resize_restore",
        "prob_correct_ensemble",
        "rank_correct_ensemble",
        "image_path",
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor, open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        futures = {}
        stats = ExperimentStats()

        for res in result_generator:
            stats.total += 1
            if res.pred_gaussian == res.target_label:
                stats.gaussian_correct += 1
            if res.pred_median == res.target_label:
                stats.median_correct += 1
            if res.pred_resize_restore == res.target_label:
                stats.resize_restore_correct += 1
            if res.pred_ensemble == res.target_label:
                stats.ensemble_correct += 1

            if res.pred_clean == res.target_label:
                stats.clean_correct += 1
                if res.pred_attacked != res.target_label:
                    stats.attack_successful += 1
                    stats.prob_correct_attack_list.append(res.attack_prob_correct)
                    stats.rank_correct_attack_list.append(res.attack_rank_correct)
                    stats.prob_correct_gaussian_list.append(res.gaussian_prob_correct)
                    stats.rank_correct_gaussian_list.append(res.gaussian_rank_correct)
                    stats.prob_correct_median_list.append(res.median_prob_correct)
                    stats.rank_correct_median_list.append(res.median_rank_correct)
                    stats.prob_correct_resize_restore_list.append(res.resize_restore_prob_correct)
                    stats.rank_correct_resize_restore_list.append(res.resize_restore_rank_correct)
                    stats.prob_correct_ensemble_list.append(res.ensemble_prob_correct)
                    stats.rank_correct_ensemble_list.append(res.ensemble_rank_correct)

                    if res.pred_gaussian == res.target_label:
                        stats.gaussian_successful_to_clean += 1
                    if res.pred_median == res.target_label:
                        stats.median_successful_to_clean += 1
                    if res.pred_resize_restore == res.target_label:
                        stats.resize_restore_successful_to_clean += 1
                    if res.pred_ensemble == res.target_label:
                        stats.ensemble_successful_to_clean += 1

            image_path = ""
            if cfg.save_images:
                image_path = os.path.join(output_folder, f"idx_{res.index}_target{res.target_label}_ens{res.pred_ensemble}.png")
                future = executor.submit(utils.save_tensor_as_image, res.reattacked_image_ts.tensor, image_path)
                futures[future] = res.index

            writer.writerow([
                res.index,
                res.target_label,
                res.pred_clean,
                res.pred_attacked,
                res.pred_gaussian,
                res.pred_median,
                res.pred_resize_restore,
                res.pred_ensemble,
                res.attack_prob_correct,
                res.attack_rank_correct,
                res.gaussian_prob_correct,
                res.gaussian_rank_correct,
                res.median_prob_correct,
                res.median_rank_correct,
                res.resize_restore_prob_correct,
                res.resize_restore_rank_correct,
                res.ensemble_prob_correct,
                res.ensemble_rank_correct,
                image_path,
            ])

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Image save error: {e}")

    print("\n=== 3分岐前処理後の再攻撃・アンサンブル結果 (BIM Only) ===")
    print(f"  Total Samples: {stats.total}")
    print(f"  Clean Accuracy: {stats.clean_correct}/{stats.total} ({stats.clean_correct/stats.total:.4f})")
    print("\n  [画像処理 + 再攻撃 の結果]")
    print(f"  Gaussian Accuracy (All): {stats.gaussian_correct}/{stats.total} ({stats.gaussian_correct/stats.total:.4f})")
    print(f"  Median Accuracy (All): {stats.median_correct}/{stats.total} ({stats.median_correct/stats.total:.4f})")
    print(f"  ResizeRestore Accuracy (All): {stats.resize_restore_correct}/{stats.total} ({stats.resize_restore_correct/stats.total:.4f})")
    print("\n  [画像処理 + 再攻撃 + アンサンブル の結果]")
    print(f"  Ensemble Accuracy (All): {stats.ensemble_correct}/{stats.total} ({stats.ensemble_correct/stats.total:.4f})")

    if stats.clean_correct > 0:
        print(f"  Attack Success Rate: {stats.attack_successful}/{stats.clean_correct} ({stats.attack_successful/stats.clean_correct:.4f})")
        if stats.attack_successful > 0:
            print("\n  [攻撃成功サンプルに対する回復率]")
            print(f"  Gaussian Recovery (Top-1): {stats.gaussian_successful_to_clean}/{stats.attack_successful} ({stats.gaussian_successful_to_clean/stats.attack_successful:.4f})")
            print(f"  Median Recovery (Top-1): {stats.median_successful_to_clean}/{stats.attack_successful} ({stats.median_successful_to_clean/stats.attack_successful:.4f})")
            print(f"  ResizeRestore Recovery (Top-1): {stats.resize_restore_successful_to_clean}/{stats.attack_successful} ({stats.resize_restore_successful_to_clean/stats.attack_successful:.4f})")
            print(f"  Ensemble Recovery (Top-1): {stats.ensemble_successful_to_clean}/{stats.attack_successful} ({stats.ensemble_successful_to_clean/stats.attack_successful:.4f})")

            def print_stats(name, data):
                if not data:
                    return
                d = np.array(data)
                print(f"  [{name}] Mean: {np.mean(d):.4f}, Median: {np.median(d):.4f}, Max: {np.max(d):.4f}")

            print("\n  [詳細統計 (攻撃成功サンプルのみ)]")
            print_stats("Attack Rank", stats.rank_correct_attack_list)
            print_stats("Gaussian Rank", stats.rank_correct_gaussian_list)
            print_stats("Median Rank", stats.rank_correct_median_list)
            print_stats("ResizeRestore Rank", stats.rank_correct_resize_restore_list)
            print_stats("Ensemble Rank", stats.rank_correct_ensemble_list)

    print(f"Saved to {csv_filename}")


if __name__ == "__main__":
    main()
