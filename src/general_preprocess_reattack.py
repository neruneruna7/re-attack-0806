# 攻撃 -> 前処理 -> 再攻撃 を実行し、その結果を評価・保存する汎用コード
from dataclasses import dataclass, field, asdict
from torch.types import Number
from typing import Any, Iterator, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import csv
import sys
from torch import Tensor
from enum import Enum
import argparse
import concurrent.futures

from re_attack_0806 import attacks, utils
from re_attack_0806.attacks import bim, fgsm
from re_attack_0806.utils.config import (
    AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm,
    FGSMAttackParam, BIMAttackParam, AttackParams,
    PreprocessingKind, GaussianBlurParams, MedianBlurParams, PreprocessingParams
    , PixelReductionParams
)
from re_attack_0806.utils.normTensor import *

import foolbox
from tqdm import tqdm

# --- 定数定義 ---
DEFAULT_MODEL_DIR = "./weight"
DEFAULT_OUTPUT_DIR = "preprocessed_reattacked_data"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_SAMPLES = -1
DEFAULT_SHUFFLE_DATALOADER = False

@dataclass
class Config:
    # 必須パラメータ
    dataset: DatasetKind
    model: ModelKind
    attack_kind: AttackKind
    attack_params: AttackParams
    preprocessing_kind: PreprocessingKind
    preprocessing_params: Optional[PreprocessingParams]
    reattack_kind: AttackKind
    reattack_params: AttackParams
    
    # デフォルト値を持つパラメータ
    model_dir: str
    batch_size: int
    output_dir: str
    num_samples: int
    save_images: bool
    shuffle_dataloader: bool

    # 内部設定
    device: torch.device
    dataset_norm: DatasetNorm

@dataclass
class AttackResult:
    """1サンプルに対する攻撃・前処理・再攻撃の結果を格納するデータクラス。"""
    index: int
    target_label: int
    pred_clean: int
    pred_attacked: int
    pred_preprocessed: int
    pred_reattacked: int
    final_image: TensorWithState

class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, train=False, batch_size=cfg.batch_size, shuffle=cfg.shuffle_dataloader)
        self._load_weights_if_exists(cfg.model_dir)
        self.model.eval()

        self.fmodel = None
        self.attack_obj = None
        self.reattack_obj = None

        is_foolbox_needed = any(k in [AttackKind.FOOLBOX_BIM, AttackKind.FOOLBOX_FGSM] for k in [cfg.attack_kind, cfg.reattack_kind])
        if is_foolbox_needed:
            # MNIST (1ch) の場合に squeeze().tolist() がスカラーを返す問題を修正
            mean_val = self.cfg.dataset_norm.mean.squeeze()
            std_val = self.cfg.dataset_norm.std.squeeze()
            mean_list = [mean_val.item()] if mean_val.dim() == 0 else mean_val.tolist()
            std_list = [std_val.item()] if std_val.dim() == 0 else std_val.tolist()

            preprocessing = dict(mean=mean_list, std=std_list, axis=-3)
            self.fmodel = foolbox.PyTorchModel(self.model, bounds=(0.0, 1.0), preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]

            # attack_obj の設定
            if cfg.attack_kind == AttackKind.FOOLBOX_BIM:
                assert isinstance(cfg.attack_params, BIMAttackParam)
                self.attack_obj = foolbox.attacks.LinfBasicIterativeAttack(steps=cfg.attack_params.iters, abs_stepsize=cfg.attack_params.alpha) # type: ignore[reportPrivateImportUsage]
            elif cfg.attack_kind == AttackKind.FOOLBOX_FGSM:
                assert isinstance(cfg.attack_params, FGSMAttackParam)
                self.attack_obj = foolbox.attacks.FGSM() # type: ignore[reportPrivateImportUsage]

            # reattack_obj の設定
            if cfg.reattack_kind == AttackKind.FOOLBOX_BIM:
                assert isinstance(cfg.reattack_params, BIMAttackParam)
                self.reattack_obj = foolbox.attacks.LinfBasicIterativeAttack(steps=cfg.reattack_params.iters, abs_stepsize=cfg.reattack_params.alpha) # type: ignore[reportPrivateImportUsage]
            elif cfg.reattack_kind == AttackKind.FOOLBOX_FGSM:
                assert isinstance(cfg.reattack_params, FGSMAttackParam)
                self.reattack_obj = foolbox.attacks.FGSM() # type: ignore[reportPrivateImportUsage]

    def _load_weights_if_exists(self, model_dir: str):
        model_name = getattr(self.model, "model_name", None)
        path = os.path.join(model_dir, f"{model_name}.pth")
        if os.path.exists(path) and model_name:
            st = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(st)
                print(f"loaded weights from {path}")
            except Exception:
                print(f"failed to load exact state_dict from {path}")

    @staticmethod
    def _to_logits(output: Tensor) -> Tensor:
        if output.dim() > 2:
            return output.view(output.size(0), -1)
        return output

    def _perform_preprocessing(self, data_norm: TensorWithState) -> TensorWithState:
        kind = self.cfg.preprocessing_kind
        params = self.cfg.preprocessing_params

        if kind == PreprocessingKind.NONE:
            return data_norm
        
        # 前処理は非正規化データに対して行うのが一般的
        data_denorm = self.cfg.dataset_norm.denormalize(data_norm)
        
        processed_denorm_tensor: Tensor
        if kind == PreprocessingKind.GAUSSIAN_BLUR:
            assert isinstance(params, GaussianBlurParams), "Invalid params for Gaussian Blur"
            # TF.gaussian_blurは[..., C, H, W]の形式を期待し、バッチ処理も可能
            processed_denorm_tensor = TF.gaussian_blur(data_denorm.tensor, kernel_size=[params.kernel_size, params.kernel_size], sigma=[params.sigma, params.sigma])
        elif kind == PreprocessingKind.MEDIAN_BLUR:
            # torchvisionにバッチ対応のmedian_blurがないため、ループ処理
            # kornia.filters.median_blur を使うとGPUで高速に処理可能だが、依存を追加する必要がある
            assert isinstance(params, MedianBlurParams), "Invalid params for Median Blur"
            processed_denorm_tensor = torch.stack([
                 TF.to_tensor(TF.to_pil_image(img).filter(ImageFilter.MedianFilter(size=params.kernel_size)))
                 for img in data_denorm.tensor.cpu()
            ]).to(self.device)
        elif kind == PreprocessingKind.PIXEL_REDUCTION:
            # 画素値を一定量減算する前処理
            # 指定がなければデフォルトで 0.1 を引く（値域を [0,1] と仮定し、clamp する）
            assert isinstance(params, PixelReductionParams), "Invalid params for Pixel Reduction"
            # 互換性のため、params に offset 属性がなければエラーにする
            if not hasattr(params, 'offset'):
                raise ValueError("PixelReductionParams must have an 'offset' attribute")
            amount = params.offset
            # data_denorm.tensor は (B, C, H, W)、値域はデノーマイズ済み（通常 [0,1]）
            # 正規化した状態で引く．そしてclampする
            _normed_tensor = self.cfg.dataset_norm.normalize(data_denorm)

            processed_norm_tensor = _normed_tensor.tensor - amount

            min_val_norm = (0 - self.cfg.dataset_norm.mean) / self.cfg.dataset_norm.std
            max_val_norm = (1 - self.cfg.dataset_norm.mean) / self.cfg.dataset_norm.std


            clamped_norm_tensor = processed_norm_tensor.clamp(min_val_norm, max_val_norm)

            processed_denorm_tensor = self.cfg.dataset_norm.denormalize(TensorWithState(clamped_norm_tensor, NORMALIZED)).tensor

            # processed_denorm_tensor = (data_denorm.tensor.float() - float(amount)).clamp(0.0, 1.0)
        else:
            raise NotImplementedError(f"Preprocessing kind {kind} is not implemented.")

        processed_denorm_ts = TensorWithState(processed_denorm_tensor, DENORMALIZED)
        return self.cfg.dataset_norm.normalize(processed_denorm_ts)


    def _perform_attack(self, data_norm: TensorWithState, target: Tensor, kind: AttackKind, params: AttackParams, is_reattack: bool) -> TensorWithState:
        if kind == AttackKind.BIM:
            assert isinstance(params, BIMAttackParam)
            return bim.bim(data_norm, target, self.model, self.device, params.epsilon, params.alpha, params.iters, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std)
        elif kind == AttackKind.FGSM:
            assert isinstance(params, FGSMAttackParam)
            return fgsm.fgsm(data_norm, target, self.model, self.device, params.epsilon, self.cfg.dataset_norm.mean, self.cfg.dataset_norm.std)
        elif kind in [AttackKind.FOOLBOX_BIM, AttackKind.FOOLBOX_FGSM]:
            attack_obj = self.reattack_obj if is_reattack else self.attack_obj
            if self.fmodel is None or attack_obj is None:
                raise RuntimeError("Foolbox model or attack object is not initialized.")
            
            data_denorm = self.cfg.dataset_norm.denormalize(data_norm)
            
            try:
                eps = params.epsilon if hasattr(params, 'epsilon') else None
                _, clipped, _ = attack_obj(self.fmodel, data_denorm.tensor, target, epsilons=eps)
                
                perturbed_denorm_tensor = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
                perturbed_denorm_ts = TensorWithState(perturbed_denorm_tensor, DENORMALIZED)
                return self.cfg.dataset_norm.normalize(perturbed_denorm_ts)
            except Exception as e:
                print(f"Foolbox {kind.value} attack failed: {e}. Returning original data.")
                return data_norm
        else:
            raise ValueError(f"Unsupported attack kind: {kind}")

    def run(self) -> Iterator[AttackResult]:
        print(f"Running Preprocess & Re-Attack with config: {self.cfg}")

        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Preprocess & Re-Attack"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                break
            
            current_batch_size = data.shape[0]
            clean_data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
            clean_data_ts_norm = self.cfg.dataset_norm.normalize(clean_data_ts)
            target_ts: Tensor = target.to(self.device).view(-1).long()
            
            logits_clean = self._to_logits(self.model(clean_data_ts_norm.tensor))
            pred_clean = logits_clean.max(1, keepdim=True)[1]

            attacked_data_ts_norm = self._perform_attack(clean_data_ts_norm, target_ts, self.cfg.attack_kind, self.cfg.attack_params, is_reattack=False)
            logits_attacked = self._to_logits(self.model(attacked_data_ts_norm.tensor))
            pred_attacked = logits_attacked.max(1, keepdim=True)[1]

            preprocessed_data_ts_norm = self._perform_preprocessing(attacked_data_ts_norm)
            logits_preprocessed = self._to_logits(self.model(preprocessed_data_ts_norm.tensor))
            pred_preprocessed = logits_preprocessed.max(1, keepdim=True)[1]

            reattack_target = pred_attacked.view(-1)
            reattacked_data_ts_norm = self._perform_attack(preprocessed_data_ts_norm, reattack_target, self.cfg.reattack_kind, self.cfg.reattack_params, is_reattack=True)
            logits_reattacked = self._to_logits(self.model(reattacked_data_ts_norm.tensor))
            pred_reattacked = logits_reattacked.max(1, keepdim=True)[1]

            probs = F.softmax(logits_attacked, dim=1)
            conf_max = probs.max(1)[0]
            # print(f"Confidence of attacked sample: {conf_max.item()}")
            print(f"Confidence - Mean: {conf_max.mean().item():.4f}, Max: {conf_max.max().item():.4f}, Min: {conf_max.min().item():.4f}")


            reattacked_data_denorm = self.cfg.dataset_norm.denormalize(reattacked_data_ts_norm).tensor
            for j in range(current_batch_size):
                if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                    return

                yield AttackResult(
                    index=global_idx,
                    # そもそも整数型になることはわかりきっているので，.as_integer_ratio()[0]でfloatをintに変換する方法を使う
                    target_label=target_ts[j].item().as_integer_ratio()[0],
                    pred_clean=pred_clean.view(-1)[j].item().as_integer_ratio()[0],
                    pred_attacked=pred_attacked.view(-1)[j].item().as_integer_ratio()[0],
                    pred_preprocessed=pred_preprocessed.view(-1)[j].item().as_integer_ratio()[0],
                    pred_reattacked=pred_reattacked.view(-1)[j].item().as_integer_ratio()[0],
                    final_image=TensorWithState(reattacked_data_denorm[j].detach().cpu(), DENORMALIZED)
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
    parser = argparse.ArgumentParser(description="General Preprocess and Re-Attack Runner")
    # 必須
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], required=True)
    parser.add_argument("--model", choices=[m.value for m in ModelKind], required=True)
    parser.add_argument("--attack-kind", choices=[k.value for k in AttackKind], required=True)
    parser.add_argument("--preprocess-kind", choices=[p.value for p in PreprocessingKind], required=True)
    parser.add_argument("--reattack-kind", choices=[k.value for k in AttackKind], required=True)
    
    # デフォルト値あり
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--save-images', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-dataloader", action="store_true", default=DEFAULT_SHUFFLE_DATALOADER)

    # 攻撃パラメータ
    parser.add_argument("--attack-eps", type=fraction_float)
    parser.add_argument("--attack-alpha", type=fraction_float)
    parser.add_argument("--attack-n", type=int)
    
    # 前処理パラメータ
    parser.add_argument("--preprocess-kernel-size", type=int)
    parser.add_argument("--preprocess-sigma", type=float)
    parser.add_argument("--preprocess-offset", type=float,
                        help="Amount to subtract from each pixel for brightness reduction (default 0.1)")

    # 再攻撃パラメータ
    parser.add_argument("--reattack-eps", type=fraction_float)
    parser.add_argument("--reattack-alpha", type=fraction_float)
    parser.add_argument("--reattack-n", type=int)

    args = parser.parse_args()

    def get_attack_params(kind, eps, alpha, n, batch_size):
        if kind in [AttackKind.FGSM, AttackKind.FOOLBOX_FGSM]:
            if eps is None: parser.error(f"{kind.value} requires --*-eps.")
            return FGSMAttackParam(epsilon=eps, batch_size=batch_size)
        elif kind in [AttackKind.BIM, AttackKind.FOOLBOX_BIM]:
            if eps is None or alpha is None or n is None: parser.error(f"{kind.value} requires --*-eps, --*-alpha, and --*-n.")
            return BIMAttackParam(epsilon=eps, alpha=alpha, iters=n, batch_size=batch_size)
        raise ValueError(f"Unsupported attack kind: {kind}")

    attack_params = get_attack_params(AttackKind(args.attack_kind), args.attack_eps, args.attack_alpha, args.attack_n, args.batch_size)
    reattack_params = get_attack_params(AttackKind(args.reattack_kind), args.reattack_eps, args.reattack_alpha, args.reattack_n, args.batch_size)

    preprocess_kind = PreprocessingKind(args.preprocess_kind)
    preprocess_params: Optional[PreprocessingParams] = None
    if preprocess_kind == PreprocessingKind.GAUSSIAN_BLUR:
        if args.preprocess_kernel_size is None or args.preprocess_sigma is None:
            parser.error("Gaussian Blur requires --preprocess-kernel-size and --preprocess-sigma.")
        preprocess_params = GaussianBlurParams(kernel_size=args.preprocess_kernel_size, sigma=args.preprocess_sigma)
    elif preprocess_kind == PreprocessingKind.MEDIAN_BLUR:
        if args.preprocess_kernel_size is None:
            parser.error("Median Blur requires --preprocess-kernel-size.")
        preprocess_params = MedianBlurParams(kernel_size=args.preprocess_kernel_size)
    elif preprocess_kind == PreprocessingKind.PIXEL_REDUCTION:
        # オフセットが指定されていなければデフォルトを使用（0.1）
        offset = args.preprocess_offset if args.preprocess_offset is not None else 0.1
        preprocess_params = PixelReductionParams(offset=offset)

    dataset = DatasetKind(args.dataset)
    device = utils.get_device()

    return Config(
        dataset=dataset, model=ModelKind(args.model),
        attack_kind=AttackKind(args.attack_kind), attack_params=attack_params,
        preprocessing_kind=preprocess_kind, preprocessing_params=preprocess_params,
        reattack_kind=AttackKind(args.reattack_kind), reattack_params=reattack_params,
        model_dir=args.model_dir, batch_size=args.batch_size, output_dir=args.output_dir,
        num_samples=args.num_samples, save_images=args.save_images,
        shuffle_dataloader=args.shuffle_dataloader, device=device, dataset_norm=DatasetNorm(dataset, device)
    )

def main():
    cfg = parse_args()
    runner = Runner(cfg)
    result_generator = runner.run()

    stats = {
        'total': 0, 'clean_correct': 0, 'attack_successful': 0, 
        'defense_successful_by_preprocess': 0,
        'defense_successful_after_reattack': 0, # 新しい指標
        'reattack_successful_from_preprocessed': 0, 
        'reattack_correct_to_original': 0
    }
    
    def format_params_str(params) -> str:
        if params is None: return "None"
        # asdictがNoneを返すことがあるため、Noneの場合は空の辞書を返す
        params_dict = asdict(params) if params is not None else {}
        return ", ".join([f"{k}: {v}" for k, v in params_dict.items() if k != 'batch_size'])

    attack_params_str = f"{cfg.attack_kind.value}, {format_params_str(cfg.attack_params)}"
    preprocess_params_str = f"{cfg.preprocessing_kind.value}, {format_params_str(cfg.preprocessing_params)}"
    reattack_params_str = f"{cfg.reattack_kind.value}, {format_params_str(cfg.reattack_params)}"

    output_folder = os.path.join(cfg.output_dir, cfg.dataset.value, cfg.model.value, attack_params_str, preprocess_params_str, reattack_params_str)
    os.makedirs(output_folder, exist_ok=True)
    
    csv_filename = os.path.join(output_folder, "results.csv")
    csv_header = ["index", "target", "pred_clean", "pred_attacked", "pred_preprocessed", "pred_reattacked", "image_path"]
    
    with concurrent.futures.ThreadPoolExecutor() as executor, open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        
        for result_tuple in result_generator:
            idx = result_tuple.index
            target = result_tuple.target_label
            pred_c = result_tuple.pred_clean
            pred_a = result_tuple.pred_attacked
            pred_p = result_tuple.pred_preprocessed
            pred_r = result_tuple.pred_reattacked
            image_ts = result_tuple.final_image

            
            stats['total'] += 1
            if pred_r == target:
                stats['reattack_correct_to_original'] += 1

            if pred_c == target:
                stats['clean_correct'] += 1
                # 初回攻撃で誤分類されたサンプルを対象
                if pred_a != target:
                    stats['attack_successful'] += 1
                    # 前処理"のみ"で正解に戻った場合
                    if pred_p == target:
                        stats['defense_successful_by_preprocess'] += 1
                    # 前処理と再攻撃を経て"最終的に"正解に戻った場合
                    if pred_r == target:
                        stats['defense_successful_after_reattack'] += 1

            if pred_p == target:
                # 前処理後に正解だったものが再攻撃で誤分類された場合
                if pred_r != target:
                    stats['reattack_successful_from_preprocessed'] += 1

            image_path = ""
            if cfg.save_images:
                image_path = os.path.join(output_folder, f"idx_{idx}_target{target}_r{pred_r}.png")
                executor.submit(utils.save_tensor_as_image, image_ts.tensor, image_path)

            writer.writerow([idx, target, pred_c, pred_a, pred_p, pred_r, image_path])

    # --- 結果表示 ---
    print("\n=== Preprocess and Re-Attack Summary ===")
    print("[実験設定]")
    print(f"  データセット: {cfg.dataset.value}, モデル: {cfg.model.value}")
    print(f"  初回攻撃: {attack_params_str}")
    print(f"  前処理: {preprocess_params_str}")
    print(f"  再攻撃: {reattack_params_str}")
    
    print("\n[結果]")
    total = stats['total']
    clean_correct = stats['clean_correct']
    attack_successful = stats['attack_successful']
    defense_by_preprocess = stats['defense_successful_by_preprocess']
    defense_after_reattack = stats['defense_successful_after_reattack']

    print(f"  処理サンプル総数: {total}")
    print(f"  クリーン精度: {clean_correct / total:.4f} ({clean_correct}/{total})")
    
    if clean_correct > 0:
        print(f"  初回攻撃成功率 (クリーン -> 誤分類): {attack_successful / clean_correct:.4f} ({attack_successful}/{clean_correct})")
    
    if attack_successful > 0:
        print(f"  防御成功率 (前処理のみ): {defense_by_preprocess / attack_successful:.4f} ({defense_by_preprocess}/{attack_successful})")
        print(f"  前処理+再攻撃による防御成功率: {defense_after_reattack / attack_successful:.4f} ({defense_after_reattack}/{attack_successful})")
    
    reattack_correct_to_original = stats['reattack_correct_to_original']
    if total > 0:
        print(f"  最終的な正解率 (全サンプル中): {reattack_correct_to_original / total:.4f} ({reattack_correct_to_original}/{total})")

    print(f"\n結果は {csv_filename} に保存されました。")
    print("========================================")


if __name__ == "__main__":
    # PIL.ImageFilter をインポート (MedianBlurで使用)
    from PIL import ImageFilter
    main()
