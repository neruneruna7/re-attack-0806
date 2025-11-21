# モデルをトレーニングする汎用コード
from dataclasses import dataclass
from torch.types import Number
from typing import Any, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import argparse
from torch import Tensor

from re_attack_0806 import utils
from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm, AttackParams, FGSMAttackParam, BIMAttackParam
from re_attack_0806.utils.normTensor import *
from re_attack_0806.strategies.factory import AttackStrategyFactory
from re_attack_0806.result_processor import ResultProcessor, AttackResult
from re_attack_0806.path_manager import PathManager

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
    """攻撃実行を統括するクラス
    
    このクラスは、モデルの読み込み、データローダーの準備、
    攻撃の実行を担当します。ストラテジーパターンにより、
    具体的な攻撃手法の実装は各ストラテジークラスに委譲されています。
    """
    
    def __init__(self, cfg: Config):
        """Runnerの初期化
        
        Args:
            cfg: 実行設定
        """
        self.cfg = cfg
        self.device = cfg.device
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, train=False, batch_size=cfg.batch_size, shuffle=cfg.shuffle_dataloader)
        self._load_weights_if_exists(cfg.model_dir)
        self.model.eval()
        
        # 攻撃ストラテジーの取得
        self.attack_strategy = AttackStrategyFactory.create(cfg.attack)

    def _load_weights_if_exists(self, model_dir: str):
        """モデルの重みファイルが存在する場合は読み込む
        
        Args:
            model_dir: モデルの重みが保存されているディレクトリ
        """
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
        """モデル出力をロジットに変換
        
        Args:
            output: モデルの出力テンソル
            
        Returns:
            ロジットテンソル
        """
        if output.dim() == 4:
            return output.mean(dim=(2, 3))
        if output.dim() > 2:
            return output.view(output.size(0), output.size(1), -1).mean(dim=2)
        return output
    
    def run(self) -> Iterator[Tuple[int, Number, Number, Number, TensorWithState, Number]]:
        """攻撃を実行してイテレータを返す
        
        ストラテジーパターンにより、攻撃手法の具体的な実装は
        各ストラテジークラスに委譲されています。
        
        Yields:
            (index, target_label, pred_before, pred_after, perturbed_image, l2_perturbation)
        """
        print(f"Running attack {self.cfg.attack} on model {self.cfg.model} with cfg: {self.cfg}")

        global_idx = 0
        for data, target in tqdm(self.test_loader, desc="Attacking"):
            if self.cfg.num_samples != -1 and global_idx >= self.cfg.num_samples:
                break
            
            current_batch_size = data.shape[0]
            data_ts = TensorWithState(data.to(self.device), DENORMALIZED)
            data_ts_norm = self.cfg.dataset_norm.normalize(data_ts)
            target_ts: Tensor = target.to(self.device).view(-1).long()
            
            # クリーン画像での予測
            output = self.model(data_ts_norm.tensor)
            logits = self._to_logits(output)
            pred = logits.max(1, keepdim=True)[1]

            # ストラテジーパターンを使用して攻撃を実行
            try:
                perturbed = self.attack_strategy.execute(
                    data_ts_norm,
                    target_ts,
                    self.model,
                    self.device,
                    self.cfg.dataset_norm,
                    self.cfg.attack_params
                )
            except Exception as e:
                print(f"攻撃の実行に失敗しました: {e}")
                continue

            # 攻撃後の予測
            out2 = self.model(perturbed.tensor)
            logits2 = self._to_logits(out2)
            pred2 = logits2.max(1, keepdim=True)[1]

            # 摂動量の計算
            denormalized_data = self.cfg.dataset_norm.denormalize(data_ts_norm)
            denormalized_perturbed = self.cfg.dataset_norm.denormalize(perturbed)
            
            perturbation_batch = denormalized_perturbed.tensor - denormalized_data.tensor
            l2_perturbations = torch.linalg.norm(perturbation_batch.view(current_batch_size, -1), ord=2, dim=1)

            # バッチ内の各サンプルについて結果を yield
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
    """メイン関数
    
    攻撃の実行、結果の処理、サマリーの出力を行います。
    PathManagerとResultProcessorを使用することで、
    責務が明確に分離されています。
    """
    cfg = parse_args()
    
    runner = Runner(cfg)
    result_generator = runner.run()

    # パス管理の一元化
    path_manager = PathManager(cfg)
    output_folder = path_manager.ensure_output_folder_exists()
    csv_filepath = path_manager.get_csv_filepath()
    
    # ResultProcessorを使用して結果を処理
    with ResultProcessor(csv_filepath, output_folder, cfg.save_attacked_images) as processor:
        for idx, target, before, after, perturbed_image, l2_perturb in result_generator:
            result = AttackResult(
                index=idx,
                target_label=target,
                prediction_before_attack=before,
                prediction_after_attack=after,
                l2_perturbation=l2_perturb
            )
            processor.process_result(result, perturbed_image)
        
        # サマリーの取得と出力
        summary = processor.get_summary()
    
    ResultProcessor.print_summary(
        summary,
        cfg.dataset.value,
        cfg.model.value,
        cfg.attack.value,
        cfg.attack_params,
        csv_filepath
    )

if __name__ == "__main__":
    main()