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
from re_attack_0806.utils.config import AttackKind, DataFactory, DatasetKind, ModelFactory, ModelKind, DatasetNorm, AttackParams, FGSMAttackParam, BIMAttackParam, AttackParamsFactory
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
    """コマンドライン引数を解析してConfigを生成
    
    サブパーサーを使用することで、各攻撃手法に特化した引数を
    直感的に指定できるようになっています。
    
    例:
        python general_ae.py fgsm --dataset mnist --model morimoto-mnist --epsilon 0.3
        python general_ae.py bim --dataset mnist --model morimoto-mnist --epsilon 0.3 --alpha 0.05 --n 10
    
    Returns:
        パース済みの設定オブジェクト
    """
    parser = argparse.ArgumentParser(
        description="敵対的サンプル生成のための汎用攻撃ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # FGSM攻撃でMNISTデータセットに対して攻撃
  %(prog)s fgsm --dataset mnist --model morimoto-mnist --epsilon 0.3
  
  # BIM攻撃でCIFAR-10データセットに対して攻撃（10サンプルのみ）
  %(prog)s bim --dataset cifar10 --model morimoto-cifar10 --epsilon 0.03 --alpha 0.01 --n 10 --num-samples 10
  
  # Foolbox FGSM攻撃で画像を保存
  %(prog)s foolbox-fgsm --dataset mnist --model morimoto-mnist --epsilon 0.3 --save-images
"""
    )
    
    # サブパーサーの作成
    subparsers = parser.add_subparsers(
        dest='attack',
        required=True,
        help='使用する攻撃手法'
    )
    
    # 共通引数を追加するヘルパー関数
    def add_common_arguments(subparser: argparse.ArgumentParser):
        """すべての攻撃手法に共通の引数を追加"""
        subparser.add_argument(
            "--dataset",
            choices=[d.value for d in DatasetKind],
            required=True,
            help="使用するデータセット"
        )
        subparser.add_argument(
            "--model",
            choices=[m.value for m in ModelKind],
            required=True,
            help="使用するモデル"
        )
        subparser.add_argument(
            "--model-dir",
            default=DEFAULT_MODEL_DIR,
            help=f"モデルの重みファイルが保存されているディレクトリ (デフォルト: {DEFAULT_MODEL_DIR})"
        )
        subparser.add_argument(
            "--output-dir",
            default=DEFAULT_OUTPUT_DIR,
            help=f"攻撃結果の画像を保存するディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})"
        )
        subparser.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help=f"バッチサイズ (デフォルト: {DEFAULT_BATCH_SIZE})"
        )
        subparser.add_argument(
            "--num-samples",
            type=int,
            default=DEFAULT_NUM_SAMPLES,
            help="処理するサンプル数。-1ですべて処理 (デフォルト: -1)"
        )
        subparser.add_argument(
            '--save-images',
            action=argparse.BooleanOptionalAction,
            default=False,
            help='攻撃後の画像を保存するかどうか (デフォルト: False)'
        )
        subparser.add_argument(
            "--shuffle-dataloader",
            action="store_true",
            default=DEFAULT_SHUFFLE_DATALOADER,
            help=f"データローダーをシャッフルする (デフォルト: {DEFAULT_SHUFFLE_DATALOADER})"
        )
    
    # FGSM攻撃のサブパーサー
    fgsm_parser = subparsers.add_parser(
        'fgsm',
        help='FGSM (Fast Gradient Sign Method) 攻撃',
        description='FGSM攻撃は、勾配の符号を利用した高速な敵対的サンプル生成手法です。'
    )
    add_common_arguments(fgsm_parser)
    fgsm_parser.add_argument(
        "--epsilon",
        type=fraction_float,
        required=True,
        help="摂動の大きさ（最大ピクセル値の変化量）"
    )
    
    # BIM攻撃のサブパーサー
    bim_parser = subparsers.add_parser(
        'bim',
        help='BIM (Basic Iterative Method) 攻撃',
        description='BIM攻撃は、FGSMを複数回反復して実行する攻撃手法です。'
    )
    add_common_arguments(bim_parser)
    bim_parser.add_argument(
        "--epsilon",
        type=fraction_float,
        required=True,
        help="摂動の最大値（L∞ノルム）"
    )
    bim_parser.add_argument(
        "--alpha",
        type=fraction_float,
        required=True,
        help="各反復でのステップサイズ"
    )
    bim_parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="反復回数"
    )
    
    # Foolbox FGSM攻撃のサブパーサー
    foolbox_fgsm_parser = subparsers.add_parser(
        'foolbox-fgsm',
        help='Foolbox FGSM攻撃',
        description='Foolboxライブラリを使用したFGSM攻撃です。'
    )
    add_common_arguments(foolbox_fgsm_parser)
    foolbox_fgsm_parser.add_argument(
        "--epsilon",
        type=fraction_float,
        required=True,
        help="摂動の大きさ（最大ピクセル値の変化量）"
    )
    
    # Foolbox BIM攻撃のサブパーサー
    foolbox_bim_parser = subparsers.add_parser(
        'foolbox-bim',
        help='Foolbox BIM攻撃',
        description='Foolboxライブラリを使用したBIM攻撃です。'
    )
    add_common_arguments(foolbox_bim_parser)
    foolbox_bim_parser.add_argument(
        "--epsilon",
        type=fraction_float,
        required=True,
        help="摂動の最大値（L∞ノルム）"
    )
    foolbox_bim_parser.add_argument(
        "--alpha",
        type=fraction_float,
        required=True,
        help="各反復でのステップサイズ"
    )
    foolbox_bim_parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="反復回数"
    )
    
    # L∞ BIM攻撃のサブパーサー
    linf_bim_parser = subparsers.add_parser(
        'linf-bim',
        help='L∞ BIM攻撃',
        description='L∞ノルム制約付きBIM攻撃です。'
    )
    add_common_arguments(linf_bim_parser)
    linf_bim_parser.add_argument(
        "--epsilon",
        type=fraction_float,
        required=True,
        help="摂動の最大値（L∞ノルム）"
    )
    linf_bim_parser.add_argument(
        "--alpha",
        type=fraction_float,
        required=True,
        help="各反復でのステップサイズ"
    )
    linf_bim_parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="反復回数"
    )
    
    args = parser.parse_args()

    # 攻撃種別に基づいてAttackParamsを生成（ファクトリーを使用）
    attack_kind = AttackKind(args.attack)
    attack_params = AttackParamsFactory.create_from_args(
        attack_kind=attack_kind,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        alpha=getattr(args, 'alpha', None),
        iters=getattr(args, 'n', None)
    )

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