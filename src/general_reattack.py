# 敵対的サンプルに対する再攻撃の汎用コード
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import argparse
from copy import deepcopy
from enum import Enum
from torch import Tensor

from lib.models import MorimotoMnist, MorimotoCifar10, Ploof
from lib import attacks, utils
from lib.attacks import bim, fgsm
import foolbox


class DatasetKind(str, Enum):
    """データセットの種類"""
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    IMAGE_NET = "imagenet"


class ModelKind(str, Enum):
    """モデルの種類"""
    MORIMOTO_MNIST = "mnist"
    MORIMOTO_CIFAR10 = "cifar10"
    PLOOF = "ploof"
    INCEPTION_V3 = "inception_v3"


class AttackKind(str, Enum):
    """攻撃手法の種類"""
    BIM = "bim"
    FGSM = "fgsm"
    FOOLBOX_BIM = "foolbox_bim"


class ReAttackKind(str, Enum):
    """再攻撃手法の種類"""
    FGSM = "fgsm"
    BIM = "bim"


class PresetKind(str, Enum):
    """プリセットの種類"""
    MORIMOTO_MNIST_BIM_BIM = "morimoto_mnist_bim_bim"
    DEFAULT = "default"


# プリセット設定: 各プリセットはConfigのフィールドを設定
PRESETS = {
    PresetKind.DEFAULT: {
        # デフォルト設定
    },
    PresetKind.MORIMOTO_MNIST_BIM_BIM: {
        "dataset": DatasetKind.MNIST,
        "model": ModelKind.MORIMOTO_MNIST,
        "attack": AttackKind.BIM,
        "reattack": ReAttackKind.BIM,
        "epsilon": 0.3,
        "alpha": 0.05,
        "iters": 10,
        "reattack_epsilon": 0.3,
        "reattack_alpha": 0.05,
        "reattack_iters": 10,
        "batch_size": 1,
    },
}


@dataclass
class Config:
    """設定クラス"""
    dataset: DatasetKind = DatasetKind.MNIST
    model: ModelKind = ModelKind.MORIMOTO_MNIST
    attack: AttackKind = AttackKind.BIM
    reattack: ReAttackKind = ReAttackKind.BIM
    model_dir: str = "./weight"
    # 初回攻撃のパラメータ
    epsilon: float = 0.3
    alpha: float = 0.05
    iters: int = 10
    # 再攻撃のパラメータ
    reattack_epsilon: float = 0.3
    reattack_alpha: float = 0.05
    reattack_iters: int = 10
    batch_size: int = 1
    num_samples: Optional[int] = 10000
    device: Optional[torch.device] = None


def apply_preset(cfg: Config, preset: PresetKind) -> Config:
    """プリセットを適用した新しいConfigを返す（シャローコピー）"""
    preset_dict = PRESETS.get(preset, {})
    new_cfg = deepcopy(cfg)
    for k, v in preset_dict.items():
        if hasattr(new_cfg, k):
            setattr(new_cfg, k, v)
    return new_cfg


class ModelFactory:
    """モデル生成ファクトリー"""
    @staticmethod
    def create(kind: ModelKind, device: torch.device) -> nn.Module:
        """指定された種類のモデルを生成"""
        if kind == ModelKind.MORIMOTO_MNIST:
            return MorimotoMnist.MnistNet().to(device)
        if kind == ModelKind.MORIMOTO_CIFAR10:
            return MorimotoCifar10.Cifar10Net().to(device)
        if kind == ModelKind.PLOOF:
            return Ploof.PloofNet(10).to(device)
        if kind == ModelKind.INCEPTION_V3:
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=False, aux_logits=False)
            return model.to(device)
        raise ValueError(f"unsupported model kind: {kind}")


class DataFactory:
    """データローダー生成ファクトリー"""
    @staticmethod
    def loader(kind: DatasetKind, batch_size: int):
        """指定されたデータセットのローダーを生成"""
        if kind == DatasetKind.MNIST:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            ds = datasets.MNIST('../data', train=False, download=True, transform=transform)
        elif kind == DatasetKind.CIFAR10:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.247, 0.243, 0.261))
            ])
            ds = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
        elif kind == DatasetKind.IMAGE_NET:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])
            ds = datasets.ImageNet('../data/image_net', split='val', transform=transform)
        else:
            raise ValueError(f"unsupported dataset kind: {kind}")
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


class ReAttackRunner:
    """再攻撃実行クラス"""
    
    def __init__(self, cfg: Config):
        """初期化"""
        self.cfg = cfg
        self.device = cfg.device or utils.get_device()
        self.model = ModelFactory.create(cfg.model, self.device)
        self.test_loader = DataFactory.loader(cfg.dataset, cfg.batch_size)
        self._load_weights_if_exists(cfg.model_dir)

    def _load_weights_if_exists(self, model_dir: str) -> None:
        """学習済みモデルを読み込む"""
        import os
        path = os.path.join(model_dir, f"{self.model.model_name}.pth")
        if os.path.exists(path):
            st = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(st)
                print(f"loaded weights from {path}")
            except Exception as e:
                print(f"failed to load exact state_dict from {path}: {e}, continuing with init model")

    @staticmethod
    def _to_logits(output: Tensor) -> Tensor:
        """出力を logits に変換する（必要に応じて次元を削減）"""
        if output.dim() == 4:
            return output.mean(dim=(2, 3))
        if output.dim() > 2:
            return output.view(output.size(0), output.size(1), -1).mean(dim=2)
        return output

    def _get_mean_std(self, num_channels: int) -> Tuple[List[float], List[float]]:
        """チャネル数に基づいて正規化パラメータを取得"""
        if num_channels == 1:
            return [0.1307], [0.3081]
        else:
            return [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

    def _perform_attack(self, data: Tensor, target: Tensor, data_grad: Tensor) -> Tensor:
        """初回攻撃を実行"""
        # データを非正規化
        data_denorm = utils.denorm(data_grad, self.device)

        if self.cfg.attack == AttackKind.BIM:
            perturbed = bim.bim_attack(
                data_denorm, self.cfg.epsilon, self.cfg.alpha, self.cfg.iters,
                target, self.model, self.device
            )
        elif self.cfg.attack == AttackKind.FOOLBOX_BIM:
            # Foolbox BIM実装
            if data.dim() == 4 and data.size(1) == 1:
                preprocessing = dict(mean=[0.1307], std=[0.3081], axis=-3)
            else:
                preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261], axis=-3)
            bounds = (0.0, 1.0)
            try:
                fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing, device=self.device) # type: ignore[reportPrivateImportUsage]
                attack = foolbox.attacks.LinfBasicIterativeAttack(steps=self.cfg.iters, abs_stepsize=self.cfg.alpha) # type: ignore[reportPrivateImportUsage]
                raw, clipped, is_adv = attack(fmodel, data_denorm, target, epsilons=self.cfg.epsilon)
                perturbed = clipped if not isinstance(clipped, (list, tuple)) else clipped[0]
            except Exception as e:
                print(f"Foolbox BIM attack failed: {e}, falling back to internal BIM")
                perturbed = bim.bim_attack(
                    data_denorm, self.cfg.epsilon, self.cfg.alpha, self.cfg.iters,
                    target, self.model, self.device
                )
        else:  # FGSM
            perturbed = fgsm.fgsm_attack(
                data_denorm, self.cfg.epsilon, data_grad, self.model, self.device
            )
        
        return perturbed

    def _perform_reattack(self, perturbed: Tensor, attacked_label: Tensor) -> Tensor:
        """再攻撃を実行（攻撃されたラベルをターゲットとして使用）"""
        if self.cfg.reattack == ReAttackKind.BIM:
            reattacked = bim.bim_attack(
                perturbed, self.cfg.reattack_epsilon, self.cfg.reattack_alpha, 
                self.cfg.reattack_iters, attacked_label, self.model, self.device
            )
        else:  # FGSM
            reattacked = fgsm.fgsm_attack(
                perturbed, self.cfg.reattack_epsilon, attacked_label, 
                self.model, self.device
            )
        
        return reattacked

    def run(self) -> List[Tuple[int, int, int, int, Tensor, Tensor, Tensor]]:
        """
        再攻撃を実行する
        
        戻り値:
            List of (index, true_label, attacked_pred, reattacked_pred, 
                     clean_data, attacked_data, reattacked_data)
        """
        print(f"Running attack {self.cfg.attack} -> reattack {self.cfg.reattack}")
        print(f"Config: {self.cfg}")
        
        self.model.eval()
        results = []

        for i, (data, target) in enumerate(self.test_loader):
            if self.cfg.num_samples != None and i >= self.cfg.num_samples:
                break
            
            data = data.to(self.device)
            target = target.to(self.device).view(-1).long()
            data.requires_grad = True

            # ステップ1: クリーンデータでの推論
            output = self.model(data)
            logits = self._to_logits(output)
            pred_clean = logits.max(1, keepdim=True)[1]

            # 勾配を計算
            loss = F.cross_entropy(logits, target)
            self.model.zero_grad()
            loss.backward()
            grad = data.grad
            
            if grad is None:
                print(f"skip idx {i}: no grad")
                continue
            
            data_grad = grad.data

            # ステップ2: 初回攻撃
            perturbed = self._perform_attack(data, target, data_grad)
            
            # 攻撃後の推論（正規化して推論）
            num_channels = perturbed.size(1)
            mean, std = self._get_mean_std(num_channels)
            if num_channels == 1:
                perturbed_norm = transforms.Normalize(mean, std)(perturbed)
            else:
                perturbed_norm = transforms.Normalize(mean, std)(perturbed)

            out_attacked = self.model(perturbed_norm)
            logits_attacked = self._to_logits(out_attacked)
            pred_attacked = logits_attacked.max(1, keepdim=True)[1]

            # ステップ3: 再攻撃（攻撃されたラベルをターゲットとして使用）
            reattacked = self._perform_reattack(perturbed, pred_attacked.view(-1))

            # 再攻撃後の推論
            if num_channels == 1:
                reattacked_norm = transforms.Normalize(mean, std)(reattacked)
            else:
                reattacked_norm = transforms.Normalize(mean, std)(reattacked)

            out_reattacked = self.model(reattacked_norm)
            logits_reattacked = self._to_logits(out_reattacked)
            pred_reattacked = logits_reattacked.max(1, keepdim=True)[1]

            # 結果を保存
            results.append((
                i,
                target.item(),
                pred_attacked.item(),
                pred_reattacked.item(),
                data.squeeze().detach().cpu(),
                perturbed.squeeze().detach().cpu(),
                reattacked.squeeze().detach().cpu()
            ))

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} samples")

        return results


def parse_args() -> Config:
    """コマンドライン引数をパースしてConfigを返す"""
    parser = argparse.ArgumentParser(description="general re-attack runner")
    parser.add_argument("--dataset", choices=[d.value for d in DatasetKind], 
                        default=DatasetKind.MNIST.value,
                        help="データセットの種類")
    parser.add_argument("--model", choices=[m.value for m in ModelKind], 
                        default=ModelKind.MORIMOTO_MNIST.value,
                        help="モデルの種類")
    parser.add_argument("--attack", choices=[a.value for a in AttackKind], 
                        default=AttackKind.BIM.value,
                        help="初回攻撃手法")
    parser.add_argument("--reattack", choices=[r.value for r in ReAttackKind], 
                        default=ReAttackKind.BIM.value,
                        help="再攻撃手法")
    parser.add_argument("--preset", choices=[p.value for p in PresetKind], 
                        default=None,
                        help="プリセット設定を適用")
    parser.add_argument("--model-dir", default="./weight",
                        help="学習済みモデルのディレクトリ")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="初回攻撃の epsilon")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="初回攻撃の alpha")
    parser.add_argument("--iters", type=int, default=10,
                        help="初回攻撃の反復回数")
    parser.add_argument("--reattack-epsilon", type=float, default=0.3,
                        help="再攻撃の epsilon")
    parser.add_argument("--reattack-alpha", type=float, default=0.05,
                        help="再攻撃の alpha")
    parser.add_argument("--reattack-iters", type=int, default=10,
                        help="再攻撃の反復回数")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="バッチサイズ")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="処理するサンプル数")
    
    args = parser.parse_args()
    
    cfg = Config(
        dataset=DatasetKind(args.dataset),
        model=ModelKind(args.model),
        attack=AttackKind(args.attack),
        reattack=ReAttackKind(args.reattack),
        model_dir=args.model_dir,
        epsilon=args.epsilon,
        alpha=args.alpha,
        iters=args.iters,
        reattack_epsilon=args.reattack_epsilon,
        reattack_alpha=args.reattack_alpha,
        reattack_iters=args.reattack_iters,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=utils.get_device()
    )

    if args.preset:
        cfg = apply_preset(cfg, PresetKind(args.preset))
    
    return cfg


def main() -> None:
    """メイン関数"""
    cfg = parse_args()
    runner = ReAttackRunner(cfg)
    results = runner.run()

    # 統計情報を計算
    total = len(results)
    if total == 0:
        print("No results to analyze")
        return

    # 正解率の計算
    correct_before_attack = 0
    attack_success = 0
    reattack_success = 0
    
    for idx, true_label, attacked_pred, reattacked_pred, _, _, _ in results:
        # 最初から正しく分類できていたか（攻撃前の正解率は別途必要だが、ここでは省略）
        # 攻撃が成功したか（ラベルが変わったか）
        if attacked_pred != true_label:
            attack_success += 1
        # 再攻撃が成功したか（元のラベルに戻ったか）
        if reattacked_pred == true_label:
            reattack_success += 1

    attack_success_rate = (attack_success / total) if total > 0 else 0.0
    reattack_success_rate = (reattack_success / total) if total > 0 else 0.0

    print("\n" + "="*60)
    print("実験結果サマリー")
    print("="*60)
    print(f"総サンプル数: {total}")
    print(f"初回攻撃成功率: {attack_success_rate:.2%} ({attack_success}/{total})")
    print(f"再攻撃成功率（正解への復元）: {reattack_success_rate:.2%} ({reattack_success}/{total})")
    print("="*60)
    
    # 詳細結果の一部を表示
    print("\n最初の10サンプルの詳細:")
    print(f"{'Index':<6} {'True':<5} {'Attacked':<9} {'Reattacked':<11}")
    print("-" * 35)
    for idx, true_label, attacked_pred, reattacked_pred, _, _, _ in results[:10]:
        print(f"{idx:<6} {true_label:<5} {attacked_pred:<9} {reattacked_pred:<11}")


if __name__ == "__main__":
    main()
