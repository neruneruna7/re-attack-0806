# CIFAR10 の先頭 N 画像を読み込み保存する簡易スクリプト
# 日本語コメント・main エントリを備え、uv run で起動可能
from typing import Optional
import os
import argparse

import torch
from torchvision import datasets, transforms

from re_attack_0806 import utils


def save_cifar10_first_n(output_dir: str, n: int = 10) -> None:
    """CIFAR10 テストセットの先頭 n 画像を `output_dir` に保存する。

    - output_dir: 画像保存先ディレクトリ
    - n: 保存する画像枚数（先頭から）
    """
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(n, len(ds))):
        img, label = ds[i]
        # img: Tensor[C,H,W] 値域 0..1
        filename = os.path.join(output_dir, f"cifar10_idx_{i}_label_{label}.png")
        utils.save_tensor_as_image(img, filename)
        print(f"saved: {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save first N CIFAR10 images")
    parser.add_argument("--out", default="outputs/cifar10_first10", help="Output directory")
    parser.add_argument("--n", type=int, default=10, help="Number of images to save (default 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    save_cifar10_first_n(args.out, args.n)


if __name__ == '__main__':
    main()
