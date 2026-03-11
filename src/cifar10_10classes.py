import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # 1. CIFAR-10 データセットの読み込み設定
    # 表示用なので正規化（Normalize）は行わず、Tensor変換のみを行う
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # データセットのダウンロードとロード（カレントディレクトリの data フォルダに保存）
    # train=True (学習用データ) を使用
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)

    # CIFAR-10 のクラス定義（順序は固定）
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. 各クラスの画像を1枚ずつ収集するロジック
    # クラスID(0-9) をキー、画像データを値とする辞書を作成
    class_images = {}
    
    # データセットを走査して、全10クラスの画像が揃うまでループする
    # ランダムアクセスではなく、先頭から順に探索する
    for img, label in dataset:
        if label not in class_images:
            class_images[label] = img
        
        # 全10クラス分集まったらループを終了
        if len(class_images) == 10:
            break

    # 3. 画像のグリッド表示
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("CIFAR-10 Dataset Samples", fontsize=16)

    # クラスID順（0から9）に並べて表示
    for i in range(10):
        class_name = classes[i]
        img_tensor = class_images[i]

        # Tensor (C, H, W) を NumPy (H, W, C) に変換
        img_np = img_tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        # サブプロットの作成 (2行5列)
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(img_np)
        ax.set_title(f"{class_name} (ID: {i})")
        ax.axis('off') # 軸目盛りを非表示にする

    # レイアウトを調整して表示
    plt.tight_layout()
    # 変更点】画面表示ではなくファイルに保存する
    output_filename = "cifar10_samples.png"
    # bbox_inches='tight' は余白を自動でトリミングして保存するオプション
    # dpi=300 は解像度を指定（高画質化）
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    
    # メモリ解放（大量の画像を処理する場合に重要）
    plt.close(fig)

    print(f"画像を作成し、'{output_filename}' として保存しました。")
    print(f"保存先: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()