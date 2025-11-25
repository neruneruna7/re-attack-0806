import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    # 1. データセットの準備
    # ここでは正規化を行わず、単に0.0-1.0のTensorに変換するだけにします
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )

    # 2. データローダーで全データを一度に取得
    # CIFAR-10はサイズが小さいので、全データを1つのバッチとして読み込めます
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    
    # データを取得（imagesの形状は [50000, 3, 32, 32]）
    images, _ = next(iter(loader))

    # 3. 平均と標準偏差の計算
    # バッチ(0)、高さ(2)、幅(3)の次元を潰して、チャンネル(1)ごとの統計を計算します
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])

    print("--- 計算結果 ---")
    print(f"Mean: {mean}")
    print(f"Std : {std}")

if __name__ == '__main__':
    main()