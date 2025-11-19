import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np

from re_attack_0806 import utils

def main():
    # 1. デバイスの設定
    device = utils.get_device()
    print(f"Using device: {device}")

    # 2. モデルの準備 (ここではResNet18を使用)
    # ※注意: CIFAR-10用に調整・学習済みの重みをロードするのが一般的ですが、
    # ここでは動作デモのため、未学習(ランダム重み)のモデルを少し調整して使用します。
    model = torchvision.models.resnet18(weights=None)
    
    # CIFAR-10は画像サイズが小さい(32x32)ため、ResNetの最初の畳み込み層を調整
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # 小さい画像なのでプーリングをスキップ
    model.fc = nn.Linear(512, 10) # 10クラス分類に変更
    
    model = model.to(device).eval() # 必ずevalモードにする

    # 3. データセットの読み込み
    # 重要: ここでは ToTensor() のみを行い、Normalizeは行わない！ (値域は 0.0 - 1.0)
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    # バッチを1つ取り出す
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    # 4. Foolboxモデルの作成 (ここで正規化を設定)
    # CIFAR-10の標準的な平均・標準偏差
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std  = [0.2023, 0.1994, 0.2010]
    
    preprocessing = dict(mean=cifar10_mean, std=cifar10_std, axis=-3)

    # bounds=(0, 1) と preprocessingを指定してラップする
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing, device=device)

    # 5. 攻撃の実行 (BIM - LinfBasicIterativeAttack)
    print("Starting Attack...")
    
    # epsilon: 画素空間(0-1)での許容摂動量 (例: 8/255)
    epsilon = 8 / 255
    steps = 10
    
    attack = fb.attacks.LinfBasicIterativeAttack(steps=steps)
    
    # raw: 攻撃結果(AE), clipped: クリッピング済みのAE, is_adv: 攻撃成功フラグ
    raw_advs, clipped_advs, is_adv = attack(fmodel, images, labels, epsilons=epsilon)

    delta = clipped_advs - images
    l2_norms = torch.norm(delta, dim=(1, 2, 3), p=2)
    average_perturbation = torch.mean(l2_norms).item()
    print(f"Average L2 norm of perturbations: {average_perturbation:.4f}")


    # 6. 結果の確認
    # 攻撃前の精度
    clean_acc = fb.utils.accuracy(fmodel, images, labels)
    # 攻撃後の精度 (1.0 - 攻撃成功率)
    adv_acc = fb.utils.accuracy(fmodel, clipped_advs, labels)

    print(f"Clean Accuracy: {clean_acc * 100:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc * 100:.2f}%")
    print(f"Attack Success Rate: {(1 - adv_acc) * 100:.2f}%")

    # # 7. 画像の可視化 (最初の1枚)
    # # テンソルをCPUに戻してnumpy化
    # img_clean = images[0].cpu().permute(1, 2, 0).numpy()
    # img_adv = clipped_advs[0].cpu().permute(1, 2, 0).numpy()
    # perturbation = (img_adv - img_clean)
    
    # # 表示用に摂動を見やすくブースト (x50)
    # perturbation_vis = np.abs(perturbation) * 50 
    # perturbation_vis = np.clip(perturbation_vis, 0, 1)

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # axes[0].imshow(img_clean); axes[0].set_title("Original")
    # axes[1].imshow(perturbation_vis); axes[1].set_title("Perturbation (x50)")
    # axes[2].imshow(img_adv); axes[2].set_title("Adversarial (BIM)")
    # plt.show()

if __name__ == "__main__":
    main()