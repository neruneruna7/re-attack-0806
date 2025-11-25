import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from re_attack_0806 import utils
from re_attack_0806.models import MorimotoMnist, MorimotoCifar10


# ==========================================
# 2. 学習設定 (論文 表4準拠)
# ==========================================
def main():
    # デバイス設定
    device = utils.get_device()
    print(f"Using device: {device}")

    # 論文 表4のパラメータ
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2  # 0.01 (論文の記述通り)
    EPOCHS = 99           # 論文の記述通り
    
    # 【重要】論文には明記がないが、Adam(lr=0.01)でのロジット爆発と過学習を防ぐために設定
    # これがないと前回のようにConfidenceが1.0に張り付き、正常な追試ができません。
    # WEIGHT_DECAY = 1e-4

    # データセットの準備 (CIFAR-10)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # 論文の精度(82.6%)を再現するための標準的なAugmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # モデル構築
    # model = Cifar10Net().to(device)
    model = MorimotoCifar10.Cifar10Net().to(device)

    # Optimizer設定 (論文条件: Adam)
    # 安定学習のためにWeight Decayを追加
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学習率スケジューラ (オプション: 論文にはないが収束を安定させるため推奨)
    # 論文通りの固定LRで行く場合はscheduler.step()を削除してください
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    print(f"Start Training: Adam(lr={LEARNING_RATE}), Epochs={EPOCHS}, Batch={BATCH_SIZE}")

    # ==========================================
    # 3. 学習ループ
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # scheduler.step()

        # ログ出力
        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(trainloader):.4f} | Acc: {train_acc:.2f}%")

        # 簡易的なロジット監視（爆発防止チェック）
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                print(f"   Debug Logits Max: {outputs.max().item():.2f}")

    # ==========================================
    # 4. 評価と保存
    # ==========================================
    print("Evaluating on Test Set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}% (Paper Target: 82.60%)")

    # モデル保存
    torch.save(model.state_dict(), "./weight/cifar10_paper_reproduction.pth")
    print("Model saved to ./weight/cifar10_paper_reproduction.pth")

if __name__ == "__main__":
    main()