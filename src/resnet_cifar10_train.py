import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm

def main():
    # 1. デバイスの設定（GPUが使えるならGPU、MacならMPS、それ以外はCPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 前処理の定義（データ拡張と正規化）
    # CIFAR-10の平均と標準偏差で正規化を行います
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # 3. データセットとデータローダーの準備
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=2)

    # 4. モデルの定義とCIFAR-10向けへの修正
    # 事前学習なし(weights=None)で初期化します
    model = resnet18(weights=None)

    # 【重要】CIFAR-10(32x32)用に最初の畳み込み層を変更
    # 元の7x7, stride 2だと画像サイズが小さくなりすぎるため、3x3, stride 1に変更
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # MaxPoolもCIFAR-10では情報損失が大きいため無効化（Identity）する
    model.maxpool = nn.Identity() # pyright: ignore[reportAttributeAccessIssue]
    # 全結合層の出力数をCIFAR-10のクラス数(10)に変更
    model.fc = nn.Linear(model.fc.in_features, 10)

    model = model.to(device)

    # 5. 損失関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 6. 学習ループ
    num_epochs = 100 # デモ用に短く設定しています（実用なら100-200推奨）
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in tqdm(enumerate(trainloader),  f"Epoch {epoch+1}/{num_epochs}", total=len(trainloader)):
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

        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(trainloader):.4f} | Acc: {train_acc:.2f}%")

    PATH = './weight/resnet18_cifar10.pth'
    torch.save(model.state_dict(), PATH)
    print("Finished Training and Saved Model")
    # 7. テストセットでの評価
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total:.2f} %')

if __name__ == '__main__':
    main()