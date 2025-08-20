from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch
import torch
import torchvision.models as models

from models.my_mnist import MnistCNN

def train_mnist():
    # データセットと前処理
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # デバイス選定：MPS → CUDA → CPU の順
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # モデル・オプティマイザ定義
    # model = MnistCNN().to(device)
    model = models.resnet18(pretrained=False, num_classes=10)
    # 1ch入力に対応させる
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 小さい画像なので maxpool をスキップ（任意）
    model.maxpool = nn.Identity()

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # 学習ループ（簡易）
    for epoch in range(1, 100):  # 論文は99エポック
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch}/99], Loss: {loss.item():.4f}')

    print("Training completed.")
    return model
