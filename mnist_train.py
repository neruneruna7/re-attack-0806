from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from models.my_mnist import MnistCNN

def train_mnist():
    # データセットと前処理
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # モデル・オプティマイザ定義
    model = MnistCNN()
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 学習ループ（簡易）
    for epoch in range(1, 100):  # 論文は99エポック
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch}/99], Loss: {loss.item():.4f}')

    print("Training completed.")
    return model
