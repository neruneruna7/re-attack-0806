import torch
import mnist_train
import cifar10_train

def main():
    # pythochで，metalGPUが使えているか確認する
    if torch.backends.mps.is_available():
        print("Metal GPU is available.")
    else:
        print("Metal GPU is not available.")
    print("Hello from re-attack-0806!")

    # MNISTの学習を実行
    # model = mnist_train.train_mnist()
    model = cifar10_train.train_cifar10()

    # モデルの保存
    torch.save(model.state_dict(), 'weight/cifar10_cnn.pth')


if __name__ == "__main__":
    main()
