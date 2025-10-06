
from lib import utils
from lib.models import MorimotoMnist, MorimotoCifar10

def main():
    device = utils.get_device()
    # model = MorimotoMnist.MnistNet().to(device)
    model = MorimotoCifar10.Cifar10Net().to(device)
    print(model)

if __name__ == "__main__":
    main()