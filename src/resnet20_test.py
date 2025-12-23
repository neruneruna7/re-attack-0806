import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from re_attack_0806.models import MorimotoCifar10, MorimotoMnist, Ploof, KumagaiResnet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# prepare test data
test_data = torchvision.datasets.CIFAR10(
    '../data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
        ]))
test_loader = torch.utils.data.DataLoader(test_data,  batch_size=1,  shuffle=False)

def count_params(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params

# test
model = KumagaiResnet.resnet20()
model.load_state_dict(torch.load("weight/resnet20.ckpt", map_location=torch.device(device))['state_dict'])
model = model.to(device)
print("parameters:", count_params(model))

model.eval()
correct = 0
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        _, pred = torch.max(outputs.data, 1)
        correct += (pred == labels).sum().item()

acc = float(correct / len(test_data))
print("test acc (Top-1): " + str(acc))