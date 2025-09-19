from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch import Tensor

from attacks import fgsm_attack

# Set random seed for reproducibility
torch.manual_seed(42)
out_dir = "data/attacked_images"

# ---- small utilities ----
def get_device() -> torch.device:
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def create_filename(epsilon: float, index: int, true_label: int) -> str:
    return f"{out_dir}/eps_{epsilon:.3f}/reattacked_{index}_label_{true_label}.png"

def folder_name(epsilon: float) -> str:
    return f"{out_dir}/eps_{epsilon:.3f}/"

def from_filename(filename: str) -> Tuple[int, int]:
    """
    filename: attacked_12_label_3.png
    戻り値: (12, 3)
    """
    parts = filename.split("_")
    if len(parts) < 4:
        raise ValueError(f"invalid filename format: {filename}")
    try:
        index = int(parts[1])
        label_part = parts[3]
        label_str = label_part.split(".")[0]  # "3.png" -> "3"
        label = int(label_str)
        return index, label
    except Exception as e:
        raise ValueError(f"invalid filename format: {filename}") from e


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # print("forward")
        # print("1", x.shape)
        x = self.conv1(x)
        # print("conv1", x.shape)
        x = F.relu(x)
        # print("conv1_relu", x.shape)

        x = self.conv2(x)
        # print("conv2", x.shape)

        x = F.relu(x)
        # print("conv2_relu", x.shape)

        x = F.max_pool2d(x, 2)
        # print("max_pool", x.shape)

        x = self.dropout1(x)
        # print("dropout1", x.shape)

        x = torch.flatten(x, 1)
        # print("flatten", x.shape)

        x = self.fc1(x)
        # print("fc1", x.shape)

        x = F.relu(x)
        # print("fc1_relu", x.shape)

        x = self.dropout2(x)
        # print("dropout2", x.shape)

        x = self.fc2(x)
        # print("fc2", x.shape)
        # print("")


        output = F.log_softmax(x, dim=1)
        return output


# restores the tensors to their original scale
def denorm(batch, device, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    # print("batch_shape", batch.shape) 
    # #バッチは４次元

    # print("mean_shape", mean.shape)
    # # 1次元
    # print(mean.view(1, -1, 1, 1).shape)
    # # 4次元

    # print("std_shape", std.shape)
    # # 1次元
    # print(std.view(1, -1, 1, 1).shape)
    # # 4次元
    # print("")

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def save_tensor_as_image(tensor: Tensor, path: str):
    """
    tensor: [B,C,H,W] or [C,H,W] or [H,W], values in 0..1 (非正規化)
    path: output png path
    """
    t = tensor.clone().detach().cpu()
    # squeeze batch/channel dims if needed
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3 and t.size(0) == 1:
        arr = t.squeeze(0).numpy()
    elif t.dim() == 3 and t.size(0) == 3:
        # convert CHW -> HWC
        arr = t.permute(1, 2, 0).numpy()
    elif t.dim() == 2:
        arr = t.numpy()
    else:
        arr = t.numpy()

    # clip and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    # create parent dir
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = Image.fromarray(arr_u8)
    img.save(path)
    info = f"saved image {path}"
    print(info)

def iterative_reattack(perturbed_denorm: Tensor, model: nn.Module, target: Tensor, device: str, step_epsilon: float, steps: int) -> Tensor:
    """
    perturbed_denorm: 非正規化されたテンソル (batch, channel, H, W), 値域 [0,1]
    step_epsilon: 1ステップあたりの摂動量（ピクセルスケール）
    steps: 繰り返す回数
    戻り値: 最終的に得られる非正規化テンソル（clamped, detach済み）
    """
    # ensure target has batch dimension (shape [B])
    if target.dim() == 0:
        target = target.unsqueeze(0)

    # start from given perturbed image (非正規化)
    adv = perturbed_denorm.clone().detach()

    # 繰り返しで再攻撃：各ステップで fgsm_attack を呼ぶ
    for _ in range(steps):
        adv = fgsm_attack(adv, step_epsilon, target, model, device).detach()

    return adv

def test( model, device, test_loader, epsilon ):
    # Accuracy counter
    correct = 0
    adv_examples = []
    target_list = []
    count = 0

    # Loop over all examples in test set
    for i, (data, target) in enumerate(test_loader):
        target_list.append(target)
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if init_pred.item() != target.item():
            continue

        data_denorm = denorm(data, device)
        perturbed_data = fgsm_attack(data_denorm, epsilon, target, model, device)

        # 最初の FGSM による予測（これが final_pred）
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        output = model(perturbed_data_normalized)
        final_pred = output.max(1, keepdim=True)[1]  # shape [B,1]

        # # --- ここで final_pred をターゲットとして再攻撃する ---
        # # 例: 1ステップで epsilon を使って再攻撃する場合
        # reattack_steps = 1
        # step_eps = epsilon / reattack_steps if reattack_steps > 0 else 0.0

        # # final_pred を適切な形 [B] の long tensor にする
        # final_label = final_pred.view(-1).to(device).long()

        # # perturbed_data は非正規化（0..1）を想定しているのでそのまま渡す
        # perturbed_data = iterative_reattack(perturbed_data, model, final_label, device, step_eps, reattack_steps)
        save_tensor_as_image(perturbed_data, create_filename(epsilon, i, target.item()) )

        # # 再ノーマライズして再分類
        # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        # output = model(perturbed_data_normalized)

        # Check for success（再攻撃後の予測を使う）
        # final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples


def main():
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    pretrained_model = "data/lenet_mnist_model.pth"

    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
    device = get_device()
    print(f"Using {device} device")

    # Initialize the network
    model = Net().to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location=device, weights_only=True))

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()



    accuracies = []
    examples = []

    print("ざぁこざぁこ")
    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)


# mainとして呼ばれたときのおまじないのあれ
if __name__ == "__main__":
    main()