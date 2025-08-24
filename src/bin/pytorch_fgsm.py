import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
# Set random seed for reproducibility
torch.manual_seed(42)

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

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device, weights_only=True))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

# FGSM attack code
def fgsm_attack(image: Tensor, epsilon: float, target: Tensor) -> Tensor:
    """
    image: 非正規化されたテンソル [B,C,H,W] (値域 0..1)
    epsilon: ピクセルスケールの摂動量
    target: 正解ラベルテンソル (device 上)
    戻り値: perturbed_image (非正規化, clamp され detach 済み)
    """
    # mean/std on device
    mean = torch.tensor([0.1307], dtype=image.dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor([0.3081], dtype=image.dtype, device=device).view(1, -1, 1, 1)

    # prepare normalized input as a leaf with requires_grad
    image_denorm = image.clone().detach().to(device)
    image_norm = (image_denorm - mean) / std
    image_norm = image_norm.clone().detach().requires_grad_(True)

    # forward / loss / backward (local, does not modify external tensors)
    output = model(image_norm)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    # backward to get dL/dx_norm
    loss.backward()
    data_grad_norm = image_norm.grad.data  # gradient w.r.t. normalized input

    # convert gradient to pixel space: dL/dx_pixel = dL/dx_norm * (1/std)
    grad_pixel = data_grad_norm / std

    # FGSM in pixel space
    sign_data_grad = grad_pixel.sign()
    perturbed = image_denorm + epsilon * sign_data_grad
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()

    # cleanup grads to avoid side effects
    image_norm.grad = None

    return perturbed

# def fgsm_reattack(attacked_image: Tensor, epsilon: float, data_grad: Tensor) -> Tensor:
#     # すべて４次元テンソル
#     # print("fgsm_reattack")
#     # print("attacked_image_shape", attacked_image.shape)
#     # print("data_grad_shape", data_grad.shape)
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # print("sign_data_grad_shape", sign_data_grad.shape)
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = attacked_image - epsilon*sign_data_grad
#     # print("perturbed_image_shape", perturbed_image.shape)
#     # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # print("clamped_perturbed_image_shape", perturbed_image.shape)
#     # print("")
#     # Return the perturbed image
#     return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
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

def iterative_reattack(perturbed_denorm: Tensor, model: nn.Module, target: Tensor, device: str, step_epsilon: float, steps: int) -> Tensor:
    """
    perturbed_denorm: 非正規化されたテンソル (batch, channel, H, W), 値域 [0,1]
    step_epsilon: 1ステップあたりの摂動量（ピクセルスケール）
    steps: 繰り返す回数
    戻り値: 最終的に得られる非正規化テンソル（clamped, detach済み）
    """
    # mean/std tensors on device
    mean = torch.tensor([0.1307], device=device).view(1, -1, 1, 1)
    std = torch.tensor([0.3081], device=device).view(1, -1, 1, 1)

    adv = perturbed_denorm.clone().detach()  # start from given perturbed image (非正規化)
    for _ in range(steps):
        adv.requires_grad = True

        # normalize for the model
        adv_normalized = (adv - mean) / std

        # forward / loss / backward to get gradient w.r.t. adv (denorm space through normalization ops)
        output = model(adv_normalized)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        # ensure previous grads cleared
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()

        # data_grad is gradient wrt adv (same shape)
        data_grad = adv.grad.data

        # FGSM step in denorm (pixel) space
        adv = adv + step_epsilon * data_grad.sign()
        adv = torch.clamp(adv, 0.0, 1.0).detach()  # clamp and detach to prepare next iteration

    return adv

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    target_list = []
    count = 0

    # Loop over all examples in test set
    for data, target in test_loader:
        target_list.append(target)
        # print("test loop")
        # print("data_shape", data.shape)
        # # 4
        # print("target_shape", target.shape)
        # # 1
        # print("")


        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call FGSM Attack (内部で loss/backward を実行して摂動を作る)
        perturbed_data = fgsm_attack(data_denorm, epsilon, target)

        # reattack_steps = 1

        # perturbed_data = iterative_reattack(perturbed_data, model, target, device, step_epsilon=epsilon/reattack_steps  , steps=reattack_steps)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        # print(f"predicted: {final_pred.item()}, expected: {target.item()}")


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # for (init_pred, final_pred, adv_ex) in adv_examples:

    #     print(f"Adversarial example: {adv_ex}")


    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, .35, step=0.05))
# plt.title("Accuracy vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy")
# plt.show()

# # Plot several examples of adversarial samples at each epsilon
# cnt = 0
# plt.figure(figsize=(8,10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons),len(examples[0]),cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
#         orig,adv,ex = examples[i][j]
#         plt.title(f"{orig} -> {adv}")
#         plt.imshow(ex, cmap="gray")
# plt.tight_layout()
# plt.show()

