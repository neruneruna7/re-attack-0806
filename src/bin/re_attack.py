from torchvision import transforms as T
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

from pytorch_fgsm import Net, create_filename, get_device

torch.manual_seed(42)
out_dir = "data/attacked_images"


# --- 追加: 保存画像を読み出して再攻撃するユーティリティ ---
def load_saved_denorm_image(epsilon: float, index: int, device) -> Tensor:
    """
    保存された PNG を読み出して非正規化テンソル [1,1,H,W] (0..1) を返す。
    ファイル名は create_filename を使用。
    """
    path = create_filename(epsilon, index)
    if not os.path.exists(path):
        raise FileNotFoundError(f"saved image not found: {path}")
    img = Image.open(path).convert("L")  # グレースケール
    to_tensor = T.ToTensor()  # returns C,H,W in 0..1
    t = to_tensor(img).unsqueeze(0).to(device)  # [1,1,H,W]
    return t

def reattack_saved_image(epsilon_saved: float, index: int, re_eps: float, model: torch.nn.Module, device, steps: int = 1):
    """
    保存済みの perturbed (非正規化) を読み出し、モデル上で予測されたラベルをターゲットにして
    fgsm_attack を steps 回繰り返し適用する（各ステップで fgsm_attack を呼ぶ実装）。
    - epsilon_saved: 保存画像の元の epsilon（create_filename に合わせる）
    - index: 保存画像の index（create_filename に合わせる）
    - re_eps: 再攻撃で使う 1-step の epsilon（ピクセルスケール）
    - steps: 繰り返し回数（1 以上）
    - save_out: True なら out_dir 以下に reattacked を保存
    戻り値: (before_pred, after_pred, reattacked_denorm_tensor)
    """
    # load saved image (非正規化)
    img_denorm = load_saved_denorm_image(epsilon_saved, index, device)  # [1,1,H,W] on device

    # normalize for model input
    mean_t = torch.tensor([0.1307], dtype=img_denorm.dtype, device=img_denorm.device).view(1, -1, 1, 1)
    std_t = torch.tensor([0.3081], dtype=img_denorm.dtype, device=img_denorm.device).view(1, -1, 1, 1)
    img_norm = (img_denorm - mean_t) / std_t

    # get current model prediction (before reattack)
    with torch.no_grad():
        out = model(img_norm)
        before_pred = out.argmax(1).view(-1).to(device).long()  # shape [1]

    # use predicted label as target for reattack
    target = before_pred.clone()

    adv = img_denorm.clone().detach()
    for step in range(steps):
        # fgsm_attack signature: (image_denorm, epsilon, target, model, device)
        adv = fgsm_attack(adv, re_eps, target, model, device).detach()

    # final prediction after reattack
    adv_norm = (adv - mean_t) / std_t
    with torch.no_grad():
        out2 = model(adv_norm)
        after_pred = out2.argmax(1).view(-1).to(device).long()


    return before_pred.item(), after_pred.item(), adv

def main():
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    pretrained_model = "data/lenet_mnist_model.pth"
    # Set random seed for reproducibility

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


    # 使い方例 (REPL か main の末尾などで):
    before, after, adv_tensor = reattack_saved_image(0.05, 0, re_eps=0.05, model=model, device=device, steps=1)
    print("before -> after:", before, "->", after)

# mainとして呼ばれたときのおまじないのあれ
if __name__ == "__main__":
    main()