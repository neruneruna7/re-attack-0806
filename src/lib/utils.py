# ---- small utilities ----
import os
from typing import Tuple
import torch
import torchvision
from PIL import Image
import numpy as np
from torch import Tensor
from torchvision import transforms



def get_device() -> torch.device:
    accelerator = torch.device("cpu")
    if torch.accelerator.is_available() :
        accelerator = torch.accelerator.current_accelerator()
    if accelerator is None :
        return torch.device("cpu")
    return accelerator

def create_filename(out_dir: str, epsilon: float, index: int, true_label: int) -> str:
    return f"{out_dir}/eps_{epsilon:.3f}/reattacked_{index}_label_{true_label}.png"

def folder_name(out_dir: str, epsilon: float) -> str:
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

# --- 追加: 保存画像を読み出す---
def load_saved_denorm_image(dir: str, epsilon: float, index: int, device, true_label: int) -> Tensor:
    """
    保存された PNG を読み出して非正規化テンソル [1,1,H,W] (0..1) を返す。
    ファイル名は create_filename を使用。
    """
    path = create_filename(dir, epsilon, index, true_label)
    if not os.path.exists(path):
        raise FileNotFoundError(f"saved image not found: {path}")
    img = Image.open(path).convert("L")  # グレースケール
    to_tensor = torchvision.transforms.ToTensor()  # returns C,H,W in 0..1
    t = to_tensor(img).unsqueeze(0).to(device)  # [1,1,H,W]
    return t

def load_saved_attacked_images(dir: str, epsilon: float, device: torch.device) -> list[Tuple[int, int, Tensor]]:
    """
    指定フォルダから保存された attacked 画像をすべて読み出し、(index, true_label, tensor) のリストを返す。
    """
    folder = folder_name(dir, epsilon)
    files = os.listdir(folder)
    imgs = []
    for f in files:
        (i, true_label) = from_filename(f)
        img_denorm = load_saved_denorm_image(dir, epsilon, i, device, true_label)
        imgs.append( (i, true_label, img_denorm) )
    return imgs


def extract_feature_vector_from_denorm(img_denorm: Tensor, model: torch.nn.Module, device: torch.device, mean=[0.1307], std=[0.3081]) -> np.ndarray:
    """
    Single responsibility: take a denormalized image tensor [1,1,H,W] in 0..1 on `device`,
    normalize it, run it through `model` (assumed to be lib.lenet.Net or similar),
    and return a 1-D numpy array feature vector extracted from the penultimate layer (fc1 output).

    This function does not modify model state and runs in eval() context without gradients.
    """
    # ensure model in eval
    was_training = model.training
    model.eval()

    # build normalization tensors
    mean_t = torch.tensor(mean, dtype=img_denorm.dtype, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=img_denorm.dtype, device=device).view(1, -1, 1, 1)

    # normalize
    img_norm = (img_denorm - mean_t) / std_t

    # Forward and capture feature before final fully-connected
    # For lib.lenet.Net, fc1 is the layer we want (after flatten)
    # We'll run forward manually up to fc1
    with torch.no_grad():
        # call modules via getattr to avoid static-analysis issues
        conv1 = getattr(model, "conv1")
        conv2 = getattr(model, "conv2")
        dropout1 = getattr(model, "dropout1")
        fc1 = getattr(model, "fc1")

        x = conv1(img_norm)
        x = torch.nn.functional.relu(x)
        x = conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = dropout1(x)
        x = torch.flatten(x, 1)
        feat = fc1(x)
        feat = torch.nn.functional.relu(feat)

    # restore training state
    if was_training:
        model.train()

    # return as numpy 1-D vector
    return feat.squeeze(0).cpu().numpy()
