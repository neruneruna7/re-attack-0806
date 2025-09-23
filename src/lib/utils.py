# ---- small utilities ----
import os
from typing import Tuple
import torch
import torchvision
from PIL import Image
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
