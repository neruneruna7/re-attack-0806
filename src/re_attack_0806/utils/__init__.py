# ---- small utilities (package) ----
import os
from typing import Tuple, TypeAlias
import torch
import torchvision
from PIL import Image
import numpy as np
from torch import Tensor
from torchvision import transforms

from re_attack_0806.utils.normTensor import TensorWithState


def get_device() -> torch.device:
    print("5. 実行デバイスを自動判別します...")
    accelerator = torch.device("cpu")
    # TPUランタイムが有効かチェック
    """
    環境に応じてTPU, GPU, CPUの中から最適なデバイスを自動選択して返す。
    torch_xlaがインストールされており、利用可能な場合はTPUを優先する。
    """
    # 1. まずTPU (XLA) のインポートとデバイス取得を試みる
    try:
        import torch_xla.core.xla_model as xm
        # インポートに成功したら、TPUデバイスを取得して返す
        accelerator = xm.xla_device()
        print(f"Process on TPU: {accelerator}")
        return accelerator
    except ImportError:
        # torch_xlaがインストールされていない場合は無視して次へ進む
        pass
    except RuntimeError:
        # インストールされているがTPUに接続できない場合などの対策
        pass
    if torch.cuda.is_available():
        # GPU (CUDA) が利用可能
        accelerator = torch.device("cuda")
        print("ハードウェアアクセラレータ: GPU (CUDA) を使用します。")
    # Mac環境でのMPSはColabでは使用できないため、ここでは判定不要
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        accelerator = torch.device("mps")
        print("ハードウェアアクセラレータ: GPU (MPS) を使用します。")
    else:
        accelerator = torch.device("cpu")
        print("ハードウェアアクセラレータ: CPU を使用します。")
    # print("ハードウェアアクセラレータ: CPU を使用します。")
    # if torch.accelerator.is_available():
    #     accelerator = torch.accelerator.current_accelerator()
    # if accelerator is None:
    #     return torch.device("cpu")
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
    to_tensor = torchvision.transforms.ToTensor()  # C,H,W の 0..1 に正規化して返す
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
        imgs.append((i, true_label, img_denorm))
    return imgs


def extract_feature_vector_from_denorm(img_denorm: Tensor, model: torch.nn.Module, device: torch.device, mean=[0.1307], std=[0.3081]) -> np.ndarray:
    """
    Single responsibility: take a denormalized image tensor [1,1,H,W] in 0..1 on `device`,
    normalize it, run it through `model` (assumed to be lib.lenet.Net or similar),
    and return a 1-D numpy array feature vector extracted from the penultimate layer (fc1 output).

    This function does not modify model state and runs in eval() context without gradients.
    """
    # モデルを評価モードにする
    was_training = model.training
    model.eval()

    # 正規化用のテンソルを構築
    mean_t = torch.tensor(mean, dtype=img_denorm.dtype, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=img_denorm.dtype, device=device).view(1, -1, 1, 1)

    # 正規化
    img_norm = (img_denorm - mean_t) / std_t

    # 順伝播して、最終全結合層の直前の特徴を取得する
    # lib.lenet.Net では fc1 が目的の層（flatten の後）なので、fc1 まで手動で順伝播を実行する
    with torch.no_grad():
        # 静的解析上の問題を避けるため getattr 経由でモジュールを呼び出す
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

    # 訓練時の状態を復元する
    if was_training:
        model.train()

    # numpy の1次元ベクトルとして返す
    return feat.squeeze(0).cpu().numpy()




def save_tensor_as_image(tensor: Tensor, path: str):
    """
    tensor: [B,C,H,W] or [C,H,W] or [H,W], values in 0..1 (非正規化)
    path: output png path
    """
    t = tensor.clone().detach().cpu()
    # 必要に応じてバッチ／チャンネル次元を潰す
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3 and t.size(0) == 1:
        arr = t.squeeze(0).numpy()
    elif t.dim() == 3 and t.size(0) == 3:
        # CHW を HWC に変換
        arr = t.permute(1, 2, 0).numpy()
    elif t.dim() == 2:
        arr = t.numpy()
    else:
        arr = t.numpy()

    # クリップして uint8 に変換
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    # 親ディレクトリを作成
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = Image.fromarray(arr_u8)
    img.save(path)
    info = f"saved image {path}"
    return info

def l2_norm_perturbation(original: TensorWithState, perturbed: TensorWithState) -> Tensor:
    if original.state != perturbed.state:
        raise ValueError("original and perturbed tensors must have the same state")
    delta = perturbed.tensor - original.tensor

    # delta = delta.view(delta.size(0), -1)  # flatten
    # l2_norms = torch.linalg.norm(delta, dim=(1,2,3), p=2)  # バッチ内の各サンプルのL2ノルムを計算
    l2_norms = torch.norm(delta, dim=(1,2,3), p=2)  # バッチ内の各サンプルのL2ノルムを計算)
    average_perturbation = torch.mean(l2_norms)
    return average_perturbation
