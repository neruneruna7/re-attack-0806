import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from torch import Tensor



def fgsm(x, y_target, model, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y_target)
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + eps * grad_sign
    return torch.clamp(x_adv, 0.0, 1.0).detach()

def fgsm_attack(image: Tensor, epsilon: float, data_grad: Tensor):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def bim(x, y_target, model, eps, alpha, iters):
    x_adv = x.clone().detach()
    x_orig = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y_target)
        model.zero_grad(); loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.detach().sign()
        # ε-ボールへクリップ（L∞）
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv.detach(), 0.0, 1.0)
    return x_adv

@torch.no_grad()
def predict_label(model, x):
    return model(x).argmax(dim=1)

def counter_attack_fgsm_stepwise(x_ae, y_ae, model, eps_max=1.0, steps=50):
    # y_ae: AE生成時にモデルが出していた誤ラベル（または現在のAE予測）
    x_cur = x_ae.clone().detach()
    for t in range(1, steps + 1):
        eps = eps_max * t / steps
        # 一時的に勾配を求める区間だけ grad を有効化
        x_req = x_cur.clone().detach().requires_grad_(True)
        logits = model(x_req)
        loss = F.cross_entropy(logits, y_ae)
        model.zero_grad(); loss.backward()
        x_next = x_req + eps * x_req.grad.detach().sign()
        x_next = torch.clamp(x_next, 0.0, 1.0).detach()
        # 予測がAE時のラベルから変わったら停止
        if (predict_label(model, x_next) != y_ae).all():
            return x_next
    return x_next  # 変わらなければ最後のものを返す

def counter_attack_bim(x_ae, y_ae, model, eps=0.3, alpha=0.05, iters=10):
    x_next = bim(x_ae, y_ae, model, eps=eps, alpha=alpha, iters=iters)
    return x_next


# デバイス選定：MPS → CUDA → CPU の順
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ImageNet学習済みResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

# 前処理（ImageNet標準）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 画像読み込み
img = Image.open("./assets/org2_1.JPEG").convert("RGB")
# unsqueeze テンソルにサイズ1の次元を新しく挿入する．引数には，その追加場所を指定
# xは画像
x = preprocess(img).unsqueeze(0).to(device)

# 元ラベル
y_true = predict_label(model, x)

# まず通常のFGSMでAE生成（例: ε=0.03）
# x_ae = fgsm(x, y_true, model, eps=0.03)
x_ae = fgsm_attack(x, 0.03, model(x))
y_adv = predict_label(model, x_ae)

print("元ラベル:", y_true.item(), " AEラベル:", y_adv.item())

# 再攻撃で矯正
x_fix = counter_attack_fgsm_stepwise(x_ae, y_adv, model, eps_max=0.5, steps=50)
y_fix = predict_label(model, x_fix)

print("矯正後ラベル:", y_fix.item())
