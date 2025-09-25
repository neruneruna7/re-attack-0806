import os
import csv

import torch
from lib import attacks
from lib.utils import folder_name, from_filename, get_device, load_saved_attacked_images, load_saved_denorm_image, extract_feature_vector_from_denorm
from torch.types import Number


out_dir = "data/attacked_images"
epsilons = [0, .05, .1, .15, .2, .25, .3]
# epsilons = [0.05]
pretrained_model = "data/lenet_mnist_model.pth"

def main():
    # build model
    from lib.lenet import Net
    device = get_device()
    model = Net().to(device)
    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location=device, weights_only=True))


    # We'll compute accuracy per (eps, re_eps) pair and save CSV
    results = {}
    for eps in epsilons:
        # read saved images for this eps
        try:
            denorm_images = load_saved_attacked_images(out_dir, eps, device)
        except Exception:
            print(f"no images for eps={eps}, skipping")
            denorm_images = []

        denorm_images = sorted(denorm_images, key=lambda x: x[0])

        for re_eps in epsilons:
            correct = 0
            total = 0
            for (idx, true_label, img_denorm) in denorm_images:
                # if idx > 10:
                #     break

                # prepare tensors
                image = img_denorm.to(device)
                label = torch.tensor([true_label], device=device)

                # apply reattack (attacks.bim_reattack returns perturbed normalized or denorm depending impl)
                try:
                    x_adv_prime = attacks.bim_reattack(model, image, label, device, epsilon=re_eps, alpha=0.05, num_iter=10)
                except Exception as e:
                    print(f"reattack failed for idx={idx} eps={eps} re_eps={re_eps}: {e}")
                    continue

                with torch.no_grad():
                    pred_orig = model(image).argmax(dim=1).item()
                    pred_re_adv = model(x_adv_prime).argmax(dim=1).item()

                # only count samples where original prediction was correct
                # (this matches previous semantics: evaluate reattack effect on initially-correct samples)
                if pred_orig == true_label:
                    total += 1
                    if pred_re_adv == true_label:
                        correct += 1

            results[(eps, re_eps)] = (correct, total)
            # acc = (correct / total) if total > 0 else float('nan')
            # print(f"attack eps: {eps}  \treattack eps: {re_eps}  \tReattack Accuracy = {correct} / {total} = {acc}")
            print("re_eps", re_eps)

        print("eps", eps)

    # 表示
    for (eps, re_eps), (correct, total) in results.items():
        acc = (correct / total) if total > 0 else float('nan')
        print(f"attack eps: {eps}  \treattack eps: {re_eps}  \tReattack Accuracy = {correct} / {total} = {acc}")


    

if __name__ == "__main__":
    main()
