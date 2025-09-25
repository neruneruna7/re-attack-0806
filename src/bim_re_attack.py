import os

import torch
from lib import attacks
from lib.utils import folder_name, from_filename, get_device, load_saved_attacked_images, load_saved_denorm_image, extract_feature_vector_from_denorm
from torch.types import Number


out_dir = "data/attacked_images"
epsilons = [0, .05, .1, .15, .2, .25, .3]
epsilons = [0.05]
pretrained_model = "data/lenet_mnist_model.pth"

def main():
    # build model
    from lib.lenet import Net
    device = get_device()
    model = Net().to(device)
    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location=device, weights_only=True))


    result = []
    for eps in epsilons:
        # 画像を読み出す
        denorm_images = load_saved_attacked_images(out_dir, eps, get_device())
        # you may want to load pretrained weights if available

        # 結果を見やすくするためにソート
        denorm_images = sorted(denorm_images, key=lambda x: x[0])  # index でソート


        result_eps = []
        for (idx, true_label, img_denorm) in denorm_images:
            # if idx > 10:
            #     break
            image, label = img_denorm, torch.tensor([true_label], device=device)
            label_taple: list[tuple[float, Number, Number, Number]]= []
            for re_eps in epsilons:

                x_adv_prime = attacks.bim_reattack(model, image, label, device, epsilon=re_eps, alpha=0.05, num_iter=10)
                # 元画像と再攻撃後の予測を比較
                with torch.no_grad():
                    pred_orig = model(image).argmax(dim=1)
                    pred_re_adv = model(x_adv_prime).argmax(dim=1)
                    # print(f"元のラベル: {label.item()}")
                    # print(f"元画像の予測: {pred_orig.item()}")
                    # print(f"再攻撃後の予測: {pred_adv.item()}")
                    label_taple.append((re_eps, label.item(), pred_orig.item(), pred_re_adv.item()))
                    # accuracies.append(acc)
                    # examples.append(ex)

            result_eps.append((idx, label_taple))

        result.append((eps, result_eps))

    print("done.", result)
    # 正解率の計算
    # 元のラベルに対して，再攻撃後の予測が一致している割合
    # ただし，元画像の予測の時点で正しいものに限る
    # 元画像の予測の時点で間違っているものは，別途カウントする
    # correct_count = 0
    # total_count = 0
    # wrong_pred_count = 0
    # for (true_label, pred_orig, pred_re_adv) in label_taple:
    #     if true_label == pred_orig:
    #         total_count += 1
    #         if true_label == pred_re_adv:
    #             correct_count += 1
    #     else:
    #         wrong_pred_count += 1

    # final_acc = correct_count / total_count if total_count > 0 else 0
    # print(f"再攻撃後の正解率: {final_acc} ({correct_count} / {total_count})")
    # print(f"元画像の予測が間違っていた数: {wrong_pred_count}")

    # 正解率の計算
    correct_count = 0
    correct_re_adv_count = 0
    total_count = 0
    for (re_eps, true_label, pred_orig, pred_re_adv) in label_taple:
        total_count += 1
        if true_label == pred_re_adv:
            correct_re_adv_count += 1
        if true_label == pred_orig:
            correct_count += 1

    final_acc = correct_count / total_count if total_count > 0 else 0
    print(f"攻撃後の正解率: {final_acc} ({correct_count} / {total_count})")
    final_re_adv_acc = correct_re_adv_count / total_count if total_count > 0 else 0
    print(f"再攻撃後の正解率: {final_re_adv_acc} ({correct_re_adv_count} / {total_count})")


    

if __name__ == "__main__":
    main()
