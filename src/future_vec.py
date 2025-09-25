import os
from lib.utils import folder_name, from_filename, get_device, load_saved_attacked_images, load_saved_denorm_image, extract_feature_vector_from_denorm

out_dir = "data/attacked_images"

def main():
    # 画像を読み出す
    denorm_images = load_saved_attacked_images(out_dir, 0.2, get_device())
    # build model
    from lib.lenet import Net
    device = get_device()
    model = Net().to(device)

    # you may want to load pretrained weights if available

    # 結果を見やすくするためにソート
    denorm_images = sorted(denorm_images, key=lambda x: x[0])  # index でソート

    for (idx, true_label, img_denorm) in denorm_images:
        if idx > 10:
            break
        vec = extract_feature_vector_from_denorm(img_denorm, model, device)
        print(f"idx={idx} label={true_label} feat_shape={vec.shape}")
        print(vec)
    

if __name__ == "__main__":
    main()
