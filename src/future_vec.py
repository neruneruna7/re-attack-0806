import os
from lib.utils import folder_name, from_filename, get_device, load_saved_attacked_images, load_saved_denorm_image

out_dir = "data/attacked_images"

def main():
    # 画像を読み出す
    denorm_images = load_saved_attacked_images(out_dir, 0.2, get_device())

    

if __name__ == "__main__":
    main()
