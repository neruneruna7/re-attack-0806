# ImageNet (ILSVRC2012) セットアップ手順

日付: 2025-11-12

この文書は、`src/general_ae.py` 実行時に出るエラー:

```
RuntimeError: ImageNet dataset not found or devkit missing.
Please download the ILSVRC2012 data and devkit and place them under ../data/image_net,
or prepare a validation folder at ../data/image_net/val and retry.
See README or ImageNet official site for instructions to obtain ILSVRC2012 archives.
```

の原因と対処手順をまとめたものです。

## 原因の概要
- torchvision の `datasets.ImageNet` は ILSVRC2012 のメタデータ（devkit）および画像アーカイブを前提に作られており、これらが所定の場所に存在しないとエラーを出します。
- ImageNet の配布は利用規約の関係で自動ダウンロードができない場合が多く、手動でダウンロードして所定のディレクトリに置く必要があります。

## 推奨配置
リポジトリの相対パスで `../data/image_net` を想定しています。一般的な配置例:

```
re-attack-0806/
  data/
    image_net/
      ILSVRC2012_devkit_t12.tar.gz    # (任意: 圧縮のままでも可)
      ILSVRC2012_devkit_t12/          # (展開した devkit)
      ILSVRC2012_img_val.tar          # (val 画像アーカイブ)
      val/                            # (展開済み val 画像またはクラス別ディレクトリ)
        n01440764/
          img1.JPEG
          img2.JPEG
        n01443537/
          ...
```

`datasets.ImageNet(root, split='val')` は devkit と val アーカイブ（または展開済みの val ディレクトリ）を期待します。

## 手順（推奨）
1. ImageNet の公式サイトにアクセスして ILSVRC2012 のアーカイブをダウンロードします（アカウント登録が必要なことが多いです）。主に次の2つを入手します:
   - `ILSVRC2012_devkit_t12.tar.gz`
   - `ILSVRC2012_img_val.tar`

2. ダウンロードしたアーカイブを `../data/image_net` に置き、展開します:

```bash
# リポジトリルートから
cd data/image_net
tar -xvf ILSVRC2012_devkit_t12.tar.gz
tar -xvf ILSVRC2012_img_val.tar
```

3. `datasets.ImageNet` が期待する形式にするために、val 画像をクラスごとのサブディレクトリに振り分けます。devkit には val 用の割り当てが入っているので、それを使って振り分けるスクリプトの例を下に示します。

### val をクラス別に振り分ける Python スクリプト例

以下を `scripts/prepare_imagenet_val.py` として保存し、`data/image_net` の直下で実行してください（`ILSVRC2012_devkit_t12` と `ILSVRC2012_img_val` が展開済みであることが前提です）。

```python
#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import scipy.io

def prepare_val(root: Path):
    # devkit の mat ファイルから synset id の順序を読み取り (例: imagenet synsets)
    devkit_dir = root / 'ILSVRC2012_devkit_t12' / 'data'
    # ここでは val_annotation もしくは meta 情報からマッピングを取得する方法を使う
    # 具体的なファイル名は devkit のバージョンによって異なることがある
    # 参考: devkit の中に val.txt や mapping ファイルが含まれている場合がある

    # 例: val.txt (image_name, class_id) の形式を想定
    val_txt = devkit_dir / 'ILSVRC2012_validation_ground_truth.txt'
    val_images_dir = root / 'ILSVRC2012_img_val'
    target_dir = root / 'val'
    target_dir.mkdir(exist_ok=True)

    # シンプルなケース: ground_truth に class indices が並ぶ場合
    if val_txt.exists():
        with open(val_txt, 'r') as f:
            labels = [line.strip() for line in f]
        # 注意: labels の形式は devkit によって異なる。必要に応じて devkit の README を確認すること。
        imgs = sorted([p.name for p in val_images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        for img_name, lbl in zip(imgs, labels):
            class_dir = target_dir / lbl
            class_dir.mkdir(exist_ok=True)
            src = val_images_dir / img_name
            dst = class_dir / img_name
            if not dst.exists():
                shutil.copy(src, dst)
    else:
        raise RuntimeError('devkit mapping file not found. Check ILSVRC2012_devkit_t12 contents.')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='image_net root dir')
    args = p.parse_args()
    prepare_val(Path(args.root))
```

注: 上のスクリプトは devkit の中身（ファイル名／フォーマット）によって微調整が必要です。devkit の `README` を確認してください。

## 代替案
- 研究・検証用であれば Tiny-ImageNet や CIFAR のような公開データセットへ置き換えて動作確認を行う。Tiny-ImageNet は構造が似ており、サンプル数が小さいため素早く動かせます。
- `../data/image_net/val` に手動で少数のクラス・画像を置いてテストすることも可能（今回の `general_ae.py` は `ImageFolder` にフォールバックします）。

## 参考
- ImageNet 公式サイト: http://www.image-net.org/
- torchvision.datasets.ImageNet ドキュメント（バージョン依存）

---

必要なら、上の `scripts/prepare_imagenet_val.py` を実際にリポジトリへ追加して自動化するパッチを作成します。希望があれば教えてください。
