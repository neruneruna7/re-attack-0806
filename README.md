# Adversarial Attack and Re-attack Experimentation Framework

このプロジェクトは、機械学習モデルに対する敵対的攻撃（Adversarial Attack）および、生成された敵対的サンプルに対する再攻撃（Re-attack）の実験を行うための，個人の研究リポジトリ．

## 1. 動作確認環境
- Python 3.11 以上
- [uv](https://github.com/astral-sh/uv) (高速なPythonパッケージインストーラおよびリゾルバ)
- m4 mac mini

## 2. 環境構築 (Setup)

1.  **リポジトリをクローンします。**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **`uv` を使って依存関係をインストールします。**
    プロジェクトルートには`pyproject.toml`と`uv.lock`が含まれており、`uv sync`を実行することで、再現可能な仮想環境が構築されます。
    ```bash
    uv sync
    ```

## 3. 実行方法 (Usage)

本プロジェクトのスクリプトは `uv run` コマンドを通じて実行します。

### 3.1. 敵対的サンプルの生成 (`general_ae.py`)

`src/general_ae.py` は、指定されたモデルとデータセットに対して敵対的攻撃を行い、敵対的サンプルを生成します。

**実行例:**
```bash
uv run src/general_ae.py --dataset mnist --model morimoto-mnist --attack bim --epsilon 0.3 --alpha 0.05 --n 10
```

### 3.2. 再攻撃の実験 (`general_reattack.py`)

`src/general_reattack.py` は、まず初期攻撃で敵対的サンプルを生成し、続けてそのサンプルに対して再攻撃を行います。

**実行例:**
```bash
uv run src/general_reattack.py \
  --dataset cifar10 \
  --model morimoto-cifar10 \
  --attack-kind fgsm \
  --attack-eps 0.3 \
  --reattack-kind bim \
  --reattack-eps 0.1 \
  --reattack-alpha 0.02 \
  --reattack-n 20
```

### 3.3. 主要なコマンドライン引数

- `--dataset`: 使用するデータセット (`mnist`, `cifar10`など)。 (必須)
- `--model`: 使用するモデル (`morimoto-mnist`, `morimoto-cifar10`など)。 (必須)
- `--attack` / `--attack-kind`: 攻撃手法 (`fgsm`, `bim`, `foolbox-bim`など)。 (必須)
- `--epsilon`, `--alpha`, `--n`: 各攻撃手法のパラメータ。手法によって必須のものが異なります。
- `--num-samples <N>`: 処理するサンプル数を `N` 個に制限します。デバッグや高速なテストに便利です。
- `--batch-size <N>`: バッチサイズを指定します (デフォルト: 64)。実行中にメモリエラーが出た場合，バッチサイズを小さくしてみてください．
- `--shuffle-dataloader`: データローダーをシャッフルします (デフォルト: 無効)。
- `--no-save-images`: 生成された画像の保存を無効にします (デフォルト: 保存する)。


