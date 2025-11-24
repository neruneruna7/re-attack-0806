
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

### 3.4. タスクランナー (cargo make)

本プロジェクトでは、`cargo make` をタスクランナーとして使用し、Pythonスクリプトの実行を簡素化しています。`Makefile.toml` に、よく使うコマンドや複雑な引数を持つコマンドが定義されています。

#### 3.4.1. cargo-makeのインストール

`cargo make` はRustのクレートであるため、[Rustの公式ドキュメント](https://www.rust-lang.org/tools/install)に従ってRustおよびCargoがインストールされている必要があります。
インストール後、以下のコマンドで `cargo-make` を導入します。

```bash
cargo install cargo-make
```

#### 3.4.2. 利用可能なタスクの確認

定義されているすべてのタスクと、その説明を一覧表示するには、以下のコマンドを実行します。

```bash
cargo make --list-all-steps
```

#### 3.4.3. タスクの実行例

`Makefile.toml` に定義されているタスクは、`cargo make <task_name>` の形式で実行できます。

- **CIFAR-10モデルの学習:**
    ```bash
    cargo make train-morimoto-cifar10
    ```
- **MNISTモデルへのBIM攻撃:**
    ```bash
    cargo make attack-morimoto-mnist-bim
    ```
- **ImageNetモデルへのFoolbox BIM攻撃 (バッチサイズ16、1024サンプル):**
    ```bash
    cargo make attack-foolbox-bim-advo
    ```

## 4. Rust版 AE攻撃CLI

本プロジェクトにはRust + Burnフレームワークで実装されたAE攻撃・防御実験用のCLIツールも含まれています。

### 4.1. 環境構築

Rustのインストールが必要です。[公式サイト](https://www.rust-lang.org/tools/install)の手順に従ってインストールしてください。

```bash
# Rustのインストール確認
rustc --version
cargo --version
```

### 4.2. ビルド方法

```bash
cd re-attack
cargo build --release
```

### 4.3. 利用可能なバイナリ

#### 4.3.1. AE攻撃 (`ae_attack`)

指定したモデルとデータセットに対して敵対的攻撃を実行します。

```bash
# BIM攻撃の実行例（GPU使用）
cargo run --release --bin ae_attack -- \
  --model resnet18 \
  --attack bim \
  --epsilon 0.3 \
  --alpha 0.01 \
  --num-iter 10 \
  --num-samples 100

# FGSM攻撃の実行例（CPU使用）
cargo run --release --bin ae_attack -- \
  --backend cpu \
  --model simple-mlp \
  --attack fgsm \
  --epsilon 0.3 \
  --num-samples 50
```

**主なオプション:**
- `--model`: モデルの種類 (`resnet18` | `simple-mlp`)
- `--attack`: 攻撃手法 (`bim` | `fgsm`)
- `--backend`: バックエンド (`cpu` | `wgpu`)
- `--epsilon`: 摂動の最大値
- `--alpha`: BIMの更新ステップサイズ
- `--num-iter`: BIMのイテレーション回数
- `--num-samples`: 処理するサンプル数
- `-v, --verbose`: 詳細ログ出力

#### 4.3.2. AE再攻撃防御 (`ae_reattack_defense`)

初期攻撃で生成した敵対的サンプルに対して再攻撃を行い、防御効果を評価します。

```bash
cargo run --release --bin ae_reattack_defense -- \
  --attack bim \
  --epsilon 0.3 \
  --reattack-method bim \
  --reattack-epsilon 0.1 \
  --reattack-alpha 0.02 \
  --reattack-num-iter 20 \
  --num-samples 100
```

**追加オプション:**
- `--reattack-method`: 再攻撃の手法
- `--reattack-epsilon`: 再攻撃のepsilon値
- `--reattack-alpha`: 再攻撃のalpha値
- `--reattack-num-iter`: 再攻撃のイテレーション回数

#### 4.3.3. 前処理+AE再攻撃防御 (`preprocess_ae_reattack_defense`)

画像に前処理（ノイズ除去、平滑化など）を適用してから再攻撃を行います。

```bash
cargo run --release --bin preprocess_ae_reattack_defense -- \
  --attack bim \
  --epsilon 0.3 \
  --preprocess-type gaussian-blur \
  --preprocess-strength 0.1 \
  --reattack-method bim \
  --reattack-epsilon 0.1 \
  --num-samples 100
```

**前処理オプション:**
- `--preprocess-type`: 前処理の種類 (`gaussian-blur` | `median-filter` | `denoise`)
- `--preprocess-strength`: 前処理の強度（0.0-1.0）

### 4.4. ヘルプの表示

各バイナリの詳細なヘルプは `--help` オプションで確認できます：

```bash
cargo run --bin ae_attack -- --help
cargo run --bin ae_reattack_defense -- --help
cargo run --bin preprocess_ae_reattack_defense -- --help
```

### 4.5. 設計の特徴

- **モジュール化**: 攻撃手法はTraitで抽象化され、新しい手法の追加が容易
- **バックエンド選択**: CPU (NdArray) / GPU (Wgpu) を実行時に選択可能
- **拡張性**: 新しいモデル、データセット、攻撃手法を簡単に追加できる設計



