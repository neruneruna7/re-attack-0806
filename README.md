# re-attack-0806

敵対的サンプル攻撃（Adversarial Attack）と再攻撃（Re-attack）による防御手法の研究プロジェクト

## 概要

本プロジェクトは、機械学習モデルに対する敵対的サンプル攻撃（AE攻撃）とそれに対する再攻撃による防御手法を実装・実験するためのコードベースです。

## 主要機能

### 1. 汎用敵対的攻撃コード (`src/general_ae.py`)
- 複数の攻撃手法（FGSM, BIM, Foolbox BIM）をサポート
- 複数のデータセット（MNIST, CIFAR10, ImageNet）に対応
- コマンドライン引数による柔軟な設定

### 2. 汎用再攻撃コード (`src/general_reattack.py`) ⭐ NEW
- 攻撃されたサンプルに対する再攻撃による防御
- 初回攻撃と再攻撃のパラメータを個別に設定可能
- 攻撃成功率と再攻撃による復元成功率の自動集計

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/neruneruna7/re-attack-0806.git
cd re-attack-0806

# 依存関係のインストール（uvを使用）
uv sync
```

## 使用方法

### 汎用再攻撃コードの実行

基本的な使い方：

```bash
# デフォルト設定での実行（MNIST, BIM攻撃, BIM再攻撃）
uv run python src/general_reattack.py

# カスタムパラメータでの実行
uv run python src/general_reattack.py \
  --dataset mnist \
  --model mnist \
  --attack bim \
  --reattack bim \
  --epsilon 0.3 \
  --alpha 0.05 \
  --iters 10 \
  --reattack-epsilon 0.2 \
  --reattack-alpha 0.03 \
  --reattack-iters 15 \
  --num-samples 100

# プリセットを使用した実行
uv run python src/general_reattack.py --preset morimoto_mnist_bim

# ヘルプの表示
uv run python src/general_reattack.py --help
```

### パラメータ説明

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--dataset` | データセット（mnist, cifar10, imagenet） | mnist |
| `--model` | モデル（mnist, cifar10, ploof, inception_v3） | mnist |
| `--attack` | 初回攻撃手法（fgsm, bim, foolbox_bim） | bim |
| `--reattack` | 再攻撃手法（fgsm, bim） | bim |
| `--epsilon` | 初回攻撃の摂動量 | 0.3 |
| `--alpha` | 初回攻撃のステップサイズ | 0.05 |
| `--iters` | 初回攻撃の反復回数 | 10 |
| `--reattack-epsilon` | 再攻撃の摂動量 | 0.3 |
| `--reattack-alpha` | 再攻撃のステップサイズ | 0.05 |
| `--reattack-iters` | 再攻撃の反復回数 | 10 |
| `--num-samples` | 処理するサンプル数 | 100 |
| `--batch-size` | バッチサイズ | 1 |
| `--model-dir` | 学習済みモデルのディレクトリ | ./weight |

### 実行フロー

1. **クリーンデータの準備**: 指定されたデータセットからテストデータを読み込み
2. **初回攻撃**: 指定された攻撃手法でクリーンデータを攻撃
3. **攻撃後の推論**: 攻撃されたデータでモデルの予測を確認
4. **再攻撃**: 攻撃されたラベルをターゲットとして再攻撃を実行
5. **再攻撃後の推論**: 再攻撃後のデータでモデルの予測を確認
6. **結果集計**: 攻撃成功率と再攻撃による復元成功率を計算・表示

## プロジェクト構造

```
re-attack-0806/
├── src/
│   ├── general_ae.py          # 汎用敵対的攻撃コード
│   ├── general_reattack.py    # 汎用再攻撃コード（NEW）
│   ├── re_attack.py           # 再攻撃の基本実装
│   └── lib/
│       ├── attacks/
│       │   ├── fgsm.py        # FGSM攻撃の実装
│       │   └── bim.py         # BIM攻撃の実装
│       ├── models/            # モデル定義
│       └── utils.py           # ユーティリティ関数
├── weight/                    # 学習済みモデル
├── doc/                       # ドキュメント
└── README.md                  # このファイル
```

## 実装の特徴

### コーディング規約
- **タイプヒント**: すべての関数に完全なタイプヒントを追加
- **純粋関数設計**: 関数の出力は引数のみに依存し、外部状態を変更しない
- **エントリーポイント**: main関数を使用し、`if __name__ == "__main__"` で実行制御

### 実行環境
- **マシン**: m4 Mac mini
- **GPU**: MPS (Metal Performance Shaders) を推奨
- **Python**: >= 3.11
- **パッケージマネージャ**: uv

## 開発

### テストの実行

```bash
# 合成データを使用したテスト
uv run python test_general_reattack.py
```

### コードの品質チェック

```bash
# 構文チェック
uv run python -m py_compile src/general_reattack.py

# ヘルプの表示（コマンドライン引数のチェック）
uv run python src/general_reattack.py --help
```

## 参考文献

関連する論文や研究については `doc/bib.md` を参照してください。

## ライセンス

本プロジェクトのライセンスについては、リポジトリ管理者にお問い合わせください。

## 貢献

本プロジェクトへの貢献を歓迎します。PRを作成する際は以下の手順に従ってください：

1. mainブランチから新しいブランチを作成
2. 変更を実装
3. テストを実行して動作確認
4. PRを作成

## 問い合わせ

質問や提案がある場合は、GitHubのIssueを作成してください。
