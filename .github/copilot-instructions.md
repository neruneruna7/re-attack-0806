## 重要な注意事項
- 日本語で回答してください．
- コード内のコメントは日本語で記述してください．
- Pythonコードには可能な限りタイプヒントを追加してください．
- 関数は純粋関数となるように設計してください．つまり，関数の出力は引数のみに依存し，関数の外部状態を変更しないようにしてください．
- エントリーポイントは必ずmain関数にしてください．
- 目的ごとに，エントリーポイントとなるスクリプトファイルを作成してください．
- 実行スクリプトは，uv run <script path> の形式で実行することができます．
- python3 ..コマンドの使用を禁止します．

## 実行環境
マシンはm4 mac miniを使用している．GPUにはMPSを指定すること．
Pythonのパッケージマネージャにはuvを使用している．

# フェーズ概要
## 必須フェーズ vs 任意フェーズ
### 必須フェーズ ほぼすべてのケースで実行
1. Phase 1: Investigation & Research - Context7/Kiriで調査．mcp-gemini-cliのgoogle search toolででWeb検索
2. Phase 4: Planning - TodoWriteで計画立案
3. Phase 5: Implementation - Serenaでコード実装
4. Phase 7: Code Review - コードの見直し．

## workflow steps
### Phase 1: Investigation & Research (調査フェーズ) 【必須】

**使用ツール**: Context7 MCP, Kiri MCP, gemini-cli (Google Search)

#### 1. 既存コードベースの調査（Serena MCPを使用）

SerenaMCPを使用して，リポジトリ内の関連コードを特定してください．以下の方法を活用してください．

**1-1. コンテキスト自動取得（推奨）**
- タスクに関連するコードスニペットを自動でランク付けして取得
- `goal`には具体的なキーワードを使用（抽象的な動詞は避ける）
- `compact: true`でトークン消費を95%削減

**1-2. 具体的なキーワード検索**
- 関数名、クラス名、エラーメッセージなど具体的な識別子で検索
- 広範な調査には`context_bundle`を使用

**1-3. 依存関係の調査**
- 影響範囲分析（inbound）や依存チェーン（outbound）を取得
- リファクタリング時の影響調査に最適

**1-4. コードの詳細取得**
- ファイルパスがわかっている場合に使用
- シンボル境界を認識して適切なセクションを抽出

#### 2. ライブラリドキュメントの確認
- Context7 MCPを使用して最新のライブラリドキュメントを取得
- Pytorch, torchvision, その他使用するライブラリの最新情報を確認
- `mcp__context7__resolve-library-id` → `mcp__context7__get-library-docs` の順で実行

#### 3. Web検索による補完調査（mcp-gemini-cliのGoogle Search Toolを使用）
- mcp-gemini-cliのGoogle Search Toolを使用して、関連する情報を検索. 特に関連する論文や，研究の例を調査．

**重要** 特に論文や研究に関する情報を得た場合，そのURLや概要を/doc/bib.md に必ず追記すること．

#### 4. 調査結果の整理
- 既存パターンやコーディング規約を把握
- 再利用可能なコンポーネントやユーティリティを特定
- Kiriで取得したコンテキストを基に実装方針を決定

**完了チェックリスト:**
- [ ] Serena MCPで関連コードを特定
- [ ] 必要なライブラリのドキュメントを確認
- [ ] 既存パターンと依存関係を把握

# プロジェクトサマリー（自動追加）

このリポジトリは，機械学習モデルに対する adversarial attack（攻撃）とそれに対する再攻撃（re-attack）を扱う実験コード群を含んでいます。主に Python / PyTorch による実験コードと，攻撃アルゴリズムの一部を実装した Rust サブプロジェクトを含みます。以下に主要な構成を示します。

- ルート
  - `main.py` : 環境（例: MPS GPU）を確認し，`mnist_train` や `cifar10_train` を呼んでモデル学習→保存する簡易エントリ例が記載されています。
  - `README.md` : 空または最小の README（要補完）。
  - `pyproject.toml` : Python 依存（例: torch, torchvision, einops, matplotlib, numpy, foolbox）および Python >= 3.11 を指定。

- データ・モデル
  - `data/` : データ関連（例: `attacked_images/` に多段攻撃や保存画像が格納されています）。
  - `models/` : モデル定義（例: `lenet` 等）。
  - `weight/` : 学習済みパラメータ（`.pth` ファイル）を格納。

- Python 実験コード（`src/` およびルート直下スクリプト）
  - `mnist_train.py`, `cifar10_train.py` : データセット別の学習スクリプト。
  - `src/` 内のスクリプト（例: `re_attack.py`, `pytorch_bim.py`, `pytorch_fgsm.py`, `bim_re_attack.py` など）：攻撃・再攻撃の実験・ユーティリティが実装されています。
  - `src/re_attack.py` では，保存済みの perturbed 画像を読み出してモデルの予測ラベルをターゲットに FGSM を繰り返す再攻撃処理の流れが実装されています。

- Rust サブプロジェクト
  - `re-attack/` : Rust 実装（`fgsm.rs`, `grad.rs`, `train.rs` 等）があり，別途コンパイル・実行可能なコードを含みます。

主要ポイント・実行メモ
- 実行マシン: m4 Mac mini を想定しています。GPU は Apple の Metal（MPS）を利用する設定になっています。
- Python 実験は PyTorch を前提としています（`pyproject.toml` の依存参照）。
- パッケージ管理・実行: この環境ではパッケージマネージャに `uv` を使用します。スクリプトの実行は通常 `uv run <script path>` の形式で行ってください（`uv.lock` がある場合はそれに従って環境が再現されます）。
- 簡易実行例（ルートで実行、uv を使う）:
  - モデル学習と保存: `uv run main.py`（`main.py` に学習→保存の例あり。内部で `mnist_train` / `cifar10_train` を切り替えられます）
  - 再攻撃実験: `uv run src/re_attack.py`（`src/re_attack.py` は単体実行で再攻撃のループを回します）
  - 注意: 必要に応じて `uv` 環境下で対話的に実験するか、スクリプト内で明示的に `get_device()` を呼んで MPS を指定してください。

注意事項・今後の提案
（このセクションはリポジトリを横断してファイルを読み取り、自動的に生成した要約です。必要に応じて加筆・修正してください。）
- 日本語で回答してください．
- Pythonコードには可能な限りタイプヒントを追加してください．
- 関数は純粋関数となるように設計してください．つまり，関数の出力は引数のみに依存し，関数の外部状態を変更しないようにしてください．
- エントリーポイントは必ずmain関数にしてください．
- 目的ごとに，エントリーポイントとなるスクリプトファイルを作成してください．


