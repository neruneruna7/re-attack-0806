# プログラム改善: 設計ドキュメント

このドキュメントでは、`general_ae.py`のリファクタリングで実施した改善内容を詳細に説明します。

## 概要

本リファクタリングでは、以下の5つの主要な改善を実施しました：

1. **ストラテジーパターンの導入**（攻撃ロジックの分離）
2. **CLIの改善**（Subparsersの導入）
3. **コードの重複排除**（DRY原則）
4. **結果処理の責務分離**
5. **パス管理の一元化**

すべての改善において**単一責任原則**を遵守し、既存の動作を変更せずに構造のみをリファクタリングしました。

## 1. ストラテジーパターンの導入

### 課題

元のコードでは、`Runner`クラスの`run`メソッド内に複数の攻撃手法（BIM、FGSM、Foolbox系）のロジックがif/elif文で混在していました。これにより以下の問題が発生していました：

- コードが長く複雑
- 新しい攻撃手法の追加時に既存コードの修正が必要
- 保守性と拡張性の低下

### 解決策

ストラテジーパターンを導入し、各攻撃ロジックを独立したクラスにカプセル化しました。

#### 構造

```
src/re_attack_0806/strategies/
├── __init__.py                    # パッケージ初期化
├── base.py                        # AttackStrategy抽象基底クラス
├── bim_strategy.py                # BIM攻撃ストラテジー
├── fgsm_strategy.py               # FGSM攻撃ストラテジー
├── foolbox_bim_strategy.py        # Foolbox BIM攻撃ストラテジー
├── foolbox_fgsm_strategy.py       # Foolbox FGSM攻撃ストラテジー
├── foolbox_utils.py               # Foolbox共通ユーティリティ
└── factory.py                     # AttackStrategyFactory
```

#### 主要クラス

**AttackStrategy（抽象基底クラス）**
```python
class AttackStrategy(ABC):
    @abstractmethod
    def execute(
        self,
        data_ts_norm: TensorWithState,
        target_ts: Tensor,
        model: nn.Module,
        device: torch.device,
        dataset_norm: DatasetNorm,
        attack_params: AttackParams
    ) -> TensorWithState:
        pass
```

**AttackStrategyFactory**

ディクショナリマッピングを使用して、攻撃種別から適切なストラテジーを生成します：

```python
class AttackStrategyFactory:
    _STRATEGY_MAP: Dict[AttackKind, Type[AttackStrategy]] = {
        AttackKind.BIM: BIMAttackStrategy,
        AttackKind.FGSM: FGSMAttackStrategy,
        # ...
    }
    
    @classmethod
    def create(cls, attack_kind: AttackKind) -> AttackStrategy:
        strategy_class = cls._STRATEGY_MAP.get(attack_kind)
        if strategy_class is None:
            raise ValueError(f"サポートされていない攻撃種別です: {attack_kind}")
        return strategy_class()
```

#### メリット

- **拡張性**: 新しい攻撃手法の追加が容易（新しいストラテジークラスを追加し、ファクトリーに登録するだけ）
- **保守性**: 各攻撃手法が独立しているため、修正の影響範囲が限定される
- **テスト容易性**: 各ストラテジーを個別にテスト可能
- **可読性**: Runnerクラスが大幅に簡素化

## 2. CLIの改善（Subparsersの導入）

### 課題

元のコードでは、すべての攻撃手法で共通の引数を使用していたため、以下の問題がありました：

- 攻撃種別ごとに必要な引数が異なるにもかかわらず、すべてオプション引数として定義
- 手動での引数検証が必要で、コードが複雑化
- ヘルプメッセージが分かりにくい

### 解決策

argparseのSubparsers機能を使用し、各攻撃手法に専用のサブコマンドを定義しました。

#### 使用例

**旧CLI**
```bash
python general_ae.py --dataset mnist --model morimoto-mnist --attack fgsm --epsilon 0.3
python general_ae.py --dataset mnist --model morimoto-mnist --attack bim --epsilon 0.3 --alpha 0.05 --n 10
```

**新CLI**
```bash
python general_ae.py fgsm --dataset mnist --model morimoto-mnist --epsilon 0.3
python general_ae.py bim --dataset mnist --model morimoto-mnist --epsilon 0.3 --alpha 0.05 --n 10
```

#### メリット

- **直感的**: 攻撃手法がサブコマンドとして明示的
- **型安全**: 各攻撃に必要な引数のみを要求、不要な引数は表示されない
- **自動検証**: argparseが引数の検証を自動的に実行
- **ヘルプメッセージ**: 各攻撃手法ごとに詳細なヘルプを表示
- **日本語対応**: すべてのヘルプメッセージを日本語で記述

## 3. コードの重複排除（DRY原則）

### 課題

元のコードでは、以下の重複がありました：

1. Foolbox攻撃でのfmodelの初期化処理
2. 攻撃パラメータの生成ロジック

### 解決策

#### FoolboxModelCreatorユーティリティ

Foolboxモデルの初期化処理を共通化：

```python
class FoolboxModelCreator:
    @staticmethod
    def create_foolbox_model(
        model: nn.Module,
        dataset_norm: DatasetNorm,
        device: torch.device
    ) -> foolbox.PyTorchModel:
        # 共通の初期化ロジック
        ...
```

#### AttackParamsFactory

攻撃パラメータの生成ロジックを一元化：

```python
class AttackParamsFactory:
    @staticmethod
    def create_from_args(
        attack_kind: AttackKind,
        epsilon: float,
        batch_size: int,
        alpha: Optional[float] = None,
        iters: Optional[int] = None
    ) -> AttackParams:
        # 攻撃種別に応じたパラメータ生成
        ...
```

#### メリット

- **保守性**: 変更が必要な場合、1箇所のみを修正
- **一貫性**: すべての場所で同じロジックを使用
- **テスト容易性**: 共通処理を一度テストすればよい

## 4. 結果処理の責務分離

### 課題

元のコードでは、mainメソッド内のループで以下の処理が混在していました：

- 結果の集計
- CSVへの書き込み
- 画像の保存
- 統計情報の計算

### 解決策

`ResultProcessor`クラスを作成し、結果処理の責務を分離しました。

#### ResultProcessorクラス

```python
class ResultProcessor:
    def __init__(self, csv_filepath: str, output_folder: str, save_images: bool):
        # 初期化
        
    def __enter__(self):
        # CSVファイルとThreadPoolExecutorを初期化
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # リソースをクリーンアップ
        
    def process_result(self, result: AttackResult, perturbed_image: TensorWithState):
        # 1つの結果を処理（CSV書き込み、画像保存、統計更新）
        
    def get_summary(self) -> AttackSummary:
        # 攻撃結果のサマリーを取得
        
    @staticmethod
    def print_summary(...):
        # サマリーを出力
```

#### メリット

- **単一責任**: 結果処理に関する責務のみを担当
- **リソース管理**: コンテキストマネージャーで適切なリソース管理
- **再利用性**: 他のスクリプトでも使用可能
- **テスト容易性**: 結果処理を独立してテスト可能

## 5. パス管理の一元化

### 課題

元のコードでは、出力ファイルやディレクトリのパスを生成するロジックがmain関数内に直接記述されていました。

### 解決策

`PathManager`クラスを作成し、パス生成ロジックを一元化しました。

#### PathManagerクラス

```python
@dataclass
class PathManager:
    config: Any
    
    def get_output_folder(self) -> str:
        # 出力フォルダのパスを取得
        
    def get_csv_filepath(self) -> str:
        # CSV結果ファイルのパスを取得
        
    def ensure_output_folder_exists(self) -> str:
        # フォルダが存在することを保証
```

#### メリット

- **単一責任**: パス管理の責務のみを担当
- **保守性**: パス構造の変更が1箇所で済む
- **テスト容易性**: パス生成ロジックを独立してテスト可能

## 全体的なメリット

このリファクタリングにより、以下のメリットが得られました：

### 1. コードの品質向上

- **可読性**: 各クラスの責務が明確で理解しやすい
- **保守性**: 変更の影響範囲が限定され、修正が容易
- **拡張性**: 新機能の追加が容易

### 2. 開発効率の向上

- **新しい攻撃手法の追加**: 新しいストラテジークラスを作成し、ファクトリーに登録するだけ
- **テスト**: 各コンポーネントを独立してテスト可能
- **デバッグ**: 問題の原因を特定しやすい

### 3. ユーザーエクスペリエンスの向上

- **直感的なCLI**: サブコマンドによる明確な操作方法
- **日本語のヘルプ**: わかりやすい説明と使用例
- **エラーメッセージ**: 的確なエラー通知

## 使用例

### FGSM攻撃

```bash
uv run src/general_ae.py fgsm \
  --dataset mnist \
  --model morimoto-mnist \
  --epsilon 0.3 \
  --num-samples 100 \
  --save-images
```

### BIM攻撃

```bash
uv run src/general_ae.py bim \
  --dataset cifar10 \
  --model morimoto-cifar10 \
  --epsilon 0.03 \
  --alpha 0.01 \
  --n 10 \
  --batch-size 32
```

### Foolbox FGSM攻撃

```bash
uv run src/general_ae.py foolbox-fgsm \
  --dataset mnist \
  --model morimoto-mnist \
  --epsilon 0.3 \
  --save-images
```

## 新しい攻撃手法の追加方法

新しい攻撃手法を追加する手順：

1. **ストラテジークラスの作成**
   
   `src/re_attack_0806/strategies/new_attack_strategy.py`を作成：
   
   ```python
   class NewAttackStrategy(AttackStrategy):
       def execute(self, ...):
           # 攻撃ロジックの実装
           ...
   ```

2. **ファクトリーへの登録**
   
   `src/re_attack_0806/strategies/factory.py`のマッピングに追加：
   
   ```python
   _STRATEGY_MAP: Dict[AttackKind, Type[AttackStrategy]] = {
       # ...
       AttackKind.NEW_ATTACK: NewAttackStrategy,
   }
   ```

3. **CLIサブパーサーの追加**
   
   `src/general_ae.py`の`parse_args`関数にサブパーサーを追加：
   
   ```python
   new_attack_parser = subparsers.add_parser(
       'new-attack',
       help='新しい攻撃手法',
       description='...'
   )
   add_common_arguments(new_attack_parser)
   # 攻撃固有の引数を追加
   ```

4. **攻撃パラメータの定義**
   
   必要に応じて`src/re_attack_0806/utils/config.py`に新しいパラメータクラスを追加

## まとめ

本リファクタリングでは、単一責任原則、DRY原則、ストラテジーパターンなどのベストプラクティスを適用することで、コードの品質、保守性、拡張性を大幅に向上させました。これにより、新しい攻撃手法の追加が容易になり、長期的な保守コストが削減されます。
