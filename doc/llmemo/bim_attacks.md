# BIM (Basic Iterative Method) と関連資料メモ

日付: 2025-11-12

このメモは mcp-gemini-cli の Google 検索を用いて収集した BIM（Basic Iterative Method）および関連する敵対的サンプル攻撃の資料をまとめたものです。

## 1. BIM 概要

Basic Iterative Method (BIM) は FGSM を反復適用することでより効果的な敵対的サンプルを生成する手法です。各反復で入力に小さな摂動を加え，元画像からの差分が L∞-ノルムで epsilon を超えないようにプロジェクション（クリップ）します。擬似コードは下記のとおり。

```
Function BasicIterativeMethodAttack(model, X_original, y_true, epsilon, alpha, num_iterations):
    X_adversarial = X_original
    For i From 1 To num_iterations:
        gradient = ComputeGradient(Loss(model(X_adversarial), y_true), X_adversarial)
        X_adversarial = X_adversarial + alpha * Sign(gradient)
        X_adversarial = Clip(X_adversarial, X_original - epsilon, X_original + epsilon)
        X_adversarial = Clip(X_adversarial, valid_pixel_range)
    Return X_adversarial
```

主要パラメータ:
- epsilon: L∞ 制約での最大摂動量
- alpha: 1 ステップあたりのステップサイズ
- num_iterations: 反復回数

## 2. 実装例（PyTorch）

検索で得た参考実装（要点）:
- 入力テンソルを `requires_grad=True` にし，ループ内で loss を計算して backward を呼び出す。
- 勾配の符号を取り，`adv = adv + alpha * sign(grad)` で更新する。
- 各反復で `adv` が元画像から epsilon を越えないよう `delta = clamp(adv - orig, -epsilon, epsilon)` を取り，`adv = clamp(orig + delta, min_val, max_val)` を実行する。

（検索結果のコードスニペットは `doc` を参照のこと）

## 3. Foolbox による実装と注意点

Foolbox は敵対的攻撃ライブラリで，BIM（Iterative FGSM）に相当する実装を提供している。Foolbox の実装や API を参照すると，モデルのラッパーや正規化の扱い・バッチ処理への対応方法など実装上の注意点がよく整理されている。

## 4. 参考 URL（検索で取得）
- Basic Iterative Method (説明・擬似コード) — 検索スニペット（mcp-gemini）
- PyTorch 実装例（bim_attack のサンプルスクリプト） — 検索スニペット（mcp-gemini）
- Foolbox ドキュメント（BIM/iterative FGSM の実装/使用例） — 検索スニペット（mcp-gemini）

※ 上記 URL と完全なコードは mcp-gemini の検索結果に含まれます。必要なら個別に URL を展開してこのファイルに追記します。

---

次のアクション提案:
- 実装をリポジトリ内コード（`src/lib/attacks/bim.py`）に合わせた具体的な実行例（サンプルスクリプト）を追加する。
- `README` または `doc/` に実行手順（uv run での実行例、MPS の注意点）を追記する。
