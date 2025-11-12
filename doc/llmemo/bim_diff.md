# BIM 実装差分レポート

日付: 2025-11-12

対象ファイル:
- `src/lib/attacks/bim.py`  （以降: package-bim）
- `src/lib/attacks____.py` （以降: legacy-bim）

目的: 両実装を比較し、挙動の違い・誤りを明確にし、legacy-bim のどこが間違っていたかを報告する。

## 1. インターフェース（関数シグネチャ）

- package-bim
	- def bim_attack(image: Tensor, epsilon: float, alpha: float, num_iter: int, target: Tensor, model: nn.Module, device: torch.device, *, mean: Optional[Tensor]=None, std: Optional[Tensor]=None) -> Tensor
	- 入力は「非正規化画像（0..1）」を想定し、内部で mean/std を使って正規化 → 勾配計算。

- legacy-bim
	- def bim_attack(image: Tensor, epsilon: float, alpha: float, n: int, target: Tensor, model: nn.Module, device: torch.device) -> Tensor
	- こちらも「非正規化画像（0..1）」を前提としているように見えるが、正規化の扱いに実装上の差がある（下記）。

結論: API はほぼ同じ目的に一致しているが、内部実装の差異が挙動に大きく影響する。

## 2. 勾配を評価する入力（クリティカルな差異）

- package-bim（正しい実装）:
	- 各反復で `perturbed`（現在の敵対的サンプル）を正規化して `perturbed_norm = (perturbed - mean) / std` を作り、`requires_grad_(True)` を付けてモデルに渡す。
	- したがって「現在の perturbed における損失の勾配」を計算して更新に使っている（正しい）。

- legacy-bim（間違い）:
	- ループ内で `image_norm = (image_denorm - mean) / std` を作っている（image_denorm はループ外で定義された元の入力）。つまり、常に「元画像（未変化）」に対する勾配を計算している。
	- その結果、更新方向は毎回元画像に対する勾配の符号であり、反復的に perturbed の局所勾配に基づく最適化を行っていない。これは BIM の本来のアルゴリズムから外れており、反復の意味を失わせる重大なバグである。

インパクト: legacy-bim は「反復的に小さく摂動を重ねる」ことで得られる効果を打ち消し、FGSM を単に複数回適用したような不正確な振る舞いになり得る（少なくとも期待される効果が得られない）。

## 3. 勾配取得方法と副作用

- package-bim
	- 標準的な実装では `loss.backward()` を用いる箇所があるが、我々の改善案では `torch.autograd.grad` を使うことを推奨している（入力勾配のみ取得し、モデルパラメータの grad に副作用を与えないため）。

- legacy-bim
	- `loss.backward()` を使っており、`model.zero_grad()` は呼んでいるものの、`backward()` によりモデルパラメータ側にも一時的に勾配が溜まる可能性がある（呼び出し側の挙動次第で副作用が残るリスク）。

推奨: 双方とも入力勾配だけが必要なので `torch.autograd.grad(loss, input)[0]` を使う方が明確で安全。

## 4. 正規化・逆正規化（pixel vs norm）の扱い

- package-bim
	- 正規化空間で勾配を計算し、ピクセル空間の勾配は `grad_pixel = grad_norm / std` として変換している（正しい変換）。

- legacy-bim
	- 同様に `grad_pixel = data_grad_norm / std` している点は正しいが、上記のように `data_grad_norm` が常に元画像由来であるため正しい意味を成さない。

## 5. 反復更新と射影（ε-ball クリップ）の扱い

- 両者ともに `perturbed = perturbed + alpha * sign(grad_pixel)` → `delta = clamp(perturbed - orig, -epsilon, +epsilon)` → `perturbed = clamp(orig + delta, 0, 1)` の順で更新しており、この点は一致している。

ただし legacy-bim の場合、grad_pixel が元画像の勾配由来なので、反復後の perturbation の意味が損なわれる。

## 6. re-attack 用ユーティリティの違い（bim_reattack）

- package-bim の `bim_reattack`:
	- 反復ごとに `x_adv_prime` を正規化して勾配を取り、ピクセル更新に変換している（内部実装と一貫）。

- legacy-bim の `bim_reattack`:
	- `x_adv_prime.requires_grad = True` のまま `outputs = model(x_adv_prime)` を実行している（正規化をしていない場合、モデルが正規化前の入力を期待しているなら OK、ただし全体での一貫性はない）。
	- こちらは `torch.autograd.grad` を使っており副作用が少ない点で優れているが、正規化の有無はモデルの前処理に依存するため注意が必要。

## 7. その他の実装上の差異

- package-bim は target を device に移し、スカラー target の時は unsqueeze している。API 利用側でバッチミスマッチを防ぐ配慮がある。
- legacy-bim は mean/std を固定で MNIST 用の値（[0.1307],[0.3081]）にしているため CIFAR 等では使いづらい（package-bim はチャンネル数に応じてデフォルト mean/std を選択する）。

## 8. まとめ — legacy-bim の「何が間違っていたか」

1. ループ内で勾配を評価する入力を `perturbed` ではなく常に `image_denorm`（元画像）にしていた点が最大のバグであり、BIM のアルゴリズムを破壊していた。
2. `loss.backward()` を使用しているためモデル勾配に副作用を与えるリスクがある（見かけ上は `model.zero_grad()` を呼んでいるが、より安全な `torch.autograd.grad` を用いるべき）。
3. mean/std が固定（MNIST）になっており汎用性が低い。

これらにより legacy-bim は「期待通りの反復的攻撃」を行わず、実験結果が劣化していた可能性が高い。

## 9. 修正提案（優先順）

1. legacy-bim のループ内で `image_norm` を `perturbed` に基づく正規化に変更する:
	 - `image_norm = (perturbed - mean) / std` にする。
2. 勾配取得を `torch.autograd.grad(loss, perturbed_norm)[0]` に変更してモデルパラメータの副作用を排除する。
3. mean/std をチャンネル数に応じて選ぶようにし、呼び出し元が指定できるようにする。
4. unit test を追加: ダミーモデルを使って `bim_attack` の出力が 0..1 の範囲にあり、反復後に perturbation が epsilon を超えないことを自動でチェックする。

## 10. 参考（実装例抜粋）
安全な反復のコア部分（擬似コード）:

```
perturbed = orig.clone()
for _ in range(num_iter):
		perturbed_norm = (perturbed - mean) / std
		perturbed_norm.requires_grad_(True)
		outputs = model(perturbed_norm)
		loss = loss_fn(outputs, target)
		grad_norm = torch.autograd.grad(loss, perturbed_norm)[0]
		grad_pixel = grad_norm / std
		with torch.no_grad():
				perturbed = perturbed + alpha * grad_pixel.sign()
				delta = clamp(perturbed - orig, -epsilon, epsilon)
				perturbed = clamp(orig + delta, 0, 1)

return perturbed
```

---

作成: 自動比較ツール + 手動レビュー

