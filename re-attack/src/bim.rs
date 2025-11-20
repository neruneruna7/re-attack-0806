use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use tracing::info;

/// モデルがBIM攻撃に対応するために実装すべきトレイト
/// BurnのModuleはデフォルトではforwardを強制しないため、これを定義して呼び出しを保証します。
pub trait ClassificationModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2>;
}

/// BIM (Basic Iterative Method) 攻撃の実装
pub fn bim_attack<B, M>(
    input: Tensor<B, 4>,
    target: Tensor<B, 1, Int>,
    model: &M,
    eps: f32,
    alpha: f32,
    num_iter: usize,
    mean: Tensor<B, 4>,
    std: Tensor<B, 4>,
    min_val: f32,
    max_val: f32,
) -> Tensor<B, 4>
where
    B: AutodiffBackend,
    M: ClassificationModel<B>,
{
    let device = input.device();

    // 攻撃パラメータを正規化空間に変換
    // eps_norm = eps / std
    let eps_norm = std.clone().recip().mul_scalar(eps);
    let alpha_norm = std.clone().recip().mul_scalar(alpha);

    // クリップ用の境界値を正規化空間に変換
    let min_val_norm = mean
        .clone()
        .mul_scalar(-1.0)
        .add_scalar(min_val)
        .div(std.clone());
    let max_val_norm = mean
        .clone()
        .mul_scalar(-1.0)
        .add_scalar(max_val)
        .div(std.clone());

    let orig_norm = input.clone();

    // 摂動を加える画像。初期値は入力画像。
    // 計算グラフを切るために .clone() しますが、BurnのTensorはClone時もIDを共有する場合があるため、
    // 明示的に detach() を呼ぶのが最も安全です。
    let mut perturbed = input.clone().detach();

    let loss_fn = CrossEntropyLossConfig::new()
        .with_pad_tokens(None)
        .init(&device);

    for _i in 0..num_iter {
        // 1. 勾配計算の準備
        // 現在のperturbedを計算グラフの「葉（Leaf）」として登録
        let input_req = perturbed.clone().require_grad();

        if _i == 0 {
            let dummy_loss = input_req.clone().sum();
            let dummy_grads = dummy_loss.backward();
            let dummy_grad = input_req.grad(&dummy_grads).unwrap();
            let dummy_grad_tensor = Tensor::<B, 4>::from_data(dummy_grad.into_data(), &device);
            let max_grad = dummy_grad_tensor.max().into_scalar();
            println!(
                "SANITY CHECK: Dummy Gradient Max = {:.4} (Should be 1.0)",
                max_grad
            );
        }

        // 2. Forward & Loss
        let output = model.forward(input_req.clone());
        let loss = loss_fn.forward(output, target.clone());

        let loss_val = loss.clone().into_scalar();
        println!("Iter {}: Loss = {:.4}", _i, loss_val);

        // 3. Backward
        let grads = loss.backward();

        // 4. 勾配の取得と型の変換 (重要!)
        // grad は Tensor<B::InnerBackend, 4> (生データ) です。
        // これを Tensor<B, 4> (微分可能バックエンド) に変換しないと perturbed と演算できません。
        let grad_inner = input_req.grad(&grads).expect("Gradient not found");

        // InnerBackendのデータをAutodiffBackendの世界に持ち上げます。
        // from_dataを経由することで、バックエンドの実装詳細（AutodiffTensorの構造など）に依存せず変換できます。
        // (パフォーマンスを極限まで高める場合はPrimitiveを直接扱いますが、通常はこれで十分です)
        let grad = Tensor::<B, 4>::from_data(grad_inner.into_data(), &device);

        let (min_g, max_g) = (
            grad.clone().min().into_scalar(),
            grad.clone().max().into_scalar(),
        );
        println!("Iter {}: Grad Range = [{:.6}, {:.6}]", _i, min_g, max_g);

        // 5. Update (Gradient Ascent)
        // perturbed = perturbed + alpha * sign(grad)
        let update = grad.sign().mul(alpha_norm.clone());

        // 加算される勾配を表示
        info!("Update Norm: ");
        info!("{}", &update);

        // input_reqではなく、detachedなperturbedに対して加算し、新しいdetachedなTensorを作る
        let perturbed_temp = perturbed.add(update);

        // 6. Projection (Epsilon Ball Clip)
        // delta = perturbed - orig
        let delta = perturbed_temp.sub(orig_norm.clone());

        // Tensor同士のClamp: max_pair / min_pair を使用
        // delta = clamp(delta, -eps, eps)
        let neg_eps_norm = eps_norm.clone().mul_scalar(-1.0);
        let delta_clipped = delta
            .max_pair(neg_eps_norm) // element-wise max
            .min_pair(eps_norm.clone()); // element-wise min

        // 元の画像にクリップされた摂動を加算
        let candidate = orig_norm.clone().add(delta_clipped);

        // 7. Domain Constraint (Pixel Value Clip)
        // 0.0 ~ 1.0 の範囲（正規化後）に収める
        perturbed = candidate
            .max_pair(min_val_norm.clone())
            .min_pair(max_val_norm.clone());

        // ループの最後で perturbed は計算グラフから切り離された状態（detach相当）になります。
        // 次のループの冒頭で .require_grad() されることで、新たなグラフが始まります。
    }

    perturbed
}

pub fn print_ascii_art<B: Backend>(tensor: Tensor<B, 4>, title: &str) {
    // Tensor [1, 1, 28, 28] -> Vec<f32>
    let data = tensor.into_data();
    let vector = data.to_vec::<f32>().expect("Should be f32");

    println!("\n=== {} ===", title);
    for y in 0..28 {
        for x in 0..28 {
            let val = vector[y * 28 + x];
            // 簡易的な輝度マップ
            let char = if val < 0.2 {
                " "
            } else if val < 0.5 {
                "."
            } else if val < 0.8 {
                "+"
            } else {
                "#"
            };
            print!("{} ", char);
        }
        println!();
    }
}
