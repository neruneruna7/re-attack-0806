/// BIM (Basic Iterative Method) 攻撃の実装
use super::{AdversarialAttack, AttackParams, ClassificationModel};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use tracing::info;

/// BIM攻撃構造体
pub struct BimAttack;

impl BimAttack {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BimAttack {
    fn default() -> Self {
        Self::new()
    }
}

impl AdversarialAttack for BimAttack {
    fn generate<B, M>(
        &self,
        input: Tensor<B, 4>,
        target: Tensor<B, 1, Int>,
        model: &M,
        params: &AttackParams,
        mean: Tensor<B, 4>,
        std: Tensor<B, 4>,
    ) -> Tensor<B, 4>
    where
        B: AutodiffBackend,
        M: ClassificationModel<B>,
    {
        let device = input.device();

        // 攻撃パラメータを正規化空間に変換
        let eps_norm = std.clone().recip().mul_scalar(params.epsilon);
        let alpha_norm = std.clone().recip().mul_scalar(params.alpha);

        // クリップ用の境界値を正規化空間に変換
        let min_val_norm = mean
            .clone()
            .mul_scalar(-1.0)
            .add_scalar(params.min_val)
            .div(std.clone());
        let max_val_norm = mean
            .clone()
            .mul_scalar(-1.0)
            .add_scalar(params.max_val)
            .div(std.clone());

        let orig_norm = input.clone();

        // 摂動を加える画像。初期値は入力画像。
        let mut perturbed = input.clone().detach();

        let loss_fn = CrossEntropyLossConfig::new()
            .with_pad_tokens(None)
            .init(&device);

        for i in 0..params.num_iter {
            // 1. 勾配計算の準備
            let input_req = perturbed.clone().require_grad();

            // 2. Forward & Loss
            let output = model.forward(input_req.clone());
            let loss = loss_fn.forward(output, target.clone());

            let loss_val = loss.clone().into_scalar();
            // info!("BIM Iter {}: Loss = {:.4}", i, loss_val);

            // 3. Backward
            let grads = loss.backward();

            // 4. 勾配の取得と型の変換
            let grad_inner = input_req.grad(&grads).expect("Gradient not found");
            let grad = Tensor::<B, 4>::from_data(grad_inner.into_data(), &device);

            // 5. Update (Gradient Ascent)
            let update = grad.sign().mul(alpha_norm.clone());
            let perturbed_temp = perturbed.add(update);

            // 6. Projection (Epsilon Ball Clip)
            let delta = perturbed_temp.sub(orig_norm.clone());
            let neg_eps_norm = eps_norm.clone().mul_scalar(-1.0);
            let delta_clipped = delta.max_pair(neg_eps_norm).min_pair(eps_norm.clone());

            let candidate = orig_norm.clone().add(delta_clipped);

            // 7. Domain Constraint (Pixel Value Clip)
            perturbed = candidate
                .max_pair(min_val_norm.clone())
                .min_pair(max_val_norm.clone());
        }

        perturbed
    }

    fn name(&self) -> &str {
        "BIM"
    }
}
