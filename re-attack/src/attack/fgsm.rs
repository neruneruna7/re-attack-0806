/// FGSM (Fast Gradient Sign Method) 攻撃の実装
use super::{AdversarialAttack, AttackParams, ClassificationModel};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use tracing::info;

/// FGSM攻撃構造体
pub struct FgsmAttack;

impl FgsmAttack {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FgsmAttack {
    fn default() -> Self {
        Self::new()
    }
}

impl AdversarialAttack for FgsmAttack {
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

        // 1. 勾配計算の準備
        let input_req = orig_norm.clone().require_grad();

        // 2. Forward & Loss
        let output = model.forward(input_req.clone());
        let loss_fn = CrossEntropyLossConfig::new()
            .with_pad_tokens(None)
            .init(&device);
        let loss = loss_fn.forward(output, target);

        let loss_val = loss.clone().into_scalar();
        // info!("FGSM Loss = {:.4}", loss_val);

        // 3. Backward
        let grads = loss.backward();

        // 4. 勾配の取得と型の変換
        let grad_inner = input_req.grad(&grads).expect("Gradient not found");
        let grad = Tensor::<B, 4>::from_data(grad_inner.into_data(), &device);

        // 5. One-step update (Gradient Sign)
        let perturbation = grad.sign().mul(eps_norm.clone());
        let perturbed = orig_norm.add(perturbation);

        // 6. Domain Constraint (Pixel Value Clip)
        let result = perturbed.max_pair(min_val_norm).min_pair(max_val_norm);

        result
    }

    fn name(&self) -> &str {
        "FGSM"
    }
}
