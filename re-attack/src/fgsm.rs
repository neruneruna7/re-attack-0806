use burn::{
    data::{dataloader::batcher::Batcher as _, dataset::vision::MnistItem},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend, cast::ToElement as _},
};

use crate::{data::MnistBacher, infer::load};

const STD: f64 = 0.3081;
const MEAN: f64 = 0.1307;

pub fn fgsm<B: AutodiffBackend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let model = load::<B>(artifact_dir, &device);

    let label = item.label;
    let batch = MnistBacher::default().batch(vec![item.clone()], &device);
    let [batch_size, height, width] = batch.images.dims();
    // チャネル数1を加えて，4次元に変換
    let image = batch
        .images
        .reshape([batch_size, 1, height, width])
        .detach();

    let image = image.require_grad();
    let output = model.forward(image.clone());

    let predicated = output.clone().argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted: {predicated}, Expected: {label}");
    if predicated.to_u8() != label {
        println!("The model already misclassified the image. No attack needed.");
    }

    // 損失計算
    // pytorchでな負の対数尤度を使ってた
    // 伴さんのではクロスエントロピーだった
    // クロスエントロピーを使ってみよう
    let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new().init::<B>(&device);
    let loss = loss_fn.forward(output, batch.targets);

    // 逆伝播
    let grad = loss.backward();
    let data_grad: Tensor<<B as AutodiffBackend>::InnerBackend, 4> = image.grad(&grad).unwrap();

    // 正規化解除
    let data_denorm = denorm::<B, 1>(image, [MEAN], [STD], &device);
    // 自動微分バックエンドではなく，通常のバックエンドに変換
    let data_denorm = data_denorm.inner();

    // fgsm攻撃
    let perturbed_data = fgsm_attack(data_denorm, 0.3, data_grad);

    let perturbed_data_normalized = (perturbed_data - MEAN) / STD;
    // 自動微分バックエンドに再変換
    let perturbed_data_normalized = Tensor::from_inner(perturbed_data_normalized);
    // 再分類
    let output_adv = model.forward(perturbed_data_normalized);

    let final_pred = output_adv.argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("Final Predicted: {final_pred}, Expected: {label}");
    if final_pred.to_u8() == label {
        println!("The attack failed.");
    } else {
        println!("The attack succeeded.");
    }
}
fn denorm<B: AutodiffBackend, const C: usize>(
    tensor: Tensor<B, 4>,
    mean: [f64; C],
    std: [f64; C],
    device: &B::Device,
) -> Tensor<B, 4> {
    let mean_b = Tensor::<B, 1>::from_floats(mean, device).reshape([1, C, 1, 1]);
    let std_b = Tensor::<B, 1>::from_floats(std, device).reshape([1, C, 1, 1]);
    tensor * std_b + mean_b
}

pub fn fgsm_attack<B: Backend>(
    image: Tensor<B, 4>,
    epsilon: f32,
    data_grad: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let sign_data_grad = data_grad.sign();
    let perturbed_image = image + epsilon * sign_data_grad;
    perturbed_image.clamp(0.0, 1.0)
}
