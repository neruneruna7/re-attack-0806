use std::{fs::create_dir_all, path::Path};

use burn::{
    data::{dataloader::batcher::Batcher as _, dataset::vision::MnistItem},
    prelude::Backend,
    tensor::{Int, Tensor, backend::AutodiffBackend, cast::ToElement as _},
};
use image::{GrayImage, Luma};
use tracing::{info, instrument};

use crate::{data::MnistBacher, infer::load};

const STD: f64 = 0.3081;
const MEAN: f64 = 0.1307;

#[instrument(skip(artifact_dir, device, items))]
pub fn fgsm<B: AutodiffBackend>(
    artifact_dir: &str,
    device: B::Device,
    epsilon: f32,
    items: &[MnistItem],
) -> (usize, usize) {
    let model = load::<B>(artifact_dir, &device);

    let mut result = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let label = item.label;
        let batch = MnistBacher::default().batch(vec![item.clone()], &device);
        let [batch_size, height, width] = batch.images.dims();
        // チャネル数1を加えて，4次元に変換
        let image = batch.images.clone().reshape([batch_size, 1, height, width]);
        // .detach();

        let output_adv = fgsm_inner(&device, &model, batch.targets, image, epsilon).detach();
        // let predicated = output.argmax(1).flatten::<1>(0, 1).into_scalar();
        let final_pred = output_adv.argmax(1).flatten::<1>(0, 1).into_scalar();

        result.push((label, final_pred.to_u8()));
        // println!("Processed {}/{}", i + 1, items.len());
        info!("Processed {}/{}", i + 1, items.len());
    }

    let mut correct = 0;
    for (label, final_pred) in result {
        // println!("Predicted: {predicated}, Expected: {label}");
        // if predicated.to_u8() != label {
        //     continue;
        //     // println!("The model already misclassified the image. No attack needed.");
        // }

        // println!("Final Predicted: {final_pred}, Expected: {label}");
        if final_pred.to_u8() == label {
            correct += 1;
            // println!("The attack failed.");
        } else {
            // println!("The attack succeeded.");
        }
    }

    let final_acc = correct as f64 / items.len() as f64;
    println!(
        "Epsilon: {epsilon} \t Test Accuracy: {correct} / {}  {final_acc}",
        items.len()
    );
    (correct, items.len())
    // print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}"
}

fn fgsm_inner<B: AutodiffBackend>(
    device: &<B as Backend>::Device,
    model: &crate::resnet18::ResNet18<B>,
    target: Tensor<B, 1, Int>,
    image: Tensor<B, 4>,
    epsilon: f32,
) -> Tensor<B, 2> {
    let image = image.require_grad();

    // println!("Original image: {image}");
    let output = model.forward(image.clone());

    // 損失計算
    // pytorchでな負の対数尤度を使ってた
    // 伴さんのではクロスエントロピーだった
    // クロスエントロピーを使ってみよう
    // let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new().init::<B>(&device);
    // let loss = loss_fn.forward(output, target);

    // 逆伝播
    // let grad = loss.backward();
    let grad = output.backward();
    let data_grad: Tensor<<B as AutodiffBackend>::InnerBackend, 4> = image.grad(&grad).unwrap();
    // info!("loss: {}", &loss);
    info!("dy/dx: {}", &data_grad);

    // // ===== デバッグ出力（ここでまず注目） =====
    // // 勾配の要約（min, max, L1 sum）
    // let grad_min = data_grad.clone().min();
    // let grad_max = data_grad.clone().max();
    // let grad_abs_sum = data_grad.clone().abs().sum();
    // info!(
    //     "data_grad stats -> min: {:?}, max: {:?}, abs_sum: {:?}",
    //     grad_min, grad_max, grad_abs_sum
    // );

    // let grad_norm = data_grad.clone().abs().abs().sum();
    // info!("data_grad L1 norm: {:?}", grad_norm);

    // // sign とその要約（差分の大きさを確認）
    // let sign_data_grad = data_grad.clone().sign();
    // let sign_abs_sum = sign_data_grad.clone().abs().sum();
    // info!(
    //     "sign_data_grad abs_sum (nonzero count proxy): {:?}",
    //     sign_abs_sum
    // );

    // // 正規化解除
    // let data_denorm = denorm::<B, 1>(image, [MEAN], [STD], &device);
    // // 自動微分バックエンドではなく，通常のバックエンドに変換
    // let data_denorm = data_denorm.inner();

    // let tmp = data_denorm.clone();
    todo!()

    // fgsm攻撃
    // let perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad);

    // let a = Tensor::<B, 4>::from_inner(perturbed_data.clone());
    // let b = Tensor::<B, 4>::from_inner(tmp);
    // let diff = (a.clone() - b).abs().sum();
    // let pert_min = a.clone().min();
    // let pert_max = a.clone().max();
    // info!(
    //     "perturbed_data stats -> min: {:?}, max: {:?}, diff from original abs_sum: {:?}",
    //     pert_min, pert_max, diff
    // );
    // #[allow(unused_variables)]
    // {
    //     // Try: into_data (consuming)
    //     let tensor_data = perturbed_data.clone().into_data();
    //     let maybe_vec: Option<Vec<f32>> = match tensor_data {
    //         // Burn のバージョンによってバリアント名が異なることがあるため複数候補を試す
    //         // うまくいかない場合はコンパイルエラーの出力を貼ってください
    //         burn::tensor::TensorData::Float(v) => Some(v),
    //         burn::tensor::TensorData::F32(v) => Some(v),
    //         // 予備: Primitive 名義の場合（バージョン依存）
    //         // burn::tensor::TensorData::Primitive(burn::tensor::Primitive::Float(v)) => Some(v),
    //         _ => {
    //             info!("unexpected TensorData variant, cannot extract Vec<f32>");
    //             None
    //         }
    //     };
    //     if let Some(flat) = maybe_vec {
    //         // shape: [1,1,H,W]
    //         let h = flat.len() / (1 * 1 * 28); // adjust if not 28
    //         let w = 28;
    //         let out_dir = format!("{}/out_images", crate::ARTIFACT_DIR);
    //         let fname = format!("{}/perturbed_e{:.3}.png", out_dir, epsilon);
    //         if let Err(e) = save_tensor_gray(&fname, &flat, h, w) {
    //             info!("failed to save perturbed image: {:?}", e);
    //         } else {
    //             info!("saved perturbed image to {}", fname);
    //         }
    //     } else {
    //         // API mismatch -> ユーザーに知らせるためログを出す
    //         info!(
    //             "could not extract Vec<f32> from perturbed_data: please adapt `.into_data()` call for your burn version"
    //         );
    //     }
    // }

    // let perturbed_data_normalized = (perturbed_data - MEAN) / STD;
    // // 自動微分バックエンドに再変換
    // let perturbed_data_normalized = Tensor::from_inner(perturbed_data_normalized);
    // // println!("Attack applied. {perturbed_data_normalized}");
    // // 再分類
    // let output_adv = model.forward(perturbed_data_normalized);

    // // // 最終予測をログ出力（バッチサイズ1を前提）
    // // let final_pred = output_adv
    // //     .clone()
    // //     .argmax(1)
    // //     .flatten::<1>(0, 1)
    // //     .into_scalar();
    // // info!("final_pred: {:?}", final_pred);

    // output_adv
}

fn denorm<B: AutodiffBackend, const C: usize>(
    tensor: Tensor<B, 4>,
    mean: [f64; C],
    std: [f64; C],
    device: &B::Device,
) -> Tensor<B, 4> {
    let mean_b = Tensor::<B, 1>::from_floats(mean, device).reshape([1, -1, 1, 1]);
    let std_b = Tensor::<B, 1>::from_floats(std, device).reshape([1, -1, 1, 1]);
    tensor * std_b + mean_b
}

#[instrument(skip(image, data_grad))]
pub fn fgsm_attack<B: Backend>(
    image: Tensor<B, 4>,
    epsilon: f32,
    data_grad: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let sign_data_grad = data_grad.sign();
    info!("sign_data_grad: {:?}", sign_data_grad);
    let perturbed_image = image + epsilon * sign_data_grad;
    perturbed_image.clamp(0.0, 1.0)
}

fn save_tensor_gray(path: &str, data: &[f32], height: usize, width: usize) -> anyhow::Result<()> {
    create_dir_all(Path::new(path).parent().unwrap_or(Path::new(".")))?;
    let mut img = GrayImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let v = data[y * width + x];
            // clip to [0,1] then map to 0..255
            let v = if v.is_finite() {
                v.max(0.0).min(1.0)
            } else {
                0.0
            };
            let px = (v * 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, Luma([px]));
        }
    }
    img.save(path)?;
    Ok(())
}
