use burn::{
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend, cast::ToElement},
};

use crate::{
    data::{MnistBacher, MnistBatch},
    resnet18::ResNet18,
    train::MnistTrainingConfig,
};

pub fn infer<B: AutodiffBackend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let model = load::<B>(artifact_dir, &device);
    let label = item.label;
    let batcher = MnistBacher::default();
    let batch = batcher.batch(vec![item], &device);

    let [batch_size, height, width] = batch.images.dims();
    // チャネル数1を加えて，4次元に変換
    let image = batch
        .images
        .reshape([batch_size, 1, height, width])
        .detach();

    let output = model.forward(image);

    let predicated = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("Predicted: {predicated}, Expected: {label}");
}

pub fn fgsm_attack<B: AutodiffBackend>(
    image: Tensor<B, 2>,
    epsilon: f32,
    data_grad: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let sign_data_grad = data_grad.sign();
    let perturbed_image = image + epsilon * sign_data_grad;
    perturbed_image.clamp(0.0, 1.0)
}

// pub fn output<B: AutodiffBackend>(
//     model: &ResNet18<B>,
//     item: MnistItem,
//     device: &B::Device,
// ) -> Tensor<B, 2> {
//     // let targets = item.targets;
//     let tensor = TensorData::from(item.image);
//     let tensor = Tensor::<B, 2>::from_data(tensor, &device);

//     let [batch_size, height, width] = item.images.dims();
//     // チャネル数1を加えて，4次元に変換
//     let image = item.images.reshape([batch_size, 1, height, width]).detach();

//     let output = model.forward(image);
//     output
// }

fn load<B: AutodiffBackend>(artifact_dir: &str, device: &B::Device) -> ResNet18<B> {
    // Load the model and perform inference
    // This is a placeholder for the actual inference logic
    let config = MnistTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("config.jsonがない");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .unwrap();

    let model = config.model.init::<B>(&device).load_record(record);
    model
}

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
    let data_grad = image.grad(&grad).unwrap();

    // 正規化解除
    let data_denorm = denorm::<B, 1>(image, [0.1307], [0.3081], &device);

    todo!()
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
