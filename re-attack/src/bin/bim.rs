use burn::data::dataset::vision::MnistDataset;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{LinearConfig, PaddingConfig2d, conv::Conv2dConfig};
use burn::prelude::*;
use burn::{
    backend::{
        Autodiff, NdArray, Wgpu,
        wgpu::{Metal, WgpuDevice},
    },
    data::dataset::Dataset,
};
use re_attack::bim::bim_attack;
use re_attack::bim::print_ascii_art;
use re_attack::infer::load;

use re_attack::ARTIFACT_DIR;
use re_attack::resnet18::ResNet18;
use re_attack::simple_mlp::SimpleMlp;

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

fn main() {
    // CPUバックエンド (Ndarray) でAutodiffを有効化
    tracing_subscriber::fmt().init();
    // pwdを表示する

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::test(); // テストデータをダウンロード
    let item = dataset.iter().next().unwrap(); // 最初の画像（通常は '7'）を取得

    println!("Initializing Model...");
    // let model = ResNet18::<AutodiffBackend>::new(&device);
    let model = SimpleMlp::<AutodiffBackend>::new(&device);
    // let model = load::<AutodiffBackend>(ARTIFACT_DIR, &device);

    // データの前処理 (Normalize)
    // MNIST: [0, 255] -> [0.0, 1.0] -> Normalize((0.1307,), (0.3081,))
    let mean_val = 0.1307;
    let std_val = 0.3081;

    // 生データからTensor作成
    let img_data = Tensor::<AutodiffBackend, 2>::from_floats(item.image, &device);
    let input = img_data.reshape([1, 1, 28, 28]).div_scalar(255.0); // [0, 1]へ

    // 正規化用Tensor (Shape: [1, 1, 1, 1] for broadcasting)
    let mean_t =
        Tensor::<AutodiffBackend, 1>::from_floats([mean_val], &device).reshape([1, 1, 1, 1]);
    let std_t = Tensor::<AutodiffBackend, 1>::from_floats([std_val], &device).reshape([1, 1, 1, 1]);

    // 入力の正規化
    let input_norm = input.clone().sub(mean_t.clone()).div(std_t.clone());

    // ターゲットラベル
    let target = Tensor::<AutodiffBackend, 1, Int>::from_ints([item.label as i32], &device);

    println!("Original Image Label: {}", item.label);
    // 元画像の表示（非正規化して表示）
    print_ascii_art(input.clone(), "Original Image");

    println!("Running BIM Attack...");
    // 攻撃設定: eps=0.2 (かなり強めのノイズ), alpha=0.05, iter=10
    let perturbed_norm = bim_attack(
        input_norm,
        target,
        &model,
        0.8, // epsilon
        0.3, // alpha
        10,  // iter
        mean_t.clone(),
        std_t.clone(),
        0.0, // min pixel
        1.0, // max pixel
    );

    // 正規化を戻して表示用にする
    let perturbed = perturbed_norm.mul(std_t).add(mean_t);

    print_ascii_art(perturbed, "Adversarial Example (Perturbed)");

    println!("Done. ノイズ（点々や記号の変化）が追加されていることを確認してください。");
}
