use burn::backend::ndarray::NdArrayDevice;
/// AE攻撃用CLIバイナリ
///
/// 指定されたモデルとデータセットに対して敵対的攻撃を実行し、
/// 敵対的サンプルを生成します。
use burn::backend::wgpu::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::MnistDataset;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use clap::Parser;
use re_attack::attack::bim::BimAttack;
use re_attack::attack::fgsm::FgsmAttack;
use re_attack::attack::{AdversarialAttack, AttackParams};
use re_attack::cli::{AttackArgs, AttackMethod, BackendType, CommonArgs};
use re_attack::data::MnistBacher;
use re_attack::resnet18::ResNet18;
use re_attack::simple_mlp::SimpleMlp;
use re_attack::ClassificationModel;
use resnet_burn::ResNet18 as BurnResNet18;
use tracing::info;

#[derive(Debug, Parser)]
#[command(name = "ae_attack")]
#[command(about = "敵対的サンプル（AE）攻撃の実行", long_about = None)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    #[command(flatten)]
    attack: AttackArgs,

    /// 生成された敵対的サンプルを保存するディレクトリ
    #[arg(long, default_value = "./output/ae_attack")]
    output_dir: String,

    /// サンプル画像を保存するかどうか
    #[arg(long)]
    save_images: bool,
}

fn main() {
    let args = Cli::parse();

    // ログの初期化
    if args.common.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    info!("=== AE攻撃実験の開始 ===");
    info!("設定: {:?}", args);

    // バックエンドに応じて実行を分岐
    match args.common.backend {
        BackendType::Cpu => run_attack::<NdArray>(&args, NdArrayDevice::Cpu),
        BackendType::Wgpu => run_attack::<Wgpu>(&args, WgpuDevice::default()),
    }

    info!("=== AE攻撃実験の完了 ===");
}

fn run_attack<B: Backend>(args: &Cli, device: B::Device) {
    type MyAutodiffBackend<B> = Autodiff<B>;

    info!("デバイス: {:?}", device);

    // データセットの読み込み
    info!("データセットの読み込み中...");
    let dataset = MnistDataset::test();
    let total_samples = args.common.num_samples.unwrap_or(dataset.len());
    info!("処理サンプル数: {}", total_samples);

    let batcher = MnistBacher::default();

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(64)
        .num_workers(10)
        .build(MnistDataset::test());

    // モデルの初期化
    info!("モデルの初期化中...");
    let (model_name, model_resnet, model_mlp) = match args.common.model {
        re_attack::cli::ModelType::Resnet18 => (
            "ResNet18",
            Some(ResNet18::<MyAutodiffBackend<B>>::new(&device)),
            None,
        ),
        re_attack::cli::ModelType::SimpleMlp => (
            "SimpleMLP",
            None,
            Some(SimpleMlp::<MyAutodiffBackend<B>>::new(&device)),
        ),
    };
    info!("使用モデル: {}", model_name);

    // 攻撃手法の選択
    let attack_params = args.attack.to_attack_params();
    info!("攻撃パラメータ: {:?}", attack_params);

    // 攻撃の実行
    let mut correct = 0;
    let mut total = 0;
    for (index, batch) in dataloader_test.iter().enumerate() {
        // データの前処理
        let mean_val = 0.1307;
        let std_val = 0.3081;

        // let img_data = Tensor::<AutodiffBackend<B>, 2>::from_floats(item.image, &device);
        // let input = img_data.reshape([1, 1, 28, 28]).div_scalar(255.0);

        let mean_t = Tensor::<MyAutodiffBackend<B>, 1>::from_floats([mean_val], &device)
            .reshape([1, 1, 1, 1]);
        let std_t = Tensor::<MyAutodiffBackend<B>, 1>::from_floats([std_val], &device)
            .reshape([1, 1, 1, 1]);

        // let input_norm = input.clone().sub(mean_t.clone()).div(std_t.clone());
        // let target = Tensor::<AutodiffBackend<B>, 1, Int>::from_ints([item.label as i32], &device);
        let input = batch.images;
        let targets = batch.targets;
        let input_norm = input.clone().sub(mean_t.clone()).div(std_t.clone());

        // 攻撃の実行
        let perturbed_norm = match args.attack.attack {
            AttackMethod::Fgsm => {
                let attack = FgsmAttack::new();
                if let Some(ref model) = model_resnet {
                    attack.generate(
                        input_norm,
                        targets.clone(),
                        model,
                        &attack_params,
                        mean_t.clone(),
                        std_t.clone(),
                    )
                } else if let Some(ref model) = model_mlp {
                    attack.generate(
                        input_norm,
                        targets.clone(),
                        model,
                        &attack_params,
                        mean_t.clone(),
                        std_t.clone(),
                    )
                } else {
                    panic!("Model not initialized");
                }
            }
            AttackMethod::Bim => {
                let attack = BimAttack::new();
                if let Some(ref model) = model_resnet {
                    attack.generate(
                        input_norm,
                        targets.clone(),
                        model,
                        &attack_params,
                        mean_t.clone(),
                        std_t.clone(),
                    )
                } else if let Some(ref model) = model_mlp {
                    attack.generate(
                        input_norm,
                        targets.clone(),
                        model,
                        &attack_params,
                        mean_t.clone(),
                        std_t.clone(),
                    )
                } else {
                    panic!("Model not initialized");
                }
            }
        };

        // 予測の評価
        let output = if let Some(ref model) = model_resnet {
            model.forward(perturbed_norm.clone())
        } else if let Some(ref model) = model_mlp {
            model.forward(perturbed_norm.clone())
        } else {
            panic!("Model not initialized");
        };

        let predicted = output.argmax(1);
        let targets_data = targets.into_data();

        for (i, pred) in predicted.into_data().iter::<i32>().enumerate() {
            let target = targets_data.iter::<i32>().nth(i).unwrap();
            // let correct = if pred == target { "✓" } else { "✗" };
            // println!(
            //     "Sample {}: predicted={}, actual={} {}",
            //     i, pred, target, correct
            // );
            total += 1;
            if pred == target {
                correct += 1;
            }
        }
        // total += 1;
        // if predicted == item.label as usize {
        //     correct += 1;
        // }

        // if (idx + 1) % 10 == 0 {
        //     info!(
        //         "進捗: {}/{} サンプル処理完了, 現在の精度: {:.2}%",
        //         idx + 1,
        //         total_samples,
        //         (correct as f32 / total as f32) * 100.0
        //     );
        // }
        info!("バッチ処理完了 {}。", index + 1);
    }

    // for (idx, item) in dataset.iter().take(total_samples).enumerate() {
    //     // データの前処理
    //     let mean_val = 0.1307;
    //     let std_val = 0.3081;

    //     let img_data = Tensor::<MyAutodiffBackend<B>, 2>::from_floats(item.image, &device);
    //     let input = img_data.reshape([1, 1, 28, 28]).div_scalar(255.0);

    //     let mean_t = Tensor::<MyAutodiffBackend<B>, 1>::from_floats([mean_val], &device)
    //         .reshape([1, 1, 1, 1]);
    //     let std_t = Tensor::<MyAutodiffBackend<B>, 1>::from_floats([std_val], &device)
    //         .reshape([1, 1, 1, 1]);

    //     let input_norm = input.clone().sub(mean_t.clone()).div(std_t.clone());
    //     let target =
    //         Tensor::<MyAutodiffBackend<B>, 1, Int>::from_ints([item.label as i32], &device);

    //     // 攻撃の実行
    //     let perturbed_norm = match args.attack.attack {
    //         AttackMethod::Fgsm => {
    //             let attack = FgsmAttack::new();
    //             if let Some(ref model) = model_resnet {
    //                 attack.generate(
    //                     input_norm,
    //                     target.clone(),
    //                     model,
    //                     &attack_params,
    //                     mean_t.clone(),
    //                     std_t.clone(),
    //                 )
    //             } else if let Some(ref model) = model_mlp {
    //                 attack.generate(
    //                     input_norm,
    //                     target.clone(),
    //                     model,
    //                     &attack_params,
    //                     mean_t.clone(),
    //                     std_t.clone(),
    //                 )
    //             } else {
    //                 panic!("Model not initialized");
    //             }
    //         }
    //         AttackMethod::Bim => {
    //             let attack = BimAttack::new();
    //             if let Some(ref model) = model_resnet {
    //                 attack.generate(
    //                     input_norm,
    //                     target.clone(),
    //                     model,
    //                     &attack_params,
    //                     mean_t.clone(),
    //                     std_t.clone(),
    //                 )
    //             } else if let Some(ref model) = model_mlp {
    //                 attack.generate(
    //                     input_norm,
    //                     target.clone(),
    //                     model,
    //                     &attack_params,
    //                     mean_t.clone(),
    //                     std_t.clone(),
    //                 )
    //             } else {
    //                 panic!("Model not initialized");
    //             }
    //         }
    //     };

    //     // 予測の評価
    //     let output = if let Some(ref model) = model_resnet {
    //         model.forward(perturbed_norm.clone())
    //     } else if let Some(ref model) = model_mlp {
    //         model.forward(perturbed_norm.clone())
    //     } else {
    //         panic!("Model not initialized");
    //     };

    //     let predicted = output.argmax(1).into_scalar().elem::<i64>() as usize;
    //     total += 1;
    //     if predicted == item.label as usize {
    //         correct += 1;
    //     }

    //     if (idx + 1) % 10 == 0 {
    //         info!(
    //             "進捗: {}/{} サンプル処理完了, 現在の精度: {:.2}%",
    //             idx + 1,
    //             total_samples,
    //             (correct as f32 / total as f32) * 100.0
    //         );
    //     }
    // }

    let final_accuracy = (correct as f32 / total as f32) * 100.0;
    info!("=== 最終結果 ===");
    info!("正解数: {}/{}", correct, total);
    info!("攻撃後の精度: {:.2}%", final_accuracy);
    info!("攻撃成功率: {:.2}%", 100.0 - final_accuracy);
}
