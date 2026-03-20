/// AE再攻撃による防御実験用CLIバイナリ
/// 
/// 初期攻撃で生成した敵対的サンプルに対して再攻撃を行い、
/// 防御効果を評価します。

use burn::backend::wgpu::Wgpu;
use burn::backend::{Autodiff, NdArray};
use burn::data::dataset::vision::MnistDataset;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::ndarray::NdArrayDevice;
use clap::Parser;
use re_attack::attack::{AdversarialAttack, AttackParams};
use re_attack::attack::bim::BimAttack;
use re_attack::attack::fgsm::FgsmAttack;
use re_attack::cli::{AttackArgs, AttackMethod, BackendType, CommonArgs};
use re_attack::resnet18::ResNet18;
use re_attack::simple_mlp::SimpleMlp;
use re_attack::ClassificationModel;
use tracing::info;

#[derive(Debug, Parser)]
#[command(name = "ae_reattack_defense")]
#[command(about = "AE再攻撃による防御実験", long_about = None)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,

    /// 初期攻撃のパラメータ
    #[command(flatten)]
    initial_attack: AttackArgs,

    /// 再攻撃の手法
    #[arg(long, default_value = "bim")]
    reattack_method: AttackMethod,

    /// 再攻撃のEpsilon
    #[arg(long, default_value = "0.1")]
    reattack_epsilon: f32,

    /// 再攻撃のAlpha
    #[arg(long, default_value = "0.02")]
    reattack_alpha: f32,

    /// 再攻撃のイテレーション回数
    #[arg(long, default_value = "20")]
    reattack_num_iter: usize,
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

    info!("=== AE再攻撃防御実験の開始 ===");
    info!("設定: {:?}", args);

    // バックエンドに応じて実行を分岐
    match args.common.backend {
        BackendType::Cpu => run_reattack::<NdArray>(&args, NdArrayDevice::Cpu),
        BackendType::Wgpu => run_reattack::<Wgpu>(&args, WgpuDevice::default()),
    }

    info!("=== AE再攻撃防御実験の完了 ===");
}

fn run_reattack<B: Backend>(args: &Cli, device: B::Device) {
    type AutodiffBackend<B> = Autodiff<B>;

    info!("デバイス: {:?}", device);

    // データセットの読み込み
    info!("データセットの読み込み中...");
    let dataset = MnistDataset::test();
    let total_samples = args.common.num_samples.unwrap_or_else(|| dataset.len().min(100));
    info!("処理サンプル数: {}", total_samples);

    // モデルの初期化
    info!("モデルの初期化中...");
    let (model_name, model_resnet, model_mlp) = match args.common.model {
        re_attack::cli::ModelType::Resnet18 => {
            (
                "ResNet18",
                Some(ResNet18::<AutodiffBackend<B>>::new(&device)),
                None,
            )
        }
        re_attack::cli::ModelType::SimpleMlp => {
            (
                "SimpleMLP",
                None,
                Some(SimpleMlp::<AutodiffBackend<B>>::new(&device)),
            )
        }
    };
    info!("使用モデル: {}", model_name);

    // 攻撃パラメータの設定
    let initial_params = args.initial_attack.to_attack_params();
    let reattack_params = AttackParams {
        epsilon: args.reattack_epsilon,
        alpha: args.reattack_alpha,
        num_iter: args.reattack_num_iter,
        min_val: 0.0,
        max_val: 1.0,
    };

    info!("初期攻撃パラメータ: {:?}", initial_params);
    info!("再攻撃パラメータ: {:?}", reattack_params);

    let mut original_correct = 0;
    let mut after_initial_attack_correct = 0;
    let mut after_reattack_correct = 0;
    let mut total = 0;

    for (idx, item) in dataset.iter().take(total_samples).enumerate() {
        // データの前処理
        let mean_val = 0.1307;
        let std_val = 0.3081;

        let img_data = Tensor::<AutodiffBackend<B>, 2>::from_floats(item.image, &device);
        let input = img_data.reshape([1, 1, 28, 28]).div_scalar(255.0);

        let mean_t = Tensor::<AutodiffBackend<B>, 1>::from_floats([mean_val], &device)
            .reshape([1, 1, 1, 1]);
        let std_t = Tensor::<AutodiffBackend<B>, 1>::from_floats([std_val], &device)
            .reshape([1, 1, 1, 1]);

        let input_norm = input.clone().sub(mean_t.clone()).div(std_t.clone());
        let target = Tensor::<AutodiffBackend<B>, 1, Int>::from_ints([item.label as i32], &device);

        // オリジナルの予測
        let original_output = if let Some(ref model) = model_resnet {
            model.forward(input_norm.clone())
        } else if let Some(ref model) = model_mlp {
            model.forward(input_norm.clone())
        } else {
            panic!("Model not initialized");
        };
        let original_pred = original_output.argmax(1).into_scalar().elem::<i64>() as usize;

        // 初期攻撃の実行
        let ae_sample = execute_attack::<B>(
            &args.initial_attack.attack,
            input_norm.clone(),
            target.clone(),
            &model_resnet,
            &model_mlp,
            &initial_params,
            mean_t.clone(),
            std_t.clone(),
        );

        // 初期攻撃後の予測
        let ae_output = if let Some(ref model) = model_resnet {
            model.forward(ae_sample.clone())
        } else if let Some(ref model) = model_mlp {
            model.forward(ae_sample.clone())
        } else {
            panic!("Model not initialized");
        };
        let ae_pred = ae_output.argmax(1).into_scalar().elem::<i64>() as usize;

        // 再攻撃の実行（AEサンプルに対して）
        let reattack_sample = execute_attack::<B>(
            &args.reattack_method,
            ae_sample,
            target.clone(),
            &model_resnet,
            &model_mlp,
            &reattack_params,
            mean_t.clone(),
            std_t.clone(),
        );

        // 再攻撃後の予測
        let reattack_output = if let Some(ref model) = model_resnet {
            model.forward(reattack_sample)
        } else if let Some(ref model) = model_mlp {
            model.forward(reattack_sample)
        } else {
            panic!("Model not initialized");
        };
        let reattack_pred = reattack_output.argmax(1).into_scalar().elem::<i64>() as usize;

        // 結果の集計
        total += 1;
        if original_pred == item.label as usize {
            original_correct += 1;
        }
        if ae_pred == item.label as usize {
            after_initial_attack_correct += 1;
        }
        if reattack_pred == item.label as usize {
            after_reattack_correct += 1;
        }

        if (idx + 1) % 10 == 0 {
            info!(
                "進捗: {}/{} サンプル処理完了",
                idx + 1,
                total_samples
            );
        }
    }

    // 最終結果の表示
    info!("=== 最終結果 ===");
    info!("元画像の精度: {:.2}% ({}/{})", 
        (original_correct as f32 / total as f32) * 100.0,
        original_correct, total);
    info!("初期攻撃後の精度: {:.2}% ({}/{})", 
        (after_initial_attack_correct as f32 / total as f32) * 100.0,
        after_initial_attack_correct, total);
    info!("再攻撃後の精度: {:.2}% ({}/{})", 
        (after_reattack_correct as f32 / total as f32) * 100.0,
        after_reattack_correct, total);
    
    let defense_recovery = after_reattack_correct as f32 - after_initial_attack_correct as f32;
    info!("防御による回復: {} サンプル ({:.2}%)", 
        defense_recovery,
        (defense_recovery / total as f32) * 100.0);
}

fn execute_attack<B: Backend>(
    method: &AttackMethod,
    input: Tensor<Autodiff<B>, 4>,
    target: Tensor<Autodiff<B>, 1, Int>,
    model_resnet: &Option<ResNet18<Autodiff<B>>>,
    model_mlp: &Option<SimpleMlp<Autodiff<B>>>,
    params: &AttackParams,
    mean: Tensor<Autodiff<B>, 4>,
    std: Tensor<Autodiff<B>, 4>,
) -> Tensor<Autodiff<B>, 4> {
    match method {
        AttackMethod::Fgsm => {
            let attack = FgsmAttack::new();
            if let Some(ref model) = model_resnet {
                attack.generate(input, target, model, params, mean, std)
            } else if let Some(ref model) = model_mlp {
                attack.generate(input, target, model, params, mean, std)
            } else {
                panic!("Model not initialized");
            }
        }
        AttackMethod::Bim => {
            let attack = BimAttack::new();
            if let Some(ref model) = model_resnet {
                attack.generate(input, target, model, params, mean, std)
            } else if let Some(ref model) = model_mlp {
                attack.generate(input, target, model, params, mean, std)
            } else {
                panic!("Model not initialized");
            }
        }
    }
}
