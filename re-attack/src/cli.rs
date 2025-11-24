/// CLI引数の共通構造体とユーティリティ
use clap::{Parser, ValueEnum};

/// 攻撃手法の選択
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum AttackMethod {
    /// FGSM (Fast Gradient Sign Method)
    Fgsm,
    /// BIM (Basic Iterative Method)
    Bim,
}

/// バックエンドの選択
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum BackendType {
    /// CPU backend (NdArray)
    Cpu,
    /// GPU backend (Wgpu)
    Wgpu,
}

/// データセットの選択
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DatasetType {
    /// MNIST dataset
    Mnist,
    // 将来的に追加: Cifar10, ImageNet など
}

/// モデルの選択
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ModelType {
    /// Simple MLP (Multi-Layer Perceptron)
    SimpleMlp,
    /// ResNet-18
    Resnet18,
}

/// 共通のCLI引数構造
#[derive(Debug, Clone, Parser)]
pub struct CommonArgs {
    /// データセット
    #[arg(long, default_value = "mnist")]
    pub dataset: DatasetType,

    /// モデルの種類
    #[arg(long, default_value = "resnet18")]
    pub model: ModelType,

    /// バックエンド
    #[arg(long, default_value = "wgpu")]
    pub backend: BackendType,

    /// 処理するサンプル数（デバッグ用）
    #[arg(long)]
    pub num_samples: Option<usize>,

    /// 詳細なログを出力
    #[arg(long, short = 'v')]
    pub verbose: bool,
}

/// 攻撃パラメータのCLI引数
#[derive(Debug, Clone, Parser)]
pub struct AttackArgs {
    /// 攻撃手法
    #[arg(long, default_value = "bim")]
    pub attack: AttackMethod,

    /// Epsilon: 許容される摂動の最大値
    #[arg(long, default_value = "0.3")]
    pub epsilon: f32,

    /// Alpha: 各イテレーションでの更新ステップサイズ (BIMのみ)
    #[arg(long, default_value = "0.01")]
    pub alpha: f32,

    /// イテレーション回数 (BIMのみ)
    #[arg(long, default_value = "10")]
    pub num_iter: usize,
}

impl AttackArgs {
    /// AttackParamsへの変換
    pub fn to_attack_params(&self) -> crate::attack::AttackParams {
        crate::attack::AttackParams {
            epsilon: self.epsilon,
            alpha: self.alpha,
            num_iter: self.num_iter,
            min_val: 0.0,
            max_val: 1.0,
        }
    }
}
