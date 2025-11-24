use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Gelu, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    tensor::{BasicAutodiffOps, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use tracing::{info, instrument};

use crate::{ClassificationModel, data::MnistBatch};

const NUM_CLASSES: usize = 10;

// --- SimpleMlp モデル定義 (ここからコピー) ---

#[derive(Module, Debug)]
pub struct SimpleMlp<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: nn::Gelu, // ReLUより勾配が通りやすいGeluを採用
}

impl<B: Backend> SimpleMlp<B> {
    pub fn new(device: &B::Device) -> Self {
        // 入力: 28x28 = 784次元
        // 隠れ層: 256次元
        let linear1 = LinearConfig::new(784, 256).init(device);
        // 出力: 10クラス
        let linear2 = LinearConfig::new(256, 10).init(device);

        Self {
            linear1,
            linear2,
            activation: nn::Gelu::new(),
        }
    }

    pub fn forward_classification(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        let targets = batch.targets;
        let [batch_size, height, width] = batch.images.dims();
        // チャネル数1を加えて，4次元に変換
        let image = batch
            .images
            .reshape([batch_size, 1, height, width])
            .detach();

        let output = self.forward(image);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

// トレイトの実装
impl<B: Backend> ClassificationModel<B> for SimpleMlp<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // [Batch, 1, 28, 28] -> [Batch, 784] にフラット化
        let x = input.flatten(1, 3);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);

        x // Logits
    }
}

// --- (ここまで) ---

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for SimpleMlp<B> {
    fn step(&self, item: MnistBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for SimpleMlp<B> {
    fn step(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
