use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::MnistBatch;

const NUM_CLASSES: usize = 10;

#[derive(Module, Debug)]
pub struct ResNet18<B: Backend> {
    // resnet_input: ResNetInput<B>,
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    activation: Relu,
    maxpool: MaxPool2d,

    layer1: BasicBlock<B>,
    layer2: BasicBlock<B>,
    layer3: BasicBlock<B>,
    layer4: BasicBlock<B>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
    // resnet_output: ResNetOutput<B>,
}

impl<B: Backend> ResNet18<B> {
    pub fn new(device: &B::Device) -> Self {
        ResNet18Config::new(NUM_CLASSES, 1, 64).init(device)
    }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = x.dims();
        // チャネル数1を加えて，4次元に変換
        let x = x.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.maxpool.forward(x);

        let x = self.layer1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.layer3.forward(x);
        let x = self.layer4.forward(x);

        let x = self.avgpool.forward(x);
        let [batch_size, channel, height, width] = x.dims();
        let x = x.reshape([batch_size, channel * height * width]);
        let x = self.fc.forward(x);
        x
    }

    pub fn forward_classification(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

#[derive(Config, Debug)]
pub struct ResNet18Config {
    #[config(default = 10)]
    num_classes: usize,
    hidden_size: usize,
    input_channel: usize,
    inplanes: usize,
    #[config(default = 1)]
    block_expansion: usize,
    #[config(default = 0.25)]
    dropout: f64,
}

impl ResNet18Config {
    pub fn init<B: Backend>(self, device: &B::Device) -> ResNet18<B> {
        ResNet18 {
            //channels[ ]は，多分入力チャネル，出力チャネル
            conv1: Conv2dConfig::new([self.input_channel, self.inplanes], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(self.inplanes).init(device),
            activation: Relu::new(),
            maxpool: MaxPool2dConfig::new([3, 3]).init(),

            layer1: BasicBlockConfig {
                in_planes: self.inplanes,
                out_planes: 64,
            }
            .init(device),
            layer2: BasicBlockConfig {
                in_planes: 64,
                out_planes: 128,
            }
            .init(device),
            layer3: BasicBlockConfig {
                in_planes: 128,
                out_planes: 256,
            }
            .init(device),
            layer4: BasicBlockConfig {
                in_planes: 256,
                out_planes: 512,
            }
            .init(device),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(512 * self.block_expansion, self.num_classes).init(device),
        }
    }
}

// fn conv3x3(in_planes: usize, out_planes: usize, stride: usize) -> Conv2dConfig {
//     Conv2dConfig::new([in_planes, out_planes], [3, 3])
//         .with_stride([1, 1])
//         .with_padding(PaddingConfig2d::)
// }
#[derive(Module, Debug)]
struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    // 正規化レイヤ
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    shortcut: Conv2d<B>,
    activation: Relu,
}

impl<B: Backend> BasicBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = x.clone();
        let shortcut = self.shortcut.forward(identity);

        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = x + shortcut;
        let x = self.activation.forward(x);
        x
    }
}

#[derive(Config, Debug)]
struct BasicBlockConfig {
    /// 入力チャネル数
    in_planes: usize,
    /// 出力チャネル数
    out_planes: usize,
}

impl BasicBlockConfig {
    /// 入力チャネル数，出力チャネル数，デバイス
    fn init<B: Backend>(&self, device: &B::Device) -> BasicBlock<B> {
        BasicBlock {
            conv1: Conv2dConfig::new([self.in_planes, self.out_planes], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(self.out_planes).init(device),
            conv2: Conv2dConfig::new([self.out_planes, self.out_planes], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(self.out_planes).init(device),
            // Use the block's input/output channel sizes for the shortcut 1x1 conv
            shortcut: Conv2dConfig::new([self.in_planes, self.out_planes], [1, 1]).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for ResNet18<B> {
    fn step(&self, item: MnistBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for ResNet18<B> {
    fn step(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
