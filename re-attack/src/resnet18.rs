use burn::{
    config,
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    train::ClassificationOutput,
};

use crate::data::MnistBatch;

const NUM_CLASSES: usize = 10;

#[derive(Module, Debug)]
pub struct ResNet18<B: Backend> {
    resnet_input: ResNetInput<B>,
    layers: Vec<BasicBlock<B>>,
    resnet_output: ResNetOutput<B>,
    activation: Relu,
}

impl<B: Backend> ResNet18<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = x.dims();
        // チャネル数1を加えて，4次元に変換
        let x = x.reshape([batch_size, 1, height, width]).detach();
        let x = self.resnet_input.forward(x);
        let x = self.layers.iter().fold(x, |out, layer| layer.forward(out));

        self.resnet_output.forward(x)
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
    layers: usize,
    #[config(default = 0.25)]
    dropout: f64,
}

impl ResNet18Config {
    pub fn init<B: Backend>(self, device: &B::Device) -> ResNet18<B> {
        let layers = (0..self.layers)
            .map(|_| {
                BasicBlockConfig {
                    in_planes: self.inplanes,
                    out_planes: self.inplanes,
                }
                .init(device)
            })
            .collect::<Vec<_>>();
        ResNet18 {
            //channels[ ]は，多分入力チャネル，出力チャネル
            resnet_input: ResNetInputConfig {
                input_channel: self.input_channel,
                in_planes: self.inplanes,
            }
            .init(device),
            layers,
            resnet_output: ResNetOutputConfig {
                block_expansion: 1,
                num_classes: self.num_classes,
            }
            .init(device),
            activation: Relu::new(),
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
            conv1: Conv2dConfig::new([self.in_planes, self.out_planes], [3, 3]).init(device),
            bn1: BatchNormConfig::new(self.out_planes).init(device),
            conv2: Conv2dConfig::new([self.out_planes, self.out_planes], [3, 3]).init(device),
            bn2: BatchNormConfig::new(self.out_planes).init(device),
            shortcut: Conv2dConfig::new([1, 8], [1, 1]).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
struct ResNetInput<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    activation: Relu,
    pool: MaxPool2d,
}

impl<B: Backend> ResNetInput<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        x
    }
}

#[derive(Config, Debug)]
struct ResNetInputConfig {
    input_channel: usize,
    in_planes: usize,
}

impl ResNetInputConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ResNetInput<B> {
        ResNetInput {
            conv1: Conv2dConfig::new([self.input_channel, self.in_planes], [3, 3]).init(device),
            bn1: BatchNormConfig::new(self.in_planes).init(device),
            activation: Relu::new(),
            pool: MaxPool2dConfig::new([3, 3]).init(),
        }
    }
}

#[derive(Module, Debug)]
struct ResNetOutput<B: Backend> {
    pool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> ResNetOutput<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, channel, height, width] = x.dims();

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, channel * height * width]);
        self.fc.forward(x)
    }
}

#[derive(Config, Debug)]
struct ResNetOutputConfig {
    block_expansion: usize,
    num_classes: usize,
}

impl ResNetOutputConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ResNetOutput<B> {
        ResNetOutput {
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(512 * self.block_expansion, self.num_classes).init(device),
        }
    }
}
