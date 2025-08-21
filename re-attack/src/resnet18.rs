use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    tensor::{BasicAutodiffOps, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use tracing::{info, instrument};

use crate::data::MnistBatch;

const NUM_CLASSES: usize = 10;

#[derive(Module, Debug)]
pub struct ResNet18<B: Backend> {
    // resnet_input: ResNetInput<B>,
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    activation: Relu,
    maxpool: MaxPool2d,

    layer1: ResNetLayer<B>,
    layer2: ResNetLayer<B>,
    layer3: ResNetLayer<B>,
    layer4: ResNetLayer<B>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
    // resnet_output: ResNetOutput<B>,
}

impl<B: Backend> ResNet18<B> {
    pub fn new(device: &B::Device) -> Self {
        ResNet18Config::new(NUM_CLASSES, 1, 64).init(device)
    }
    #[instrument(skip(self, x))]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = x.dims();
        // チャネル数1を加えて，4次元に変換
        let x = x.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.maxpool.forward(x);

        info!("before block layer, shape: {:?}", x.dims());
        println!("0");
        let x = self.layer1.forward(x);
        // todo!("debug end");
        info!("after block layer 1, shape: {:?}", x.dims());
        println!("1");
        let x = self.layer2.forward(x);
        info!("after block layer 2, shape: {:?}", x.dims());
        println!("2");
        let x = self.layer3.forward(x);
        info!("after block layer 3, shape: {:?}", x.dims());
        println!("3");
        let x = self.layer4.forward(x);
        info!("after block layer 4, shape: {:?}", x.dims());
        println!("4");

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
            conv1: Conv2dConfig::new([self.input_channel, 64], [7, 7])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(3, 3))
                .with_bias(false)
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            activation: Relu::new(),
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(),

            layer1: ResNetLayerConfig::new(64, 64, [1, 1]).init(device),
            layer2: ResNetLayerConfig::new(64, 128, [2, 2]).init(device),
            layer3: ResNetLayerConfig::new(128, 256, [2, 2]).init(device),
            layer4: ResNetLayerConfig::new(256, 512, [2, 2]).init(device),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(512 * self.block_expansion, self.num_classes).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct ResNetLayer<B: Backend> {
    blocks: [BasicBlock<B>; 2],
}

impl<B: Backend> ResNetLayer<B> {
    #[instrument()]
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.blocks[0].forward(x);
        let x = self.blocks[1].forward(x);
        x
    }
}

#[derive(Config, Debug)]
struct ResNetLayerConfig {
    in_planes: usize,
    out_planes: usize,
    stride: [usize; 2],
}

impl ResNetLayerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ResNetLayer<B> {
        let downsample = if self.stride != [1, 1] || self.in_planes != self.out_planes {
            Some(DownSampleConfig::new(
                self.in_planes,
                self.out_planes,
                self.stride,
            ))
        } else {
            None
        };
        ResNetLayer {
            blocks: [
                BasicBlockConfig::new(self.in_planes, self.out_planes)
                    .with_stride(self.stride)
                    .with_downsample(downsample)
                    .init(device),
                BasicBlockConfig::new(self.out_planes, self.out_planes).init(device),
            ],
        }
    }
}

#[derive(Module, Debug)]
struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    // 正規化レイヤ
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    shortcut: Option<DownSample<B>>,
    activation: Relu,
}

impl<B: Backend> BasicBlock<B> {
    #[instrument()]
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // デバッグ: 各経路の形状を出力して不整合箇所を特定する
        let identity = x.clone();
        let id_dims = identity.dims();
        // ショートカットを先に計算（現在の実装の流れ）
        let shortcut = if let Some(shortcut) = &self.shortcut {
            let s = shortcut.forward(identity.clone());
            info!(
                "BasicBlock shortcut shape: main_input={:?} -> shortcut={:?}",
                id_dims,
                s.dims()
            );
            s
        } else {
            info!("BasicBlock shortcut: None, input shape: {:?}", id_dims);
            identity.clone()
        };

        // メイン経路
        let mut out = self.conv1.forward(x);
        info!("After conv1 shape: {:?}", out.dims());
        out = self.bn1.forward(out);
        out = self.activation.forward(out);

        out = self.conv2.forward(out);
        info!("After conv2 shape: {:?}", out.dims());
        out = self.bn2.forward(out);

        // ここで形状が合わない場合は明示的にログを出して panic する
        if out.dims() != shortcut.dims() {
            panic!(
                "BasicBlock: shape mismatch between main and shortcut \n
                Shape mismatch before add: main={:?}, shortcut={:?}",
                out.dims(),
                shortcut.dims(),
            );
        }

        let out = out + shortcut;
        let out = self.activation.forward(out);
        out
    }
}

#[derive(Config, Debug)]
struct BasicBlockConfig {
    /// 入力チャネル数
    in_planes: usize,
    /// 出力チャネル数
    out_planes: usize,
    /// カーネルの移動距離
    #[config(default = "[1, 1]")]
    stride: [usize; 2],
    #[config(default = 1)]
    dilation: usize,
    #[config(default = "None")]
    downsample: Option<DownSampleConfig>,
}

impl BasicBlockConfig {
    /// 入力チャネル数，出力チャネル数，デバイス
    fn init<B: Backend>(&self, device: &B::Device) -> BasicBlock<B> {
        println!(
            "BasicBlockConfig.init: in={}, out={}, stride={:?}",
            self.in_planes, self.out_planes, self.stride
        );
        // let downsample = if self.stride != [1, 1] || self.in_planes != self.out_planes {
        //     Some(DownSampleConfig::new(self.in_planes, self.out_planes, self.stride).init(device))
        // } else {
        //     None
        // };
        BasicBlock {
            conv1: Conv2dConfig::new([self.in_planes, self.out_planes], [3, 3])
                .with_stride(self.stride)
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn1: BatchNormConfig::new(self.out_planes).init(device),
            conv2: Conv2dConfig::new([self.out_planes, self.out_planes], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn2: BatchNormConfig::new(self.out_planes).init(device),
            // Use the block's input/output channel sizes for the shortcut 1x1 conv
            shortcut: self.downsample.as_ref().map(|ds| ds.init(device)),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
struct DownSample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> DownSample<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        self.bn.forward(x)
    }
}

#[derive(Config, Debug)]
struct DownSampleConfig {
    in_planes: usize,
    out_planes: usize,
    stride: [usize; 2],
}

impl DownSampleConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DownSample<B> {
        DownSample {
            conv: Conv2dConfig::new([self.in_planes, self.out_planes], [1, 1])
                .with_stride(self.stride)
                .init(device),
            bn: BatchNormConfig::new(self.out_planes).init(device),
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
