use burn::{
    config,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
};
// #[derive(Module, Debug)]
// pub struct Mnist<B: Backend> {
//     conv1: Conv2d<B>,
//     conv2: Conv2d<B>,
//     conv3: Conv2d<B>,
//     fc1: Linear<B>,
//     fc2: Linear<B>,
//     dropout: Dropout,
//     activation: Relu,
// }

#[derive(Config, Debug)]
pub struct MnistConfig {
    #[config(default = 10)]
    num_classes: usize,
    hidden_size: usize,
    #[config(default = 0.25)]
    dropout: f64,
}

impl MnistConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> Mnist<B> {
        Mnist {
            //channels[ ]は，多分入力チャネル，出力チャネル
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv3: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            fc1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            fc2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
        }
    }
}
