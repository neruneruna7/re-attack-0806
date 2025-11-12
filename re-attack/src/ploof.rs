use burn::{
    config::Config,
    nn::{
        Linear, LinearConfig,
        conv::Conv2d,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Ploof<B: Backend> {
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

#[derive(Config, Debug)]
pub struct PloofConfig {
    #[config(default = 10)]
    num_classes: usize,
}

impl PloofConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> Ploof<B> {
        Ploof {
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(device),
            fc: LinearConfig::new(64, self.num_classes).init(device),
        }
    }
}
