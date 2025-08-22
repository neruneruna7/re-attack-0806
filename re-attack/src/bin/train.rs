use burn::{
    backend::{
        Autodiff,
        wgpu::{Metal, WgpuDevice},
    },
    optim::{AdamConfig, decay::WeightDecayConfig},
};
use re_attack::train;
use re_attack::{ARTIFACT_DIR, resnet18::ResNet18Config, train::MnistTrainingConfig};

fn main() {
    // tracing_subscriber::fmt().init();

    let device = WgpuDevice::default();
    let config = MnistTrainingConfig::new(
        ResNet18Config::new(10, 1, 64),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );
    train::run::<Autodiff<Metal>>(ARTIFACT_DIR, config, device);
}
