use burn::{
    backend::{
        wgpu::{Metal, WgpuDevice},
        Autodiff,
    },
    optim::{decay::WeightDecayConfig, AdamConfig},
};
use re_attack::train;
use re_attack::{resnet18::ResNet18Config, train::MnistTrainingConfig, ARTIFACT_DIR};
use resnet_burn::ResNet;

fn main() {
    // tracing_subscriber::fmt().init();

    let device = WgpuDevice::default();
    let model = ResNet::resnet18(10, &device);
    let config = MnistTrainingConfig::new(
        ResNet18Config::new(10, 1, 64),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );
    train::run::<Autodiff<Metal>>(ARTIFACT_DIR, config, device);
}
