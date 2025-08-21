// mod mnist;
mod data;
mod resnet18;
mod train;

use burn::backend::{
    Autodiff,
    wgpu::{Metal, WgpuDevice},
};

fn main() {
    // tracing_subscriber::fmt().init();

    let device = WgpuDevice::default();
    train::run::<Autodiff<Metal>>(device);
}
