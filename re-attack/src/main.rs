// mod mnist;
mod data;
mod resnet18;
mod train;

use burn::backend::{
    Autodiff,
    wgpu::{Metal, WgpuDevice},
};

fn main() {
    println!("Hello, world!");
    let device = WgpuDevice::default();
    train::run::<Autodiff<Metal>>(device);
}
