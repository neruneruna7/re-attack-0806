use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Tensor;

type Backend = Wgpu;
type AutoDIffBackend = Autodiff<Backend>;

fn main() {
    let device = Default::default();

    let x = Tensor::<AutoDIffBackend, 1>::from_floats([5.0], &device).require_grad();
    let y = x.clone().powf_scalar(2.0) * 3.0;

    // 勾配を求めて表示する
    let gradients = y.backward();
    let tensor_grad = x.grad(&gradients).unwrap();

    println!("y = {}", y);
    println!("dy/dx = {:?}", tensor_grad.into_scalar());
}
