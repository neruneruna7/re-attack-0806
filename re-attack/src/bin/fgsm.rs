use burn::{
    backend::{
        Autodiff,
        wgpu::{Metal, WgpuDevice},
    },
    data::dataset::Dataset as _,
};
use re_attack::{fgsm, infer, train};

use re_attack::ARTIFACT_DIR;

fn main() {
    // tracing_subscriber::fmt().init();
    // pwdを表示する

    let device = WgpuDevice::default();
    // train::run::<Autodiff<Metal>>(device);
    fgsm::fgsm::<Autodiff<Metal>>(
        ARTIFACT_DIR,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
