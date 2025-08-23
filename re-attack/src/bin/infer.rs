use burn::{
    backend::{
        Autodiff,
        wgpu::{Metal, WgpuDevice},
    },
    data::{dataloader::batcher, dataset::Dataset as _},
};
use re_attack::{data::MnistBacher, infer::infer, train};

use re_attack::ARTIFACT_DIR;

fn main() {
    // tracing_subscriber::fmt().init();
    // pwdを表示する

    let device = WgpuDevice::default();

    // let batcher = MnistBacher::default();
    // let batch = batcher.batch(
    //     vec![
    //         burn::data::dataset::vision::MnistDataset::test()
    //             .get(42)
    //             .unwrap(),
    //     ],
    //     &device,
    // );
    let item = burn::data::dataset::vision::MnistDataset::test()
        .get(42)
        .unwrap();

    infer::<Autodiff<Metal>>(ARTIFACT_DIR, device, item);
}
