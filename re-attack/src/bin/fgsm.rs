use burn::{
    backend::{
        Autodiff,
        wgpu::{Metal, WgpuDevice},
    },
    data::dataset::Dataset as _,
};
use re_attack::{fgsm, infer, train};

use re_attack::ARTIFACT_DIR;
use tracing::info;

fn main() {
    tracing_subscriber::fmt().init();
    // pwdを表示する

    let device = WgpuDevice::default();
    // train::run::<Autodiff<Metal>>(device);
    let items = burn::data::dataset::vision::MnistDataset::test()
        .iter()
        .take(50)
        .map(|x| x.clone())
        .collect::<Vec<_>>();
    let epsilons = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
    let mut results = Vec::new();
    for &epsilon in &epsilons {
        info!("Epsilon: {epsilon}");
        let (correct, total) =
            fgsm::fgsm::<Autodiff<Metal>>(ARTIFACT_DIR, device.clone(), epsilon, &items);
        results.push((epsilon, correct, total));
    }

    for (epsilon, correct, num) in results {
        let final_acc = correct as f64 / num as f64;
        println!("Epsilon: {epsilon} Test Accuracy:  {correct}/{num} = {final_acc}");
    }

    // fgsm::fgsm::<Autodiff<Metal>>(ARTIFACT_DIR, device, &items);
}
