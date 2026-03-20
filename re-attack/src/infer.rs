use burn::{
    config::Config,
    data::{self, dataloader::batcher::Batcher, dataset::vision::MnistItem},
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, cast::ToElement, Int, Tensor, TensorData},
};

use crate::{
    data::{MnistBacher, MnistBatch},
    resnet18::ResNet18,
    train::MnistTrainingConfig,
};

pub fn infer<B: AutodiffBackend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let model = load::<B>(artifact_dir, &device);
    let label = item.label;
    let batcher = MnistBacher::default();
    let batch = batcher.batch(vec![item], &device);

    // let [batch_size, dims, height, width] = batch.images.dims();
    // // チャネル数1を加えて，4次元に変換
    // let image = batch
    //     .images
    //     .reshape([batch_size, 1, height, width])
    //     .detach();

    let output = model.forward(batch.images);

    let predicated = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("Predicted: {predicated}, Expected: {label}");
}

pub fn load<B: AutodiffBackend>(artifact_dir: &str, device: &B::Device) -> ResNet18<B> {
    // Load the model and perform inference
    // This is a placeholder for the actual inference logic
    let config = MnistTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("config.jsonがない");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .unwrap();

    let model = config.model.init::<B>(&device).load_record(record);
    model
}
