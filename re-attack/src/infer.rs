use burn::{
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
};

use crate::{data::MnistBacher, train::MnistTrainingConfig};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    // Load the model and perform inference
    // This is a placeholder for the actual inference logic
    let config = MnistTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("config.jsonがない");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .unwrap();

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBacher::default();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicated = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("Predicted: {predicated}, Expected: {label}");
}
