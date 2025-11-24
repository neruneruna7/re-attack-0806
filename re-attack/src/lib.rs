// mod mnist;
pub mod attack;
pub mod bim;
pub mod cli;
pub mod data;
// pub mod fgsm;
// pub mod grad;
pub mod infer;
// pub mod ploof;
pub mod resnet18;
pub mod simple_mlp;
pub mod train;

pub static ARTIFACT_DIR: &str = "./tmp/burn-resnet18-mnist";

// 既存のbim.rsからClassificationModelを再エクスポート
pub use attack::ClassificationModel;
