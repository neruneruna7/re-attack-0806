use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    tensor::{Int, Tensor},
};

#[derive(Debug, Clone, Default)]
pub struct MnistBacher {}

#[derive(Debug, Clone)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBacher {
    fn batch(&self, items: Vec<MnistItem>, device: &<B as Backend>::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| {
                // example見たら，バックエンドのところがNdArrayだった.なぜ？
                Tensor::<B, 2>::from_data(data.convert::<B::FloatElem>(), device)
            })
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // 画素値を平均0，標準偏差1にしているらしい？ 学習を安定させるため？
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                    device,
                )
            })
            .collect();

        // テンソルを結合してバッチを作成
        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
