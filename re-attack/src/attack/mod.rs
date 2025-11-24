/// 敵対的攻撃モジュール
/// 
/// このモジュールは、各種敵対的攻撃手法の共通インターフェースと実装を提供します。

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

pub mod bim;
pub mod fgsm;

/// 敵対的攻撃パラメータの共通構造体
#[derive(Debug, Clone)]
pub struct AttackParams {
    /// epsilon: 許容される摂動の最大値
    pub epsilon: f32,
    /// alpha: 各イテレーションでの更新ステップサイズ
    pub alpha: f32,
    /// num_iter: イテレーション回数
    pub num_iter: usize,
    /// min_val: 画像ピクセル値の最小値（正規化前）
    pub min_val: f32,
    /// max_val: 画像ピクセル値の最大値（正規化前）
    pub max_val: f32,
}

impl Default for AttackParams {
    fn default() -> Self {
        Self {
            epsilon: 0.3,
            alpha: 0.01,
            num_iter: 10,
            min_val: 0.0,
            max_val: 1.0,
        }
    }
}

/// 分類モデルのための共通トレイト
pub trait ClassificationModel<B: Backend>: Module<B> {
    /// フォワード推論を実行
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2>;
}

/// 敵対的攻撃手法の共通トレイト
pub trait AdversarialAttack {
    /// 攻撃を実行し、敵対的サンプルを生成
    fn generate<B, M>(
        &self,
        input: Tensor<B, 4>,
        target: Tensor<B, 1, Int>,
        model: &M,
        params: &AttackParams,
        mean: Tensor<B, 4>,
        std: Tensor<B, 4>,
    ) -> Tensor<B, 4>
    where
        B: AutodiffBackend,
        M: ClassificationModel<B>;

    /// 攻撃手法の名前を返す
    fn name(&self) -> &str;
}
