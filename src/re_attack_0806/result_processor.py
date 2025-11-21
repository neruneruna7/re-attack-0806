"""結果処理を担当するクラス"""
import csv
import os
from dataclasses import dataclass, asdict
from typing import Optional, TextIO
import concurrent.futures
from torch.types import Number

from re_attack_0806.utils.normTensor import TensorWithState
from re_attack_0806.utils.config import AttackParams
from re_attack_0806 import utils


@dataclass
class AttackResult:
    """攻撃結果を格納するデータクラス"""
    index: int
    target_label: Number
    prediction_before_attack: Number
    prediction_after_attack: Number
    l2_perturbation: Number
    image_filepath: str = ""


@dataclass
class AttackSummary:
    """攻撃結果のサマリーを格納するデータクラス"""
    total_samples: int
    clean_correct_count: int
    attack_success_count: int
    attack_success_rate: float
    mean_perturbation: float


class ResultProcessor:
    """攻撃結果を処理するクラス
    
    このクラスは、攻撃結果のCSV保存、画像保存、統計情報の計算を担当します。
    単一責任原則に基づき、結果処理に関する責務をまとめています。
    """
    
    def __init__(
        self,
        csv_filepath: str,
        output_folder: str,
        save_images: bool = False,
    ):
        """ResultProcessorの初期化
        
        Args:
            csv_filepath: CSV結果ファイルのパス
            output_folder: 画像出力フォルダのパス
            save_images: 画像を保存するかどうか
        """
        self.csv_filepath = csv_filepath
        self.output_folder = output_folder
        self.save_images = save_images
        self.csv_file: Optional[TextIO] = None
        self.csv_writer: Optional[csv.writer] = None
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        
        # 統計情報の初期化
        self.total_samples = 0
        self.clean_correct_count = 0  # クリーン画像を正しく分類したサンプル数
        self.attack_success_count = 0
        self.total_perturbation = 0.0
    
    def __enter__(self):
        """コンテキストマネージャーの開始
        
        CSVファイルとThreadPoolExecutorを初期化します。
        """
        # CSVファイルを開く
        self.csv_file = open(self.csv_filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # ヘッダーを書き込む
        self.csv_writer.writerow([
            "index",
            "target_label",
            "prediction_before_attack",
            "prediction_after_attack",
            "l2_perturbation",
            "image_filepath"
        ])
        
        # ThreadPoolExecutorを初期化
        self.executor = concurrent.futures.ThreadPoolExecutor()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了
        
        CSVファイルとThreadPoolExecutorをクローズします。
        """
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        
        if self.csv_file is not None:
            self.csv_file.close()
    
    def process_result(
        self,
        result: AttackResult,
        perturbed_image: TensorWithState
    ):
        """1つの攻撃結果を処理
        
        Args:
            result: 攻撃結果
            perturbed_image: 摂動を加えた画像テンソル
        """
        # 統計情報の更新
        self.total_samples += 1
        
        if result.prediction_before_attack == result.target_label:
            self.clean_correct_count += 1
            self.total_perturbation += result.l2_perturbation
            
            if result.prediction_after_attack != result.prediction_before_attack:
                self.attack_success_count += 1
        
        # 画像の保存（非同期）
        image_filepath = ""
        if self.save_images:
            image_filepath = os.path.join(
                self.output_folder,
                f"idx_{result.index}_label_{result.prediction_after_attack}.png"
            )
            if self.executor is not None:
                self.executor.submit(
                    utils.save_tensor_as_image,
                    perturbed_image.tensor,
                    image_filepath
                )
        
        # CSVへの書き込み
        if self.csv_writer is not None:
            self.csv_writer.writerow([
                result.index,
                result.target_label,
                result.prediction_before_attack,
                result.prediction_after_attack,
                result.l2_perturbation,
                image_filepath
            ])
    
    def get_summary(self) -> AttackSummary:
        """攻撃結果のサマリーを取得
        
        Returns:
            攻撃結果のサマリー
        """
        attack_success_rate = (
            self.attack_success_count / self.clean_correct_count
            if self.clean_correct_count > 0
            else 0.0
        )
        
        mean_perturbation = (
            self.total_perturbation / self.clean_correct_count
            if self.clean_correct_count > 0
            else 0.0
        )
        
        return AttackSummary(
            total_samples=self.total_samples,
            clean_correct_count=self.clean_correct_count,
            attack_success_count=self.attack_success_count,
            attack_success_rate=attack_success_rate,
            mean_perturbation=mean_perturbation
        )
    
    @staticmethod
    def print_summary(
        summary: AttackSummary,
        dataset: str,
        model: str,
        attack: str,
        attack_params: AttackParams,
        csv_filepath: str
    ):
        """攻撃結果のサマリーを出力
        
        Args:
            summary: 攻撃結果のサマリー
            dataset: データセット名
            model: モデル名
            attack: 攻撃名
            attack_params: 攻撃パラメータ
            csv_filepath: CSV結果ファイルのパス
        """
        # 攻撃パラメータを整形
        params_dict = asdict(attack_params)
        params_dict.pop('batch_size', None)
        attack_params_formatted = ", ".join([
            f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in params_dict.items()
        ])
        
        print("\n=== Attack Summary ===")
        print("[実験設定]")
        print(f"  データセット: {dataset}")
        print(f"  モデル: {model}")
        print(f"  攻撃: {attack} ({attack_params_formatted})")
        
        print("\n[結果]")
        print(f"  処理サンプル総数: {summary.total_samples}")
        print(f"  クリーン画像を正しく分類したサンプル数: {summary.clean_correct_count}")
        print(f"  攻撃成功率: {summary.attack_success_rate:.4f} = {summary.attack_success_count} / {summary.clean_correct_count}")
        print("  攻撃成功率には、元の画像を正しく分類したサンプルのみを考慮している。")
        print(f"  平均摂動量(L2ノルム): {summary.mean_perturbation:.4f}")
        print(f"結果は {csv_filepath} に保存されました。")
        print("======================")
