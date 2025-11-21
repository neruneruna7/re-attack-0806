"""パス管理を一元化するクラス"""
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass
class PathManager:
    """出力パスを管理するクラス
    
    このクラスは、攻撃結果の出力先パスを一元的に管理します。
    単一責任原則に基づき、パス生成ロジックをこのクラスに集約しています。
    """
    
    # これは問題があるな
    config: 'Any'  # Configオブジェクト（型チェックを避けるためAnyを使用）
    
    def get_output_folder(self) -> str:
        """出力フォルダのパスを取得
        
        フォルダ構造: {output_dir}/{dataset}/{attack}/eps_{epsilon}
        
        Returns:
            出力フォルダの絶対パス
        """
        eps_str = f"{self.config.attack_params.epsilon:.3f}"
        
        output_folder = os.path.join(
            self.config.output_dir,
            self.config.dataset.value,
            self.config.attack.value,
            f"eps_{eps_str}"
        )
        
        return output_folder
    
    def get_csv_filepath(self) -> str:
        """CSV結果ファイルのパスを取得
        
        Returns:
            CSV結果ファイルの絶対パス
        """
        output_folder = self.get_output_folder()
        return os.path.join(output_folder, "attack_results.csv")
    
    def ensure_output_folder_exists(self) -> str:
        """出力フォルダが存在することを保証
        
        フォルダが存在しない場合は作成します。
        
        Returns:
            作成された（または既に存在する）出力フォルダのパス
        """
        output_folder = self.get_output_folder()
        os.makedirs(output_folder, exist_ok=True)
        return output_folder
