#!/usr/bin/env python3
"""
general_reattack.py の動作確認用テストスクリプト
実際のデータをダウンロードせずに、合成データで動作確認を行う
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

# テスト用の簡単なモデル
class SimpleModel(nn.Module):
    """テスト用の簡単なCNNモデル"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.model_name = "test_model"

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_imports():
    """モジュールのインポートをテスト"""
    print("Testing imports...")
    try:
        import sys
        sys.path.insert(0, '/home/runner/work/re-attack-0806/re-attack-0806/src')
        from general_reattack import (
            Config, DatasetKind, ModelKind, AttackKind, ReAttackKind,
            apply_preset, PresetKind
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Config クラスのテスト"""
    print("\nTesting Config class...")
    try:
        import sys
        sys.path.insert(0, '/home/runner/work/re-attack-0806/re-attack-0806/src')
        from general_reattack import Config, DatasetKind, ModelKind, AttackKind, ReAttackKind
        
        cfg = Config(
            dataset=DatasetKind.MNIST,
            model=ModelKind.MORIMOTO_MNIST,
            attack=AttackKind.BIM,
            reattack=ReAttackKind.BIM,
            epsilon=0.3,
            reattack_epsilon=0.2
        )
        
        assert cfg.dataset == DatasetKind.MNIST
        assert cfg.epsilon == 0.3
        assert cfg.reattack_epsilon == 0.2
        print("✓ Config class works correctly")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_preset():
    """プリセット機能のテスト"""
    print("\nTesting preset functionality...")
    try:
        import sys
        sys.path.insert(0, '/home/runner/work/re-attack-0806/re-attack-0806/src')
        from general_reattack import Config, apply_preset, PresetKind
        
        cfg = Config()
        cfg_preset = apply_preset(cfg, PresetKind.MORIMOTO_MNIST_BIM_BIM)
        
        assert cfg_preset.epsilon == 0.3
        assert cfg_preset.alpha == 0.05
        print("✓ Preset application works correctly")
        return True
    except Exception as e:
        print(f"✗ Preset test failed: {e}")
        return False


def test_attack_logic():
    """攻撃と再攻撃のロジックをテスト"""
    print("\nTesting attack logic with synthetic data...")
    try:
        import sys
        sys.path.insert(0, '/home/runner/work/re-attack-0806/re-attack-0806/src')
        sys.path.insert(0, '/home/runner/work/re-attack-0806/re-attack-0806/src/lib')
        
        from lib.attacks import fgsm, bim
        
        # デバイス設定
        device = torch.device("cpu")
        
        # 簡単なモデルを作成
        model = SimpleModel().to(device)
        model.eval()
        
        # 合成データを作成 (1x1x28x28)
        data = torch.rand(1, 1, 28, 28, device=device) * 0.5 + 0.25  # 0.25-0.75の範囲
        target = torch.tensor([5], device=device)
        
        print(f"  Original data shape: {data.shape}")
        print(f"  Target label: {target.item()}")
        
        # 初回攻撃（FGSM）
        attacked = fgsm.fgsm_attack(data, epsilon=0.1, target=target, model=model, device=device)
        print(f"  Attacked data shape: {attacked.shape}")
        
        # 攻撃後の予測
        with torch.no_grad():
            # データを正規化（簡易版）
            mean_t = torch.tensor([0.1307], device=device).view(1, -1, 1, 1)
            std_t = torch.tensor([0.3081], device=device).view(1, -1, 1, 1)
            attacked_norm = (attacked - mean_t) / std_t
            pred_attacked = model(attacked_norm).argmax(1)
        print(f"  Predicted label after attack: {pred_attacked.item()}")
        
        # 再攻撃（BIM）
        reattacked = bim.bim_attack(
            attacked, epsilon=0.1, alpha=0.01, num_iter=5,
            target=pred_attacked, model=model, device=device
        )
        print(f"  Reattacked data shape: {reattacked.shape}")
        
        # 再攻撃後の予測
        with torch.no_grad():
            reattacked_norm = (reattacked - mean_t) / std_t
            pred_reattacked = model(reattacked_norm).argmax(1)
        print(f"  Predicted label after reattack: {pred_reattacked.item()}")
        
        print("✓ Attack logic test passed")
        return True
    except Exception as e:
        print(f"✗ Attack logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_command_line_parsing():
    """コマンドライン引数のパースをテスト"""
    print("\nTesting command-line argument parsing...")
    try:
        import sys
        import subprocess
        
        result = subprocess.run(
            ["uv", "run", "python", "./src/general_reattack.py", "--help"],
            cwd="~/workspace/univ/lab/re-attack-0806/re-attack-0806",
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "general re-attack runner" in result.stdout:
            print("✓ Command-line parsing works correctly")
            return True
        else:
            print(f"✗ Command-line parsing failed")
            return False
    except Exception as e:
        print(f"✗ Command-line test failed: {e}")
        return False


def main():
    """メインテスト関数"""
    print("=" * 60)
    print("general_reattack.py テストスイート")
    print("=" * 60)
    
    results = []
    
    # 各テストを実行
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Preset", test_preset()))
    results.append(("Attack Logic", test_attack_logic()))
    results.append(("CLI Parsing", test_command_line_parsing()))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<20} : {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n合計: {passed}/{total} テストが成功")
    
    if passed == total:
        print("\n✓ すべてのテストが成功しました！")
        return 0
    else:
        print(f"\n✗ {total - passed} 個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())
