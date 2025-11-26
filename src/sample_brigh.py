import torch

def adjust_brightness(images: torch.Tensor) -> torch.Tensor:
    """
    4次元テンソルの輝度値を-0.1し、0.0-1.0の範囲に収める関数
    
    Args:
        images: 形状が (B, C, H, W) の4次元テンソル。値は0.0-1.0を想定。
    Returns:
        輝度調整後のテンソル
    """
    # 1. ブロードキャスト機能により、全要素から0.1が減算される
    darker_images = images - 0.1
    
    # 2. 値が0.0未満や1.0超過にならないよう範囲を制限（クランプ）する
    #    負の値が発生すると画像表示時や後続の計算で不具合の原因となるため推奨されます
    clamped_images = torch.clamp(darker_images, min=0.0, max=1.0)
    
    return clamped_images

# --- 使用例 ---
# ダミーデータ: バッチ4, 3チャンネル, 256x256
batch_tensor = torch.rand(4, 3, 256, 256)

# 処理の実行
result = adjust_brightness(batch_tensor)

print(f"処理前の最大値: {batch_tensor.max():.4f}, 最小値: {batch_tensor.min():.4f}")
print(f"処理後の最大値: {result.max():.4f}, 最小値: {result.min():.4f}")