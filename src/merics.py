import pandas as pd
import io


def calculate_metrics(df):
    """
    データフレームから論文準拠の指標を計算する関数
    """
    total_samples = len(df)
    
    # 1. 正解判定 (True: 正解, False: 不正解)
    # クリーン画像（攻撃前）の判定
    correct_clean = df['prediction_before_attack'] == df['target_label']
    # 攻撃画像の判定
    correct_adv = df['prediction_after_attack'] == df['target_label']

    # 2. Table 1 に対応する Accuracy (正解率) の計算
    # 全サンプルに対する正解率
    acc_clean = correct_clean.sum() / total_samples
    acc_adv = correct_adv.sum() / total_samples

    # 3. Destruction Rate (破壊率) の計算
    # 論文の式(1)に基づく: 「元々正解していた画像」のうち「攻撃後に不正解になった」割合
    # 分母: 元々正解だった数
    num_correct_clean = correct_clean.sum()
    
    if num_correct_clean > 0:
        # 分子: 元々正解 かつ 攻撃後に不正解
        # (~correct_adv は correct_adv の反転、つまり不正解)
        num_destroyed = (correct_clean & (~correct_adv)).sum()
        destruction_rate = num_destroyed / num_correct_clean
    else:
        destruction_rate = 0.0
        print("警告: 元々正解していた画像が0枚です。")

    # 4. 単純な Attack Success Rate (ASR)
    # 全サンプルに対する攻撃成功率（誤分類率）
    # Untargeted Attackの場合、ターゲットと一致しなければ成功
    asr_all = (~correct_adv).sum() / total_samples

    return {
        "Total Samples": total_samples,
        "Clean Accuracy (Top-1)": acc_clean,
        "Adversarial Accuracy (Top-1)": acc_adv, # これが論文 Table 1 の値
        "Destruction Rate": destruction_rate,    # これが論文 Eq(1) の値（本来の攻撃成功率）
        "Simple ASR (All samples)": asr_all
    }

def main():
    # CSVを読み込む（実運用では pd.read_csv("filename.csv") を使用）
    df = pd.read_csv("./attacked_data/imagenet/bim/eps_0.031/attack_results.csv")

    # 計算実行
    metrics = calculate_metrics(df)

    # 結果表示
    print("-" * 30)
    print("計算結果")
    print("-" * 30)
    print(f"総サンプル数: {metrics['Total Samples']}")
    print(f"Clean Images Top-1 Accuracy (攻撃前の正解率): {metrics['Clean Accuracy (Top-1)']:.2%} ({int(metrics['Clean Accuracy (Top-1)'] * metrics['Total Samples'])}/{metrics['Total Samples']})")
    print(f"Adv. Images Top-1 Accuracy  (攻撃後の正解率): {metrics['Adversarial Accuracy (Top-1)']:.2%} ({int(metrics['Adversarial Accuracy (Top-1)'] * metrics['Total Samples'])}/{metrics['Total Samples']})")
    print("-" * 30)
    print(f"Destruction Rate (破壊率/攻撃成功率): {metrics['Destruction Rate']:.2%}")
    print("※ 元々正解していた画像のうち、攻撃で誤分類された割合")
    print("-" * 30)


if __name__ == "__main__":
    main()