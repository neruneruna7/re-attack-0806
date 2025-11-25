import toml
import pandas as pd
import matplotlib.pyplot as plt
import os

# TOMLファイルのパス
TOML_FILE_PATH = 'winf2025.toml'

def load_data(file_path):
    """TOMLを読み込み、データをDataFrame化する"""
    if not os.path.exists(file_path):
        print(f"エラー: '{file_path}' が見つかりません。")
        return pd.DataFrame()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
    except Exception as e:
        print(f"TOML読み込みエラー: {e}")
        return pd.DataFrame()

    rows = []
    target_category = 'preprocess_and_reattack'
    
    if 'experiments' not in data or target_category not in data['experiments']:
        return pd.DataFrame()

    groups = data['experiments'][target_category]

    for group_name, exp_list in groups.items():
        for exp in exp_list:
            results = exp.get('results', {})
            re_attack = exp.get('re_attack', {})
            preprocess = exp.get('preprocess', {})
            
            # --- 各種メトリクスの計算 ---
            total_samples = results.get('total_samples', 10000)
            defense_total = results.get('defense_total_count', 5327)
            
            # 1. 前処理のみでの防御成功率
            pre_only_count = results.get('defense_preprocess_success_count', 0)
            rate_pre_only = pre_only_count / defense_total if defense_total > 0 else 0

            # 2. 前処理+再攻撃での防御成功率
            reattack_count = results.get('defense_reattack_success_count', 0)
            rate_reattack = reattack_count / defense_total if defense_total > 0 else 0

            # 3. 最終的な正解率
            final_count = results.get('final_correct_count', 0)
            rate_final = final_count / total_samples if total_samples > 0 else 0

            row = {
                'group': group_name,
                'method_name': preprocess.get('method', group_name),
                'epsilon': float(re_attack.get('epsilon', 0)),
                'rate_pre_only': rate_pre_only,
                'rate_reattack': rate_reattack,
                'rate_final': rate_final
            }
            rows.append(row)

    return pd.DataFrame(rows)

def plot_3_horizontal(df):
    if df.empty:
        print("プロットするデータがありません。")
        return

    # プロットするグループの定義（左から順に）
    target_groups = [
        'bim_gauss_bim', 
        'bim_pixel_reduction_bim', 
        'bim_my1_bim'
    ]
    
    # グラフ設定 (1行3列)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plt.style.use('ggplot')

    # 各グラフのテーマカラー（手法ごとのイメージカラー）
    group_colors = {
        'bim_gauss_bim': 'tab:blue',
        'bim_pixel_reduction_bim': 'tab:orange',
        'bim_my1_bim': 'tab:green'
    }

    # 指標ごとのスタイル定義
    metrics_style = {
        'rate_reattack': {'ls': '-',  'marker': 'o', 'label': 'Defense (Re-Attack)'}, # 実線
        'rate_final':    {'ls': '--', 'marker': 's', 'label': 'Final Accuracy'},      # 破線
        'rate_pre_only': {'ls': ':',  'marker': 'x', 'label': 'Defense (Pre-Only)'}   # 点線
    }

    # ループ処理で3つのグラフを描画
    for ax, group in zip(axes, target_groups):
        # 該当グループのデータを抽出
        group_df = df[df['group'] == group].sort_values('epsilon')
        
        if group_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            continue

        method_name = group_df['method_name'].iloc[0]
        base_color = group_colors.get(group, 'gray')

        # 3つの指標をプロット
        for metric_col, style in metrics_style.items():
            ax.plot(
                group_df['epsilon'], 
                group_df[metric_col], 
                color=base_color,      # 手法ごとのテーマカラーを使用
                linestyle=style['ls'], # 指標ごとの線種
                marker=style['marker'],
                linewidth=2,
                label=style['label']
            )

        # グラフごとの装飾
        ax.set_title(method_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Re-Attack Epsilon")
        ax.grid(True)
        ax.set_ylim(-0.05, 1.05)
        
        # 凡例を表示
        ax.legend(fontsize='small')

    # 一番左のグラフだけにY軸ラベルをつける
    axes[0].set_ylabel("Rate (Success / Accuracy)", fontsize=12)

    plt.tight_layout()
    save_path = 'experiment_results_horizontal.png'
    plt.savefig(save_path)
    print(f"グラフを '{save_path}' に保存しました。")

if __name__ == "__main__":
    df = load_data(TOML_FILE_PATH)
    plot_3_horizontal(df)