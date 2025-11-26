import toml
import matplotlib.pyplot as plt
import os

def load_data(toml_path):
    """
    TOMLファイルを読み込みます。
    """
    with open(toml_path, 'r', encoding='utf-8') as f:
        return toml.load(f)

def setup_japanese_font():
    """
    matplotlibで日本語を表示するためのフォント設定を行います。
    OSに合わせて利用可能なフォントを探索します。
    """
    import matplotlib.font_manager as fm
    
    # 優先する日本語フォントのリスト
    font_candidates = [
        'Hiragino Sans', 'Hiragino Kaku Gothic ProN', # Mac
        'Meiryo', 'Yu Gothic', 'MS Gothic',           # Windows
        'Noto Sans CJK JP', 'TakaoGothic', 'IPAGothic' # Linux
    ]
    
    system_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in system_fonts:
            plt.rcParams['font.family'] = font
            print(f"フォントを設定しました: {font}")
            return
    
    print("警告: 日本語フォントが見つかりませんでした。豆腐（文字化け）が発生する可能性があります。")

def extract_baseline_data(data):
    """
    再攻撃のみ（no_preprocess）のデータを抽出します。
    戻り値: (epsilonリスト, 防御成功率リスト)
    """
    entries = data['experiments']['preprocess_and_reattack']['bim_no_preprocess_bim']
    
    # epsilonでソート
    entries.sort(key=lambda x: x['re_attack']['epsilon'])
    
    epsilons = [e['re_attack']['epsilon'] for e in entries]
    rates = [e['results']['defense_success_rate_with_reattack'] for e in entries]
    
    return epsilons, rates

def extract_method_data(data, method_key):
    """
    特定の前処理手法のデータを抽出・計算します。
    戻り値: (epsilonリスト, 防御成功率リスト, 全体正解率リスト)
    """
    entries = data['experiments']['preprocess_and_reattack'][method_key]
    
    # epsilonでソート
    entries.sort(key=lambda x: x['re_attack']['epsilon'])
    
    epsilons = []
    defense_rates = []
    global_rates = []
    
    for entry in entries:
        res = entry['results']
        eps = entry['re_attack']['epsilon']
        
        # 1. 前処理と再攻撃の両方を行ったときの防御成功率
        # (防御成功数 / 初回攻撃成功数) ※初回攻撃に成功したサンプルの中での防御率
        if res['defense_total_count'] > 0:
            def_rate = res['defense_reattack_success_count'] / res['defense_total_count']
        else:
            def_rate = 0.0
            
        # 2. すべてのサンプルに対して計算したときの前処理と再攻撃の両方を行ったときの防御成功率
        # (最終的な正解数 / 全サンプル数) ※モデルとしての総合的な頑健性
        if res['total_samples'] > 0:
            global_rate = res['final_correct_count'] / res['total_samples']
        else:
            global_rate = 0.0
            
        epsilons.append(eps)
        defense_rates.append(def_rate)
        global_rates.append(global_rate)
        
    return epsilons, defense_rates, global_rates

def plot_graph(baseline_eps, baseline_rates, 
               method_eps, method_def_rates, method_global_rates, 
               method_name_jp, output_filename):
    """
    3つの指標をプロットして画像を保存します。
    """
    plt.figure(figsize=(10, 6))
    
    # ユニバーサルデザイン(UD)に配慮したカラーパレット
    # 黒/グレー: ベースライン, オレンジ: 防御成功率, 青: 全体正解率
    color_baseline = '#595959' # ダークグレー
    color_defense = '#E69F00'  # オレンジ
    color_global = '#0072B2'   # ブルー
    
    # 1. 再攻撃だけをしたときの防御成功率 (Baseline)
    plt.plot(baseline_eps, baseline_rates, marker='x', linestyle='--', 
             color=color_baseline, label='再攻撃のみ（前処理なし）', linewidth=2)
    
    # 2. 前処理と再攻撃の両方を行ったときの防御成功率
    plt.plot(method_eps, method_def_rates, marker='o', linestyle='-', 
             color=color_defense, label='前処理＋再攻撃（防御成功率）', linewidth=2)
    
    # 3. すべてのサンプルに対して計算したときの前処理と再攻撃の両方を行ったときの防御成功率
    plt.plot(method_eps, method_global_rates, marker='s', linestyle='-', 
             color=color_global, label='前処理＋再攻撃（全体正解率）', linewidth=2)
    
    # グラフの装飾
    plt.title(f'前処理手法: {method_name_jp} における防御性能', fontsize=16)
    plt.xlabel('Epsilon (摂動の大きさ)', fontsize=14)
    plt.ylabel('確率 (Probability)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.ylim(0, 1.05) # 確率は0-1の範囲
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"グラフを保存しました: {output_filename}")
    plt.close()

def main():
    toml_file = "winf2025.toml" # 読み込むTOMLファイル名
    
    if not os.path.exists(toml_file):
        print(f"エラー: {toml_file} が見つかりません。")
        return

    # 日本語フォント設定
    setup_japanese_font()
    
    # データ読み込み
    data = load_data(toml_file)
    
    # ベースラインデータの取得
    base_eps, base_rates = extract_baseline_data(data)
    
    # 各手法の設定 (TOMLのキー, 日本語表示名, 保存ファイル名)
    methods = [
        ('bim_gauss_bim', 'ガウシアンブラー', 'graph_gaussian.png'),
        ('bim_pixel_reduction_bim', '輝度調整', 'graph_pixel_reduction.png'),
        ('bim_laplacian_bim', 'ラプラシアン先鋭化', 'graph_laplacian.png')
    ]
    
    for key, jp_name, fname in methods:
        if key in data['experiments']['preprocess_and_reattack']:
            # 手法データの取得・計算
            m_eps, m_def_rates, m_glob_rates = extract_method_data(data, key)
            
            # プロット実行
            plot_graph(base_eps, base_rates, 
                       m_eps, m_def_rates, m_glob_rates, 
                       jp_name, fname)
        else:
            print(f"警告: データ内にキー '{key}' が見つかりませんでした。")

if __name__ == "__main__":
    main()