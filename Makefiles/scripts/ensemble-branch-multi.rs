//! ```cargo
//! [dependencies]
//! ```
//!
//! 分岐型アンサンブル再攻撃 (Branched Ensemble Re-Attack) のパラメータスイープ実行スクリプト
//!
use std::process::{Command, ExitStatus};

fn main() {
    // --- 固定パラメータ ---
    let dataset = "cifar10";
    let model = "resnet20";
    let batch_size = "128";

    // Attack (BIM) 固定パラメータ
    let attack_alpha = "1/255";
    let attack_iter = "10";

    // Re-Attack (BIM) 固定パラメータ
    let reattack_alpha = "1/255";
    let reattack_iter = "10";

    // --- 変動パラメータ (Epsilon) ---
    // ここを変更して実験条件を調整してください

    // 例: 攻撃強度(eps)を変化させる
    // let attack_eps_list = vec!["2/255", "4/255", "8/255", "16/255"];
    let attack_eps_list = vec!["4/255"];

    // 例: Re-Attack強度(eps)を変化させる
    // 攻撃強度に対して、どの程度の強度で再攻撃(復元)を試みるかを探索します
    let reattack_eps_list = vec!["2/255", "4/255", "8/255"];

    // --- 実験ループ ---
    for attack_eps in &attack_eps_list {
        for reattack_eps in &reattack_eps_list {
            println!("----------------------------------------------------------------");
            println!(
                "Running Branched Ensemble Re-Attack: Attack Eps={}, Re-Attack Eps={}",
                attack_eps, reattack_eps
            );
            println!("----------------------------------------------------------------");

            let status = run_python_script(
                dataset,
                model,
                attack_eps,
                attack_alpha,
                attack_iter,
                reattack_eps,
                reattack_alpha,
                reattack_iter,
                batch_size,
            );

            if !status.success() {
                eprintln!(
                    "Error: Command failed with Attack Eps={}, Re-Attack Eps={}",
                    attack_eps, reattack_eps
                );
                // エラー発生時の挙動制御 (必要であれば exit(1) など)
            }
        }
    }
}

fn run_python_script(
    dataset: &str,
    model: &str,
    attack_eps: &str,
    attack_alpha: &str,
    attack_iter: &str,
    reattack_eps: &str,
    reattack_alpha: &str,
    reattack_iter: &str,
    batch_size: &str,
) -> ExitStatus {
    let mut cmd = Command::new("uv");

    cmd.args(&[
        "run",
        // ここを実行したいPythonファイル名に変更
        "src/ensemble_branch_reattack.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--attack-eps",
        attack_eps,
        "--attack-alpha",
        attack_alpha,
        "--attack-n",
        attack_iter,
        "--reattack-eps",
        reattack_eps,
        "--reattack-alpha",
        reattack_alpha,
        "--reattack-n",
        reattack_iter,
        "--batch-size",
        batch_size,
        "--no-save-images", // 大量実験時は保存しない設定を推奨
                            // 必要に応じてサンプル数を制限
                            // "--num-samples", "100",
    ]);

    // コマンドを表示（デバッグ用）
    println!("Executing: {:?}", cmd);

    cmd.status().expect("Failed to execute process")
}
