//! ```cargo
//! [dependencies]
//! ```
//!
//! アンサンブル推論のパラメータスイープ実行スクリプト
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

    // 例: Re-Attack強度(eps)を変化させる (固定したい場合は1つだけ記述)
    let reattack_eps_list = vec!["2/255", "4/255", "8/255", "16/255"];

    // --- 実験ループ ---
    for attack_eps in &attack_eps_list {
        for reattack_eps in &reattack_eps_list {
            println!("----------------------------------------------------------------");
            println!(
                "Running Ensemble Re-Attack: Attack Eps={}, Re-Attack Eps={}",
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
                // エラーが出ても続行する場合はここをコメントアウト、止める場合は exit
                // std::process::exit(1);
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
        "src/ensemble_reattack.py",
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
        "--no-save-images",
        // 必要に応じて追加
        // "--num-samples", "100",
    ]);

    // コマンドを表示（デバッグ用）
    println!("Executing: {:?}", cmd);

    cmd.status().expect("Failed to execute process")
}
