//! ```cargo
//! [dependencies]
//! ```

use std::fs::OpenOptions;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> std::io::Result<()> {
    // 1. 設定: 変化させるパラメータと出力ファイル名
    let eps_values = vec!["2/255", "4/255", "8/255", "16/255"];
    let output_path = "experiment_log.txt";

    println!("実験を開始します。結果は {} に保存されます。", output_path);

    // 2. ログファイルの初期化（書き込みモードで開き、内容を空にする）
    // ループ内で追記するために、ファイルハンドルを保持します
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;

    // 開始時刻などのヘッダーを書き込み
    writeln!(file, "=== Experiment Start ===")?;

    // 3. ループ処理
    for eps in eps_values {
        println!("実行中: eps = {} ...", eps);

        // ログファイルに見出し（区切り線）を書き込み
        writeln!(file, "
------------------------------------------------------------")?;
        writeln!(file, "Parameter: reattack-eps = {}", eps)?;
        writeln!(file, "------------------------------------------------------------
")?;

        // 4. コマンドの構築と実行
        // uv run ... の引数を設定
        let status = Command::new("uv")
            .args(&[
                "run", "src/general_preprocess_reattack.py",
                "--dataset", "cifar10",
                "--model", "morimoto-cifar10",
                "--attack-kind", "bim",
                "--attack-eps", "8/255",
                "--attack-alpha", "2/255",
                "--attack-n", "10",
                "--preprocess-kind", "gaussian_blur",
                "--preprocess-kernel-size", "3",
                "--preprocess-sigma", "0.5",
                "--reattack-kind", "fgsm",
                "--reattack-eps", eps,  // 変化させるパラメータ
                "--batch-size", "128",
                "--num-samples", "-1",
                "--no-save-images",
            ])
            // 標準出力とエラー出力を同じファイルハンドルに接続
            // try_clone()を使用して、各プロセスにファイルへの書き込み権限を複製して渡す
            .stdout(Stdio::from(file.try_clone()?))
            .stderr(Stdio::from(file.try_clone()?))
            .status()?;

        if !status.success() {
            eprintln!("警告: eps={} の実行中にエラーが発生しました", eps);
        }
    }

    writeln!(file, "
=== Experiment Finished ===")?;
    println!("全ての実験が完了しました。");

    Ok(())
}
