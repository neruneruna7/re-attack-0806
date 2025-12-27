#!/usr/bin/env rust-script

//! ```cargo
//! [dependencies]
//! csv = "1.1"
//! serde = { version = "1.0", features = ["derive"] }
//! ```

use serde::Deserialize;
use std::env;
use std::error::Error;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Record {
    target: u32,
    pred_clean: u32,
    pred_attacked: u32,
    pred_preprocessed: u32,
    pred_reattacked: u32,
    #[allow(dead_code)]
    image_path: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1. コマンドライン引数からファイルパスを取得
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rust-script calc_metrics.rs <csv_file_path>");
        return Ok(());
    }
    let file_path = &args[1];

    if !Path::new(file_path).exists() {
        eprintln!("Error: File '{}' not found.", file_path);
        return Ok(());
    }

    // 2. CSVリーダーの準備
    let mut rdr = csv::Reader::from_path(file_path)?;

    // カウンタ変数の初期化
    let mut total_samples = 0.0;

    // A. Clean Accuracy用
    let mut clean_correct_count = 0.0; // 元画像で正解した数 (ASRの分母になる)

    // B. Attack Success Rate用
    let mut attack_success_count = 0.0; // 元画像正解 かつ 攻撃成功した数 (DSRの分母になる)

    // C. Defense Success Rate用
    let mut defense_preproc_recovered = 0.0; // 攻撃成功のうち、前処理で回復した数
    let mut defense_reattack_recovered = 0.0; // 攻撃成功のうち、再攻撃対策で回復した数

    // D. Final Accuracy用
    let mut final_correct_count = 0.0; // 最終的な出力が正解した数 (全サンプル母数)

    // 3. レコードごとの集計
    for result in rdr.deserialize() {
        let record: Record = result?;
        total_samples += 1.0;

        // 元画像が正解かどうか
        let is_clean_correct = record.pred_clean == record.target;

        // A. Clean Accuracy (母数: 全サンプル)
        if is_clean_correct {
            clean_correct_count += 1.0;

            // B. 攻撃成功率 (母数: 元画像で正解したサンプル)
            // 元画像が正解だったものに対してのみ、攻撃成功判定を行う
            let is_attack_successful = record.pred_attacked != record.target;

            if is_attack_successful {
                attack_success_count += 1.0;

                // C. 防御成功率 (母数: 元画像正解 かつ 攻撃成功したサンプル)

                // C-1. 前処理のみで回復
                if record.pred_preprocessed == record.target {
                    defense_preproc_recovered += 1.0;
                }

                // C-2. 前処理 + 再攻撃で回復
                if record.pred_reattacked == record.target {
                    defense_reattack_recovered += 1.0;
                }
            }
        }

        // D. 最終的な全体正解率 (母数: 全サンプル)
        // ここは元画像の正解不正解に関わらず、最終結果が合っているかを見る
        if record.pred_reattacked == record.target {
            final_correct_count += 1.0;
        }
    }

    if total_samples == 0.0 {
        eprintln!("No data found in CSV.");
        return Ok(());
    }

    // 4. メトリクスの計算

    // A. Clean Accuracy
    let clean_accuracy = clean_correct_count / total_samples;

    // B. Attack Success Rate (母数: Clean Correct Count)
    // Clean Accuracyが0の場合は0.0とする
    let attack_success_rate = if clean_correct_count > 0.0 {
        attack_success_count / clean_correct_count
    } else {
        0.0
    };

    // C. Defense Success Rate (母数: Attack Success Count)
    let defense_success_preproc = if attack_success_count > 0.0 {
        defense_preproc_recovered / attack_success_count
    } else {
        0.0
    };

    let defense_success_reattack = if attack_success_count > 0.0 {
        defense_reattack_recovered / attack_success_count
    } else {
        0.0
    };

    // D. Final Accuracy
    let final_accuracy = final_correct_count / total_samples;

    // 5. 結果の出力
    println!("=== Evaluation Metrics ===");
    println!("Total Samples: {}", total_samples);
    println!(
        "Clean Correct Samples (Denominator for ASR): {}",
        clean_correct_count
    );
    println!(
        "Successful Attacks on Clean (Denominator for DSR): {}",
        attack_success_count
    );
    println!("--------------------------------------------------");

    println!(
        "1. Clean Accuracy (元画像の精度): {:.2}% ({}/{})",
        clean_accuracy * 100.0,
        clean_correct_count,
        total_samples
    );

    println!(
        "2. Attack Success Rate (攻撃成功率): {:.2}% ({}/{}) [Based on Clean Correct]",
        attack_success_rate * 100.0,
        attack_success_count,
        clean_correct_count
    );

    println!(
        "3. DSR [Preproc Only] (防御成功率: 前処理のみ): {:.2}% ({}/{})",
        defense_success_preproc * 100.0,
        defense_preproc_recovered,
        attack_success_count
    );

    println!(
        "4. DSR [Def + Re-attack] (防御成功率: 前処理+再攻撃): {:.2}% ({}/{})",
        defense_success_reattack * 100.0,
        defense_reattack_recovered,
        attack_success_count
    );

    println!(
        "5. Final Overall Accuracy (最終的な全体正解率): {:.2}% ({}/{})",
        final_accuracy * 100.0,
        final_correct_count,
        total_samples
    );

    Ok(())
}
