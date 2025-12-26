#!/usr/bin/env rust-script

//! ```cargo
//! [dependencies]
//! plotters = "0.3"
//! csv = "1.1"
//! serde = { version = "1.0", features = ["derive"] }
//! ```

use plotters::prelude::*;
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Record {
    // CSVのヘッダー名と一致させる必要があります
    // 必要に応じて型を変更してください (例: 文字列なら String)
    index: u32,
    target: u32,
    pred_clean: u32,
    pred_attacked: u32,
    pred_preprocessed: u32,
    pred_reattacked: u32,
    // image_pathはグラフに使わないので、読み飛ばしてもOKですが、
    // CSVにあるなら定義しておくとエラーになりにくいです
    // (使わないフィールドには _ をつける慣習がありますが、Deserializeのために名前は合わせます)
    image_path: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1. コマンドライン引数からファイルパスを取得
    let args: Vec<String> = env::args().collect();
    let default_path = "data.csv".to_string();
    let file_path = args.get(1).unwrap_or(&default_path);

    if !Path::new(file_path).exists() {
        eprintln!("Error: File '{}' not found.", file_path);
        eprintln!("Usage: rust-script plot.rs <path_to_csv>");
        return Ok(());
    }

    println!("Reading data from: {}", file_path);

    // 2. CSVファイルから読み込み
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    let total = records.len() as f64;
    if total == 0.0 {
        return Err("No data found in CSV".into());
    }
    println!("Total records: {}", total);

    // 手法名と判定ロジック
    let methods: [(&str, fn(&Record) -> bool); 4] = [
        ("Clean", |r| r.target == r.pred_clean),
        ("Attacked", |r| r.target == r.pred_attacked),
        ("Preproc", |r| r.target == r.pred_preprocessed),
        ("Re-attacked", |r| r.target == r.pred_reattacked),
    ];

    // 結果の集計
    let results: Vec<(&str, f64)> = methods
        .iter()
        .map(|(name, check)| {
            let correct = records.iter().filter(|r| check(r)).count();
            (*name, (correct as f64 / total) * 100.0)
        })
        .collect();

    // 集計結果をコンソールにも表示
    for (name, acc) in &results {
        println!("{}: {:.2}%", name, acc);
    }

    // 3. プロットの設定
    let out_file = "accuracy_comparison.png";
    let root = BitMapBackend::new(out_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let num_bars = results.len();
    let x_range = -0.5f64..(num_bars as f64 - 0.5f64);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Model Accuracy (n={})", total), ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_range, 0.0..100.0f64)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Accuracy (%)")
        .x_labels(num_bars)
        .x_label_formatter(&|x| {
            let index = x.round() as usize;
            if index < results.len() {
                results[index].0.to_string()
            } else {
                "".to_string()
            }
        })
        .draw()?;

    // 4. 棒グラフの描画
    chart.draw_series(results.iter().enumerate().map(|(i, (_name, acc))| {
        let x_center = i as f64;
        let bar_half_width = 0.4;

        Rectangle::new(
            [
                (x_center - bar_half_width, 0.0),
                (x_center + bar_half_width, *acc),
            ],
            BLUE.filled(),
        )
    }))?;

    root.present()?;

    println!("Graph has been saved to: {}", out_file);
    Ok(())
}
