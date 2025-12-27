#!/usr/bin/env rust-script

//! ```cargo
//! [dependencies]
//! plotters = "0.3"
//! csv = "1.1"
//! serde = { version = "1.0", features = ["derive"] }
//! ```

use plotters::prelude::full_palette::GREY;
use plotters::prelude::*;
use serde::Deserialize;
use std::error::Error;
use std::path::Path;

// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
// 設定: ここに (Epsilon, ファイルパス) のペアを記述します
// 文字列の前に 'r' を付けるとRaw文字列となり、パスの記述が楽になります
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
const INPUTS: &[(f64, &str)] = &[
    (
        0.01,
        r"preprocessed_reattacked_data/cifar10/resnet20/bim, epsilon: 0.03137254901960784, alpha: 0.00392156862745098, iters: 10/gaussian_blur, kernel_size: 3, sigma: 0.5/bim, epsilon: 0.00784313725490196, alpha: 0.00392156862745098, iters: 10/results.csv",
    ),
    (
        0.01,
        r"preprocessed_reattacked_data/cifar10/resnet20/bim, epsilon: 0.03137254901960784, alpha: 0.00392156862745098, iters: 10/gaussian_blur, kernel_size: 3, sigma: 0.5/bim, epsilon: 0.01568627450980392, alpha: 0.00392156862745098, iters: 10/results.csv",
    ),
    (
        0.0313,
        r"preprocessed_reattacked_data/cifar10/resnet20/bim, epsilon: 0.03137254901960784, alpha: 0.00392156862745098, iters: 10/gaussian_blur, kernel_size: 3, sigma: 0.5/bim, epsilon: 0.03137254901960784, alpha: 0.00392156862745098, iters: 10/results.csv",
    ),
    // 必要に応じて行を追加してください
    // (0.06, r"パス/to/results.csv"),
];

#[derive(Debug, Deserialize)]
struct Record {
    target: u32,
    pred_clean: u32,
    pred_attacked: u32,
    pred_preprocessed: u32,
    pred_reattacked: u32,
}

#[derive(Debug, Clone)]
struct AggregatedData {
    epsilon: f64,
    acc_attacked: f64,
    acc_reattacked: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut data_points: Vec<AggregatedData> = Vec::new();

    println!("=== Processing {} defined inputs ===", INPUTS.len());

    // 定数 INPUTS をイテレートして処理
    for (epsilon, path_str) in INPUTS {
        let path = Path::new(path_str);

        if !path.exists() {
            eprintln!("[Warning] File not found: {} (Eps: {})", path_str, epsilon);
            // ファイルがない場合でも続行する場合は continue
            // continue;
        }

        println!("Loading Eps: {} from {}", epsilon, path_str);

        match calculate_accuracies(path) {
            Ok((acc_attacked, acc_reattacked)) => {
                println!(
                    " -> Acc(Atk): {:.2}%, Acc(Def): {:.2}%",
                    acc_attacked * 100.0,
                    acc_reattacked * 100.0
                );
                data_points.push(AggregatedData {
                    epsilon: *epsilon,
                    acc_attacked,
                    acc_reattacked,
                });
            }
            Err(e) => {
                eprintln!(" -> [Error] Failed to read/parse CSV: {}", e);
            }
        }
    }

    if data_points.is_empty() {
        eprintln!("No valid data points found. Please check file paths in 'INPUTS'.");
        return Ok(());
    }

    // Epsilon順にソート
    data_points.sort_by(|a, b| a.epsilon.partial_cmp(&b.epsilon).unwrap());

    // グラフ描画設定
    let out_file = "manual_trend_plot.png";
    let root = BitMapBackend::new(out_file, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_epsilon = data_points.last().unwrap().epsilon;
    let x_max = if max_epsilon == 0.0 {
        0.1
    } else {
        max_epsilon * 1.1
    };

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Defense Performance vs Attack Intensity",
            ("sans-serif", 40),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..x_max, 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Epsilon")
        .y_desc("Accuracy")
        .draw()?;

    // 折れ線: 攻撃のみ (Attacked)
    chart
        .draw_series(LineSeries::new(
            data_points.iter().map(|d| (d.epsilon, d.acc_attacked)),
            &GREY,
        ))?
        .label("Attacked")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREY));

    chart.draw_series(
        data_points
            .iter()
            .map(|d| Cross::new((d.epsilon, d.acc_attacked), 5, GREY.filled())),
    )?;

    // 折れ線: 防御 + 再攻撃 (Re-attacked)
    let orange = RGBColor(255, 165, 0);
    let line_style = ShapeStyle {
        color: orange.to_rgba(),
        filled: true,
        stroke_width: 3,
    };

    chart
        .draw_series(LineSeries::new(
            data_points.iter().map(|d| (d.epsilon, d.acc_reattacked)),
            line_style,
        ))?
        .label("Defense + Re-attacked")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], line_style));

    chart.draw_series(
        data_points
            .iter()
            .map(|d| Circle::new((d.epsilon, d.acc_reattacked), 5, orange.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Graph generated successfully: {}", out_file);

    Ok(())
}

fn calculate_accuracies<P: AsRef<Path>>(path: P) -> Result<(f64, f64), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut total = 0.0;
    let mut correct_attacked = 0.0;
    let mut correct_reattacked = 0.0;

    for result in rdr.deserialize() {
        let record: Record = result?;
        total += 1.0;

        if record.target == record.pred_attacked {
            correct_attacked += 1.0;
        }
        if record.target == record.pred_reattacked {
            correct_reattacked += 1.0;
        }
    }

    if total == 0.0 {
        return Ok((0.0, 0.0));
    }

    Ok((correct_attacked / total, correct_reattacked / total))
}
