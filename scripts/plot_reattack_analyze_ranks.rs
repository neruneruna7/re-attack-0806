#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! csv = "1.3"
//! serde = { version = "1.0", features = ["derive"] }
//! plotters = "0.3.5"
//! rand = "0.8"
//! ```

use plotters::prelude::full_palette::GREY;
use plotters::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;

#[derive(Debug, serde::Deserialize)]
struct Record {
    index: usize,
    target_label: u32,
    pred_clean: u32,
    pred_attacked: u32,
    pred_reattacked: u32,
    #[serde(rename = "rank_correct_attack")]
    rank_attack: u32,
    #[serde(rename = "rank_correct_reattack")]
    rank_reattack: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let default_path = "0_reattack_results.csv".to_string();
    let file_path = if args.len() > 1 {
        &args[1]
    } else {
        &default_path
    };

    println!("Reading CSV from: {}", file_path);

    let file = fs::File::open(file_path)
        .map_err(|e| format!("Failed to open file {}: {}", file_path, e))?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut data_points = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        // クリーン正解 かつ 攻撃成功
        if record.pred_clean == record.target_label && record.pred_attacked != record.target_label {
            data_points.push((record.rank_attack, record.rank_reattack));
        }
    }

    println!(
        "Total successful attack samples extracted: {}",
        data_points.len()
    );

    if data_points.is_empty() {
        println!("No matching data found. Exiting.");
        return Ok(());
    }

    fs::create_dir_all("plots")?;

    draw_scatter_plot(&data_points, "plots/1_rank_scatter.png")?;
    println!("Saved plots/1_rank_scatter.png");

    draw_heatmap(&data_points, "plots/2_rank_transition.png")?;
    println!("Saved plots/2_rank_transition.png");

    draw_histogram(&data_points, "plots/3_rank_improvement.png")?;
    println!("Saved plots/3_rank_improvement.png");

    Ok(())
}

/// 散布図を描画する関数
fn draw_scatter_plot(data: &[(u32, u32)], out_path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_rank = data
        .iter()
        .map(|(a, r)| *a.max(r))
        .max()
        .unwrap_or(10)
        .max(10) as f64;

    // 0.5 から開始することで、1.0 が最初のグリッドに乗るように調整
    let range = 0.5..(max_rank + 0.5);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Rank Transition: Attack vs Re-Attack",
            ("sans-serif", 40).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(range.clone(), range.clone())?;

    // ラベル数をランク数に合わせる
    let labels_count = max_rank as usize;

    chart
        .configure_mesh()
        .x_desc("Rank after Attack")
        .y_desc("Rank after Re-Attack")
        .x_labels(labels_count)
        .y_labels(labels_count)
        .light_line_style(&WHITE) // 補助線を見えなくしてメイングリッドだけにする
        .draw()?;

    // 対角線 (y=x)
    chart
        .draw_series(LineSeries::new(
            vec![(0.5, 0.5), (max_rank + 0.5, max_rank + 0.5)],
            &RED.mix(0.5),
        ))?
        .label("No Change")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED.mix(0.5)));

    let mut rng = rand::thread_rng();

    chart.draw_series(data.iter().map(|(x, y)| {
        let jitter_x = rng.gen_range(-0.15..0.15);
        let jitter_y = rng.gen_range(-0.15..0.15);
        Circle::new(
            (*x as f64 + jitter_x, *y as f64 + jitter_y),
            3,
            BLUE.mix(0.3).filled(),
        )
    }))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

/// ヒートマップ（遷移行列）を描画する関数
fn draw_heatmap(data: &[(u32, u32)], out_path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (1024, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_rank = 10; // 表示する最大ランク
    let mut matrix = HashMap::new();
    let mut max_count = 0;

    for &(attack, reattack) in data {
        // 11以上のランクは "11" ( >10 ) として集計
        let x = if attack > max_rank {
            max_rank + 1
        } else {
            attack
        };
        let y = if reattack > max_rank {
            max_rank + 1
        } else {
            reattack
        };

        let counter = matrix.entry((x, y)).or_insert(0);
        *counter += 1;
        if *counter > max_count {
            max_count = *counter;
        }
    }

    let display_max = max_rank + 1;

    // ★修正: 0.5 ~ 11.5 の範囲で座標を作成
    // これにより、整数値 1.0, 2.0... が各区間の中心になる
    let range = 0.5..(display_max as f64 + 0.5);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Rank Transition Matrix (Counts)",
            ("sans-serif", 40).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(range.clone(), range.clone())?;

    chart
        .configure_mesh()
        .x_labels(display_max as usize) // ラベル数を合わせる
        .y_labels(display_max as usize)
        .x_desc("Rank after Attack")
        .y_desc("Rank after Re-Attack")
        .disable_x_mesh()
        .disable_y_mesh()
        .x_label_formatter(&|v| {
            if *v > 10.5 {
                ">10".to_string()
            } else {
                format!("{:.0}", v)
            }
        }) // >10 の表記
        .y_label_formatter(&|v| {
            if *v > 10.5 {
                ">10".to_string()
            } else {
                format!("{:.0}", v)
            }
        })
        .draw()?;

    for x in 1..=display_max {
        for y in 1..=display_max {
            if let Some(&count) = matrix.get(&(x, y)) {
                let intensity = (count as f64 / max_count as f64).sqrt();
                let color = HSLColor(0.65, 1.0, 0.5 * (2.0 - intensity));

                // ★修正: 矩形を (x-0.5, y-0.5) から (x+0.5, y+0.5) に描画
                // これで整数 x, y が箱の中心になる
                let x_f = x as f64;
                let y_f = y as f64;

                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x_f - 0.5, y_f - 0.5), (x_f + 0.5, y_f + 0.5)],
                    color.filled(),
                )))?;

                let text_color = if intensity > 0.5 { WHITE } else { BLACK };

                // テキスト位置は中心 (x, y) でOK
                chart.draw_series(std::iter::once(
                    EmptyElement::at((x_f, y_f))
                        + Text::new(
                            format!("{}", count),
                            (0, 0), // 中心に配置
                            ("sans-serif", 15).into_font().color(&text_color),
                        ),
                ))?;
            }
        }
    }

    Ok(())
}

/// 改善量ヒストグラムを描画する関数
fn draw_histogram(data: &[(u32, u32)], out_path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(out_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let improvements: Vec<i32> = data.iter().map(|(a, r)| *a as i32 - *r as i32).collect();

    let min_val = *improvements.iter().min().unwrap_or(&0);
    let max_val = *improvements.iter().max().unwrap_or(&0);

    let mut hist = HashMap::new();
    let mut max_freq = 0;
    for &val in &improvements {
        let count = hist.entry(val).or_insert(0);
        *count += 1;
        if *count > max_freq {
            max_freq = *count;
        }
    }

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Distribution of Rank Improvement",
            ("sans-serif", 40).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((min_val..max_val + 1).into_segmented(), 0..max_freq + 10)?;

    chart
        .configure_mesh()
        .x_desc("Rank Improvement (Positive = Improved)")
        .y_desc("Count")
        .draw()?;

    chart.draw_series((min_val..=max_val).map(|x| {
        let count = *hist.get(&x).unwrap_or(&0);
        let color = if x > 0 {
            GREEN.mix(0.7)
        } else if x < 0 {
            RED.mix(0.7)
        } else {
            GREY.mix(0.7)
        };

        Rectangle::new(
            [
                (SegmentValue::Exact(x), 0),
                (SegmentValue::Exact(x + 1), count),
            ],
            color.filled(),
        )
    }))?;

    Ok(())
}
