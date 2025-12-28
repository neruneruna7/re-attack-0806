#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! plotters = "0.3"
//! ```

use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // === データ定義 ===
    // 横軸: 再攻撃強度の「分子」の値を使用します (2, 4, 8, 16)
    // これにより、グラフ描画時に整数の座標として扱えるため、分数表記が綺麗になります。
    let re_attack_numerators = vec![2.0, 4.0, 8.0, 16.0];

    // 縦軸データ
    // 1. Ensemble Accuracy (All)
    let ensemble_accuracy = vec![0.5321, 0.5773, 0.5207, 0.5066];

    // 2. Re-Attack Recovery (Top-1)
    let re_attack_recovery = vec![0.4533, 0.5460, 0.5475, 0.5474];

    // 3. Ensemble Recovery (Top-1)
    let ensemble_recovery = vec![0.5616, 0.6206, 0.5854, 0.5764];

    // === グラフ描画設定 ===
    let root = BitMapBackend::new("reattack_accuracy_fraction.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // 軸の範囲設定
    // 横軸: 分子の値なので 0 から 18 程度まで (最大値16を含む範囲)
    let x_range = 0f64..18.0f64;
    let y_range = 0.0f64..1.0f64;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Re-Attack Robustness (Initial Attack: 4/255)",
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart
        .configure_mesh()
        .x_desc("Re-Attack Strength (epsilon)")
        .y_desc("Accuracy / Recovery Rate")
        // ここがポイント: 値(分子)を受け取り、"/255" を付けてフォーマットする
        .x_label_formatter(&|x| format!("{:.0}/255", x))
        .draw()?;

    // === 系列の描画 ===

    // 1. Ensemble Accuracy (赤色)
    chart
        .draw_series(LineSeries::new(
            re_attack_numerators
                .iter()
                .zip(ensemble_accuracy.iter())
                .map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Ensemble Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // 2. Re-Attack Recovery (青色)
    chart
        .draw_series(LineSeries::new(
            re_attack_numerators
                .iter()
                .zip(re_attack_recovery.iter())
                .map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Re-Attack Recovery")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // 3. Ensemble Recovery (緑色)
    chart
        .draw_series(LineSeries::new(
            re_attack_numerators
                .iter()
                .zip(ensemble_recovery.iter())
                .map(|(&x, &y)| (x, y)),
            &GREEN,
        ))?
        .label("Ensemble Recovery")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // データ点（丸）の描画
    for (color, data) in [
        (&RED, &ensemble_accuracy),
        (&BLUE, &re_attack_recovery),
        (&GREEN, &ensemble_recovery),
    ] {
        chart.draw_series(PointSeries::of_element(
            re_attack_numerators
                .iter()
                .zip(data.iter())
                .map(|(&x, &y)| (x, y)),
            5,
            ShapeStyle::from(color).filled(),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;
    }

    // 凡例
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    println!("Graph saved to 'reattack_accuracy_fraction.png'");
    Ok(())
}
