#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! plotters = "0.3"
//! ```

use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // === データ定義 ===
    // 横軸: Epsilon (摂動の大きさ)
    let epsilons = vec![
        2.0 / 255.0,  // 0.0078
        4.0 / 255.0,  // 0.0157
        8.0 / 255.0,  // 0.0314
        16.0 / 255.0, // 0.0627
    ];

    // 系列A: 再攻撃のみ (前処理なし) - 防御成功率 (Recovery)
    // 参考画像: グレーの線, マーカー 'x'
    let recovery_reattack_only = vec![0.4533, 0.5460, 0.5475, 0.5474];

    // 系列B: 前処理 + 再攻撃 - 防御成功率 (Recovery)
    // 参考画像: オレンジの線, マーカー '●'
    let recovery_prep_reattack = vec![0.7906, 0.7932, 0.7928, 0.7928];

    // 系列C: 前処理 + 再攻撃 - 全体正解率 (Accuracy)
    // 参考画像: 青の線, マーカー '■'
    let accuracy_prep_reattack = vec![0.7026, 0.6863, 0.6836, 0.6836];

    // === グラフ描画設定 ===
    let output_file = "gaussian_defense_graph.png";
    let root = BitMapBackend::new(output_file, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // フォント設定 (日本語表示用)
    // Mac: "Hiragino Sans", Windows: "Meiryo", Linux: "Noto Sans CJK JP" など
    // 環境に合わせて適宜変更してください。
    let font_family = "Hiragino Sans";
    let title_style = (font_family, 30).into_font().style(FontStyle::Bold);
    let label_style = (font_family, 20).into_font();
    let axis_style = (font_family, 16).into_font();

    // 軸の範囲
    let x_max = 18.0 / 255.0; // 少し余裕を持たせる
    let x_range = 0f64..x_max;
    let y_range = 0.0f64..1.0f64;

    let mut chart = ChartBuilder::on(&root)
        .caption("前処理手法: ガウシアンブラー における防御性能", title_style)
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range, y_range)?;

    // メッシュ（グリッド）と軸ラベルの設定
    chart
        .configure_mesh()
        .x_desc("Epsilon (摂動の大きさ)")
        .y_desc("確率 (Probability)")
        .axis_desc_style(label_style) // 軸ラベルのフォント
        .label_style(axis_style) // 目盛りのフォント
        .light_line_style(&WHITE.mix(0.8)) // グリッドを薄く
        .x_label_formatter(&|x| format!("{:.3}", x)) // 小数点3桁表示 (0.008, 0.016...)
        .draw()?;

    // ==========================================
    // 系列の描画
    // ==========================================

    let point_size = 6;
    let line_width = 2;

    // 1. [GRAY] 再攻撃のみ (前処理なし) -> マーカー: X
    // plottersで破線は難しいため、グレーの実線で表現します
    let color_gray = RGBColor(105, 105, 105);
    chart
        .draw_series(LineSeries::new(
            epsilons
                .iter()
                .zip(recovery_reattack_only.iter())
                .map(|(&x, &y)| (x, y)),
            &color_gray,
        ))?
        .label("再攻撃のみ (前処理なし)")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color_gray));

    chart.draw_series(PointSeries::of_element(
        epsilons
            .iter()
            .zip(recovery_reattack_only.iter())
            .map(|(&x, &y)| (x, y)),
        point_size,
        ShapeStyle::from(&color_gray).stroke_width(2), // 塗りつぶしなしのX
        &|coord, size, style| EmptyElement::at(coord) + Cross::new((0, 0), size, style),
    ))?;

    // 2. [ORANGE] 前処理+再攻撃 (防御成功率) -> マーカー: Circle (●)
    // 視認性の良いオレンジ
    let color_orange = RGBColor(230, 159, 0);
    chart
        .draw_series(LineSeries::new(
            epsilons
                .iter()
                .zip(recovery_prep_reattack.iter())
                .map(|(&x, &y)| (x, y)),
            ShapeStyle::from(&color_orange).stroke_width(line_width),
        ))?
        .label("前処理+再攻撃 (防御成功率)")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color_orange));

    chart.draw_series(PointSeries::of_element(
        epsilons
            .iter()
            .zip(recovery_prep_reattack.iter())
            .map(|(&x, &y)| (x, y)),
        point_size,
        ShapeStyle::from(&color_orange).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    // 3. [BLUE] 前処理+再攻撃 (全体正解率) -> マーカー: Square (■)
    // 視認性の良い青
    let color_blue = RGBColor(0, 114, 178);
    chart
        .draw_series(LineSeries::new(
            epsilons
                .iter()
                .zip(accuracy_prep_reattack.iter())
                .map(|(&x, &y)| (x, y)),
            ShapeStyle::from(&color_blue).stroke_width(line_width),
        ))?
        .label("前処理+再攻撃 (全体正解率)")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color_blue));

    chart.draw_series(PointSeries::of_element(
        epsilons
            .iter()
            .zip(accuracy_prep_reattack.iter())
            .map(|(&x, &y)| (x, y)),
        point_size,
        ShapeStyle::from(&color_blue).filled(),
        &|coord, size, style| {
            // 四角形の描画
            EmptyElement::at(coord) + Rectangle::new([(0 - size, 0 - size), (size, size)], style)
        },
    ))?;

    // === 凡例の描画 ===
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK.mix(0.5))
        .position(SeriesLabelPosition::LowerRight) // 右下に配置（グラフとかぶらないように）
        .label_font((font_family, 18).into_font()) // 凡例のフォントサイズ
        .draw()?;

    println!("Graph saved to '{}'", output_file);
    Ok(())
}
