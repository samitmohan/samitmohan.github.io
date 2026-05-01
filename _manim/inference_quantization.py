from manimlib import *
import numpy as np


class QuantizationOutliers(Scene):
    """Activation outlier problem and the SmoothQuant fix.

    Shows how a handful of outlier channels destroy per-tensor
    quantization precision, and how SmoothQuant rescales to fix it.
    """

    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        NORMAL_COLOR = "#4ecdc4"
        OUTLIER_COLOR = "#ff6b6b"
        QUANT_COLOR = "#ffd93d"
        SMOOTH_COLOR = "#52b788"
        LABEL_DIM = "#8888aa"

        # =====================================================================
        # STEP 1: Title
        # =====================================================================
        title = Text("The Outlier Problem in Quantization", font_size=30, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # =====================================================================
        # STEP 2: Histogram of activation values
        # =====================================================================
        hist_label = Text("Activation Distribution (per channel)", font_size=18, color=LABEL_DIM)
        hist_label.next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(hist_label), run_time=0.3)

        # Normal distribution bins: 99% of channels in [-1, 1]
        bin_edges = np.linspace(-1, 1, 21)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts = np.exp(-0.5 * (bin_centers ** 2) / 0.3**2)
        counts = counts / counts.max()

        bar_max_h = 2.0
        bar_w = 0.22
        hist_anchor = DOWN * 0.8 + LEFT * 0.5

        # X-axis
        x_axis = Line(
            hist_anchor + LEFT * 5.5 + DOWN * 0.1,
            hist_anchor + RIGHT * 5.5 + DOWN * 0.1,
            color=GREY_B, stroke_width=1.5,
        )
        self.play(ShowCreation(x_axis), run_time=0.3)

        # Axis labels
        label_neg1 = Text("-1", font_size=14, color=GREY_B)
        label_neg1.move_to(hist_anchor + DOWN * 0.35 + LEFT * 2.0)
        label_pos1 = Text("+1", font_size=14, color=GREY_B)
        label_pos1.move_to(hist_anchor + DOWN * 0.35 + RIGHT * 2.0)
        label_neg100 = Text("-100", font_size=14, color=OUTLIER_COLOR)
        label_neg100.move_to(hist_anchor + DOWN * 0.35 + LEFT * 4.5)
        label_pos100 = Text("+100", font_size=14, color=OUTLIER_COLOR)
        label_pos100.move_to(hist_anchor + DOWN * 0.35 + RIGHT * 4.5)

        # Normal bars
        normal_bars = VGroup()
        for i, (bc, c) in enumerate(zip(bin_centers, counts)):
            bar = Rectangle(
                width=bar_w, height=bar_max_h * c,
                stroke_color=NORMAL_COLOR, stroke_width=1,
                fill_color=NORMAL_COLOR, fill_opacity=0.6,
            )
            bar.move_to(hist_anchor + RIGHT * bc * 2.0)
            bar.align_to(x_axis, DOWN)
            bar.shift(UP * 0.1)
            normal_bars.add(bar)

        self.play(
            *[GrowFromEdge(b, DOWN) for b in normal_bars],
            FadeIn(label_neg1), FadeIn(label_pos1),
            run_time=0.8,
        )

        # 99% label
        pct_label = Text("99% of channels", font_size=16, color=NORMAL_COLOR)
        pct_label.next_to(normal_bars, UP, buff=0.15)
        self.play(FadeIn(pct_label), run_time=0.3)
        self.wait(0.3)

        # 3 outlier bars at extremes
        outlier_bars = VGroup()
        outlier_positions = [-4.5, -4.0, 4.5]
        outlier_heights = [0.5, 0.3, 0.6]

        for pos, h in zip(outlier_positions, outlier_heights):
            bar = Rectangle(
                width=bar_w, height=bar_max_h * h,
                stroke_color=OUTLIER_COLOR, stroke_width=1.5,
                fill_color=OUTLIER_COLOR, fill_opacity=0.7,
            )
            bar.move_to(hist_anchor + RIGHT * pos)
            bar.align_to(x_axis, DOWN)
            bar.shift(UP * 0.1)
            outlier_bars.add(bar)

        outlier_label = Text("3 outlier channels", font_size=16, color=OUTLIER_COLOR)
        outlier_label.next_to(outlier_bars, UP, buff=0.15)

        self.play(
            *[GrowFromEdge(b, DOWN) for b in outlier_bars],
            FadeIn(outlier_label),
            FadeIn(label_neg100), FadeIn(label_pos100),
            run_time=0.6,
        )
        self.wait(0.5)

        # =====================================================================
        # STEP 3: Per-tensor quantization problem
        # =====================================================================
        step2_group = VGroup(
            normal_bars, outlier_bars, x_axis,
            label_neg1, label_pos1, label_neg100, label_pos100,
            pct_label, outlier_label, hist_label,
        )
        self.play(
            step2_group.animate.shift(UP * 1.5).scale(0.55),
            run_time=0.6,
        )

        # Quantization range bar
        quant_label = Text("Per-Tensor INT8 Quantization Range", font_size=20, color=QUANT_COLOR)
        quant_label.move_to(DOWN * 0.3)

        # Full range bar [-100, 100]
        full_range = Rectangle(
            width=10.0, height=0.6,
            stroke_color=QUANT_COLOR, stroke_width=2,
            fill_color=QUANT_COLOR, fill_opacity=0.1,
        )
        full_range.next_to(quant_label, DOWN, buff=0.25)

        range_left = Text("-100", font_size=14, color=QUANT_COLOR)
        range_left.next_to(full_range, LEFT, buff=0.1)
        range_right = Text("+100", font_size=14, color=QUANT_COLOR)
        range_right.next_to(full_range, RIGHT, buff=0.1)

        self.play(
            FadeIn(quant_label),
            FadeIn(full_range),
            FadeIn(range_left), FadeIn(range_right),
            run_time=0.5,
        )

        # Tiny sliver for the [-1, 1] range
        sliver_width = 10.0 * (2.0 / 200.0)
        sliver = Rectangle(
            width=sliver_width, height=0.6,
            stroke_color=NORMAL_COLOR, stroke_width=2,
            fill_color=NORMAL_COLOR, fill_opacity=0.6,
        )
        sliver.move_to(full_range.get_center())

        sliver_label = Text("99% of values crammed here", font_size=14, color=NORMAL_COLOR)
        sliver_label.next_to(sliver, DOWN, buff=0.15)

        sliver_arrow = Arrow(
            sliver_label.get_top(), sliver.get_bottom(),
            buff=0.05, color=NORMAL_COLOR, stroke_width=2,
            max_tip_length_to_length_ratio=0.3,
        )

        self.play(
            FadeIn(sliver),
            FadeIn(sliver_label),
            GrowArrow(sliver_arrow),
            run_time=0.6,
        )

        # Waste label
        waste_text = Text(
            "99% of precision wasted on outliers",
            font_size=22, color=OUTLIER_COLOR, weight=BOLD,
        )
        waste_text.next_to(full_range, DOWN, buff=0.8)
        self.play(Write(waste_text), run_time=0.8)
        self.wait(0.5)

        # =====================================================================
        # STEP 4: SmoothQuant fix
        # =====================================================================
        step3_group = VGroup(
            quant_label, full_range, range_left, range_right,
            sliver, sliver_label, sliver_arrow, waste_text,
        )
        self.play(
            FadeOut(step2_group),
            FadeOut(step3_group),
            run_time=0.5,
        )

        smooth_title = Text("SmoothQuant Fix", font_size=28, color=SMOOTH_COLOR, weight=BOLD)
        smooth_title.to_edge(UP, buff=0.3)
        self.play(FadeIn(smooth_title), run_time=0.4)

        # Formula
        formula = Text(
            "X_smooth = X / s     W_smooth = W * s",
            font_size=22, color=WHITE,
        )
        formula.next_to(smooth_title, DOWN, buff=0.4)
        self.play(Write(formula), run_time=0.6)

        # Before: activation histogram with outliers
        before_label = Text("Before: Activations", font_size=18, color=OUTLIER_COLOR)
        before_label.move_to(LEFT * 3.5 + UP * 0.5)

        # Compact histogram with outliers
        before_bars = VGroup()
        before_anchor = LEFT * 3.5 + DOWN * 0.5
        for bc, c in zip(bin_centers[::2], counts[::2]):
            bar = Rectangle(
                width=0.15, height=1.2 * c,
                stroke_color=NORMAL_COLOR, stroke_width=1,
                fill_color=NORMAL_COLOR, fill_opacity=0.5,
            )
            bar.move_to(before_anchor + RIGHT * bc * 1.2)
            bar.align_to(before_anchor, DOWN)
            before_bars.add(bar)

        before_outlier1 = Rectangle(
            width=0.15, height=0.5,
            stroke_color=OUTLIER_COLOR, stroke_width=1.5,
            fill_color=OUTLIER_COLOR, fill_opacity=0.7,
        )
        before_outlier1.move_to(before_anchor + LEFT * 2.5)
        before_outlier1.align_to(before_anchor, DOWN)
        before_outlier2 = Rectangle(
            width=0.15, height=0.6,
            stroke_color=OUTLIER_COLOR, stroke_width=1.5,
            fill_color=OUTLIER_COLOR, fill_opacity=0.7,
        )
        before_outlier2.move_to(before_anchor + RIGHT * 2.5)
        before_outlier2.align_to(before_anchor, DOWN)

        # Smoothing arrow
        smooth_arrow = Arrow(
            LEFT * 0.8 + DOWN * 0.3, RIGHT * 0.8 + DOWN * 0.3,
            buff=0.1, color=SMOOTH_COLOR, stroke_width=3,
            max_tip_length_to_length_ratio=0.15,
        )
        divide_label = Text("/ s", font_size=20, color=SMOOTH_COLOR, weight=BOLD)
        divide_label.next_to(smooth_arrow, UP, buff=0.1)

        # After: well-behaved histogram
        after_label = Text("After: Activations", font_size=18, color=SMOOTH_COLOR)
        after_label.move_to(RIGHT * 3.5 + UP * 0.5)

        after_bars = VGroup()
        after_anchor = RIGHT * 3.5 + DOWN * 0.5
        after_bin_centers = np.linspace(-1, 1, 15)
        after_counts = np.exp(-0.5 * (after_bin_centers ** 2) / 0.4**2)
        after_counts = after_counts / after_counts.max()

        for bc, c in zip(after_bin_centers, after_counts):
            bar = Rectangle(
                width=0.15, height=1.2 * c,
                stroke_color=SMOOTH_COLOR, stroke_width=1,
                fill_color=SMOOTH_COLOR, fill_opacity=0.5,
            )
            bar.move_to(after_anchor + RIGHT * bc * 1.5)
            bar.align_to(after_anchor, DOWN)
            after_bars.add(bar)

        self.play(
            FadeIn(before_label),
            *[GrowFromEdge(b, DOWN) for b in before_bars],
            FadeIn(before_outlier1, shift=UP * 0.2),
            FadeIn(before_outlier2, shift=UP * 0.2),
            run_time=0.6,
        )
        self.wait(0.3)

        self.play(
            GrowArrow(smooth_arrow),
            FadeIn(divide_label),
            run_time=0.4,
        )

        self.play(
            FadeIn(after_label),
            *[GrowFromEdge(b, DOWN) for b in after_bars],
            run_time=0.6,
        )

        well_behaved = Text("Quantization-friendly", font_size=16, color=SMOOTH_COLOR)
        well_behaved.next_to(after_bars, DOWN, buff=0.2)
        self.play(FadeIn(well_behaved), run_time=0.3)
        self.wait(0.4)

        # =====================================================================
        # STEP 5: Final comparison
        # =====================================================================
        step4_elements = VGroup(
            formula, before_label, before_bars,
            before_outlier1, before_outlier2,
            smooth_arrow, divide_label,
            after_label, after_bars, well_behaved,
        )
        self.play(
            FadeOut(step4_elements),
            FadeOut(smooth_title),
            run_time=0.5,
        )

        final_title = Text("SmoothQuant: Migration Difficulty", font_size=26, color=WHITE, weight=BOLD)
        final_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(final_title), run_time=0.3)

        # Before box
        before_box = Rectangle(
            width=5.0, height=2.2,
            stroke_color=OUTLIER_COLOR, stroke_width=2,
            fill_color=OUTLIER_COLOR, fill_opacity=0.08,
        )
        before_title_text = Text("Before", font_size=22, color=OUTLIER_COLOR, weight=BOLD)
        before_act = Text("Activations: hard to quantize", font_size=16, color=OUTLIER_COLOR)
        before_wt = Text("Weights: easy to quantize", font_size=16, color=SMOOTH_COLOR)
        before_content = VGroup(before_title_text, before_act, before_wt).arrange(DOWN, buff=0.2)
        before_content.move_to(before_box.get_center())
        before_panel = VGroup(before_box, before_content)

        # After box
        after_box = Rectangle(
            width=5.0, height=2.2,
            stroke_color=SMOOTH_COLOR, stroke_width=2,
            fill_color=SMOOTH_COLOR, fill_opacity=0.08,
        )
        after_title_text = Text("After (SmoothQuant)", font_size=22, color=SMOOTH_COLOR, weight=BOLD)
        after_act = Text("Activations: easy to quantize", font_size=16, color=SMOOTH_COLOR)
        after_wt = Text("Weights: slightly harder", font_size=16, color=QUANT_COLOR)
        after_content = VGroup(after_title_text, after_act, after_wt).arrange(DOWN, buff=0.2)
        after_content.move_to(after_box.get_center())
        after_panel = VGroup(after_box, after_content)

        panels = VGroup(before_panel, after_panel).arrange(RIGHT, buff=0.8)
        panels.next_to(final_title, DOWN, buff=0.6)

        arrow_between = Arrow(
            before_panel.get_right(), after_panel.get_left(),
            buff=0.15, color=WHITE, stroke_width=3,
            max_tip_length_to_length_ratio=0.12,
        )

        self.play(
            FadeIn(before_panel, shift=RIGHT * 0.2),
            FadeIn(after_panel, shift=LEFT * 0.2),
            GrowArrow(arrow_between),
            run_time=0.7,
        )

        # Bottom result
        result = Text(
            "Both activations and weights now fit in INT8 with minimal accuracy loss",
            font_size=18, color=QUANT_COLOR,
        )
        result.next_to(panels, DOWN, buff=0.5)
        self.play(Write(result), run_time=0.8)

        after_glow = SurroundingRectangle(
            after_panel, color=SMOOTH_COLOR, buff=0.08, stroke_width=3
        )
        self.play(ShowCreation(after_glow), run_time=0.4)
        self.wait(1.5)
