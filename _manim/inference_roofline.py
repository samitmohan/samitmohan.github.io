from manimlib import *
import numpy as np


class RooflineModel(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        MEM_COLOR = "#ff6b6b"
        COMPUTE_COLOR = "#2ecc71"
        RIDGE_COLOR = "#ffd93d"
        AXIS_COLOR = "#8888aa"
        GRID_COLOR = "#333355"

        # --- Title ---
        title = Text(
            "The Roofline Model",
            font_size=30, color=WHITE, weight=BOLD
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # =====================================================================
        # AXES - log scale representation
        # =====================================================================
        # x: arithmetic intensity (ops/byte), log scale from 0.5 to 1024
        # y: performance (TFLOPS), log scale from 0.1 to 2000
        # We map log2 values to linear axes

        axes = Axes(
            x_range=[np.log2(0.5), np.log2(1024), 1],   # log2 space: -1 to 10
            y_range=[np.log2(0.5), np.log2(2048), 1],    # log2 space: -1 to 11
            width=9,
            height=5,
            axis_config={
                "color": AXIS_COLOR,
                "stroke_width": 2,
                "include_ticks": False,
            },
        ).shift(DOWN * 0.3 + LEFT * 0.2)

        x_label = Text("Arithmetic Intensity (ops/byte)", font_size=18, color=AXIS_COLOR)
        x_label.next_to(axes.x_axis, DOWN, buff=0.35)

        y_label = Text("Performance (TFLOPS)", font_size=18, color=AXIS_COLOR)
        y_label.next_to(axes.y_axis, LEFT, buff=0.35).rotate(PI / 2)

        # X-axis tick labels (log scale)
        x_ticks = VGroup()
        for exp in [0, 1, 2, 4, 8, 10]:
            val = 2 ** exp
            tick_pos = axes.c2p(exp, np.log2(0.5))
            tick_mark = Line(
                tick_pos + DOWN * 0.08, tick_pos + UP * 0.08,
                color=AXIS_COLOR, stroke_width=1.5
            )
            label = Text(str(val), font_size=12, color="#666688")
            label.next_to(tick_mark, DOWN, buff=0.08)
            x_ticks.add(tick_mark, label)

        # Y-axis tick labels (log scale)
        y_ticks = VGroup()
        for tflops in [1, 10, 100, 990]:
            log_val = np.log2(tflops)
            tick_pos = axes.c2p(np.log2(0.5), log_val)
            tick_mark = Line(
                tick_pos + LEFT * 0.08, tick_pos + RIGHT * 0.08,
                color=AXIS_COLOR, stroke_width=1.5
            )
            label = Text(str(tflops), font_size=12, color="#666688")
            label.next_to(tick_mark, LEFT, buff=0.08)
            y_ticks.add(tick_mark, label)

        self.play(
            ShowCreation(axes), Write(x_label), Write(y_label),
            FadeIn(x_ticks), FadeIn(y_ticks),
            run_time=0.8,
        )
        self.wait(0.3)

        # =====================================================================
        # ROOFLINE LINES
        # =====================================================================
        # Memory bandwidth ceiling: perf = bandwidth * arithmetic_intensity
        # H100 SXM: bandwidth = 3.35 TB/s, peak compute = 990 TFLOPS (BF16)
        # Ridge point: 990 / 3.35 = ~295 ops/byte

        bandwidth = 3.35  # TB/s = TFLOPS per ops/byte
        peak_compute = 990  # TFLOPS
        ridge_ai = peak_compute / bandwidth  # ~295 ops/byte
        log_ridge = np.log2(ridge_ai)

        # Memory-bound diagonal: from left edge to ridge point
        # perf = bandwidth * AI => log2(perf) = log2(bandwidth) + log2(AI)
        mem_line = axes.get_graph(
            lambda x: np.log2(bandwidth) + x,  # x is log2(AI)
            x_range=[np.log2(0.5), log_ridge],
            color=MEM_COLOR,
            stroke_width=3,
        )

        # Compute ceiling: horizontal line from ridge point to right
        compute_line = axes.get_graph(
            lambda x: np.log2(peak_compute),
            x_range=[log_ridge, np.log2(1024)],
            color=COMPUTE_COLOR,
            stroke_width=3,
        )

        # Memory-bound region shading
        mem_region_points = [
            axes.c2p(np.log2(0.5), np.log2(0.5)),
            axes.c2p(log_ridge, np.log2(peak_compute)),
            axes.c2p(log_ridge, np.log2(0.5)),
        ]
        mem_region = Polygon(
            *mem_region_points,
            stroke_width=0,
            fill_color=MEM_COLOR, fill_opacity=0.08,
        )

        # Compute-bound region shading
        compute_region_points = [
            axes.c2p(log_ridge, np.log2(0.5)),
            axes.c2p(log_ridge, np.log2(peak_compute)),
            axes.c2p(np.log2(1024), np.log2(peak_compute)),
            axes.c2p(np.log2(1024), np.log2(0.5)),
        ]
        compute_region = Polygon(
            *compute_region_points,
            stroke_width=0,
            fill_color=COMPUTE_COLOR, fill_opacity=0.08,
        )

        # Labels for regions
        mem_region_label = Text("Memory-bound", font_size=16, color=MEM_COLOR)
        mem_region_label.move_to(axes.c2p(np.log2(4), np.log2(4)))

        compute_region_label = Text("Compute-bound", font_size=16, color=COMPUTE_COLOR)
        compute_region_label.move_to(axes.c2p(np.log2(600), np.log2(4)))

        # Line labels
        bw_label = Text("BW = 3.35 TB/s", font_size=14, color=MEM_COLOR)
        bw_label.next_to(axes.c2p(np.log2(8), np.log2(bandwidth * 8)), UL, buff=0.1)

        peak_label = Text("Peak = 990 TFLOPS", font_size=14, color=COMPUTE_COLOR)
        peak_label.next_to(axes.c2p(np.log2(800), np.log2(peak_compute)), UP, buff=0.1)

        # Animate the two lines and regions
        self.play(
            FadeIn(mem_region),
            ShowCreation(mem_line),
            FadeIn(mem_region_label),
            Write(bw_label),
            run_time=0.8,
        )
        self.wait(0.3)

        self.play(
            FadeIn(compute_region),
            ShowCreation(compute_line),
            FadeIn(compute_region_label),
            Write(peak_label),
            run_time=0.8,
        )
        self.wait(0.3)

        # =====================================================================
        # RIDGE POINT
        # =====================================================================
        ridge_pos = axes.c2p(log_ridge, np.log2(peak_compute))
        ridge_dot = Dot(ridge_pos, color=RIDGE_COLOR, radius=0.1)
        ridge_label = Text("Ridge Point\n~295 ops/byte", font_size=14, color=RIDGE_COLOR)
        ridge_label.next_to(ridge_dot, UR, buff=0.15)

        self.play(FadeIn(ridge_dot), Write(ridge_label), run_time=0.6)
        self.wait(0.3)

        # =====================================================================
        # WORKLOAD DOTS
        # =====================================================================
        # Decode: ~2 ops/byte, sits on memory-bound diagonal
        decode_ai = 2.0
        decode_perf = bandwidth * decode_ai  # ~6.7 TFLOPS
        decode_pos = axes.c2p(np.log2(decode_ai), np.log2(decode_perf))
        decode_dot = Dot(decode_pos, color=MEM_COLOR, radius=0.12)
        decode_label = Text("Decode\n(~2 ops/byte)", font_size=14, color=MEM_COLOR, weight=BOLD)
        decode_label.next_to(decode_dot, DR, buff=0.15)

        # Prefill: ~300 ops/byte, near compute ceiling
        prefill_ai = 300.0
        prefill_perf = peak_compute  # at ceiling
        prefill_pos = axes.c2p(np.log2(prefill_ai), np.log2(prefill_perf))
        prefill_dot = Dot(prefill_pos, color=COMPUTE_COLOR, radius=0.12)
        prefill_label = Text("Prefill\n(~300 ops/byte)", font_size=14, color=COMPUTE_COLOR, weight=BOLD)
        prefill_label.next_to(prefill_dot, UP, buff=0.15)

        self.play(
            FadeIn(decode_dot), Write(decode_label),
            run_time=0.6,
        )
        self.wait(0.3)

        self.play(
            FadeIn(prefill_dot), Write(prefill_label),
            run_time=0.6,
        )
        self.wait(0.3)

        # =====================================================================
        # UTILIZATION CALLOUT
        # =====================================================================
        # Decode uses ~6.7 / 990 = ~0.7% of compute
        util_pct = (decode_perf / peak_compute) * 100

        # Dashed line from decode dot up to compute ceiling
        ceiling_pos = axes.c2p(np.log2(decode_ai), np.log2(peak_compute))
        gap_line = DashedLine(
            decode_pos, ceiling_pos,
            color=RIDGE_COLOR, stroke_width=1.5, dash_length=0.08
        )
        gap_label = Text(
            f"~{util_pct:.0f}% GPU utilization",
            font_size=14, color=RIDGE_COLOR, weight=BOLD
        )
        gap_label.next_to(gap_line, LEFT, buff=0.15)

        self.play(ShowCreation(gap_line), Write(gap_label), run_time=0.6)
        self.wait(0.5)

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        summary = Text(
            "Decode wastes >99% of GPU compute. Batching is the fix.",
            font_size=18, color=RIDGE_COLOR,
        )
        summary.to_edge(DOWN, buff=0.35)

        summary_box = SurroundingRectangle(
            summary, buff=0.12,
            stroke_color=RIDGE_COLOR, stroke_width=2,
            fill_color=RIDGE_COLOR, fill_opacity=0.08,
        )
        self.play(Write(summary), ShowCreation(summary_box), run_time=0.7)
        self.wait(1.5)
