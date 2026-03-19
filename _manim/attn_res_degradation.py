from manim import *
import numpy as np


class Degradation(Scene):
    """Bar chart showing deeper plain networks have HIGHER error - the degradation problem."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        # Title
        title = Text("The Degradation Problem", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.4)
        subtitle = Text(
            "Deeper plain networks = worse accuracy (not overfitting)",
            font_size=16, color=GREY_B
        )
        subtitle.next_to(title, DOWN, buff=0.15)

        self.play(Write(title), run_time=0.6)
        self.play(FadeIn(subtitle), run_time=0.4)
        self.wait(0.3)

        # Data: CIFAR-10 test error rates from ResNet project
        models = ["Plain-20", "Plain-32", "Plain-44", "Plain-56"]
        errors = [9.26, 10.00, 11.22, 13.58]

        # Bar dimensions
        n_bars = len(models)
        bar_width = 1.1
        total_width = n_bars * bar_width + (n_bars - 1) * 0.35
        start_x = -total_width / 2 + bar_width / 2
        max_height = 3.8
        base_y = -2.2

        max_err = max(errors)
        min_err = 0

        # Baseline axis
        axis_line = Line(
            start=LEFT * (total_width / 2 + 0.3) + UP * base_y,
            end=RIGHT * (total_width / 2 + 0.3) + UP * base_y,
            color=GREY, stroke_width=1.5
        )
        self.play(Create(axis_line), run_time=0.3)

        # Color gradient: cyan (good) to red (bad)
        color_low = ManimColor("#4ecdc4")
        color_high = ManimColor("#ff6b6b")

        bars = VGroup()

        # Reference line at Plain-20 level
        ref_height = (errors[0] / max_err) * max_height
        ref_y = base_y + ref_height

        for i, (model, error) in enumerate(zip(models, errors)):
            height = (error / max_err) * max_height
            t = (error - min(errors)) / (max(errors) - min(errors))

            bar = Rectangle(
                width=bar_width,
                height=height,
                fill_color=interpolate_color(color_low, color_high, t),
                fill_opacity=0.85,
                stroke_color=WHITE,
                stroke_width=1
            )
            x_pos = start_x + i * (bar_width + 0.35)
            bar.move_to(RIGHT * x_pos + UP * (base_y + height / 2))
            bars.add(bar)

            # Model label below bar
            label = Text(model, font_size=16, color=GREY_B)
            label.next_to(bar, DOWN, buff=0.1)

            # Error value above bar
            val = Text(f"{error}%", font_size=18, color=YELLOW)
            val.next_to(bar, UP, buff=0.08)

            self.play(
                GrowFromEdge(bar, DOWN),
                FadeIn(label),
                FadeIn(val),
                run_time=0.45
            )

        self.wait(0.3)

        # Dashed reference line at Plain-20 level
        ref_line = DashedLine(
            start=LEFT * (total_width / 2 + 0.3) + UP * ref_y,
            end=RIGHT * (total_width / 2 + 0.3) + UP * ref_y,
            color="#4ecdc4", stroke_width=1.5, dash_length=0.1
        )
        ref_label = Text("Plain-20 baseline", font_size=12, color="#4ecdc4")
        ref_label.next_to(ref_line, RIGHT, buff=0.1)

        self.play(Create(ref_line), FadeIn(ref_label), run_time=0.4)
        self.wait(0.3)

        # Arrow showing wrong direction
        arrow = Arrow(
            start=bars[0].get_top() + UP * 0.6,
            end=bars[-1].get_top() + UP * 0.6,
            color="#ff6b6b",
            stroke_width=3,
            max_tip_length_to_length_ratio=0.08
        )
        arrow_label = Text(
            "more layers = worse performance",
            font_size=18, color="#ff6b6b"
        )
        arrow_label.next_to(arrow, UP, buff=0.1)

        self.play(GrowArrow(arrow), Write(arrow_label), run_time=0.6)
        self.wait(2.0)
