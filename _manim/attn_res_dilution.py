from manim import *
import numpy as np


class Dilution(Scene):
    """Shows each layer's contribution shrinking as network gets deeper."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Information Dilution", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # We'll show stacked bars for 4, 16, 64 layers
        configs = [
            (4, "4 layers", LEFT * 4),
            (16, "16 layers", ORIGIN),
            (64, "64 layers", RIGHT * 4),
        ]

        all_groups = VGroup()

        for n_layers, label_text, x_offset in configs:
            bar_height = 4.0
            bar_width = 2.0
            contribution = 1.0 / (n_layers + 1)  # +1 for embedding

            # Outer container
            container = Rectangle(
                width=bar_width, height=bar_height,
                stroke_color=GREY, stroke_width=1,
                fill_opacity=0
            )
            container.move_to(x_offset + DOWN * 0.3)

            # Highlight one layer's contribution
            highlight_height = contribution * bar_height
            highlight = Rectangle(
                width=bar_width - 0.04, height=max(highlight_height, 0.03),
                fill_color="#ff6b6b", fill_opacity=0.8,
                stroke_width=0
            )
            # Place it somewhere in the middle
            highlight.move_to(
                container.get_bottom() + UP * (bar_height * 0.5)
            )

            # Fill the rest with dim color
            rest = Rectangle(
                width=bar_width - 0.04, height=bar_height,
                fill_color="#4ecdc4", fill_opacity=0.2,
                stroke_width=0
            )
            rest.move_to(container.get_center())

            # Label
            label = Text(label_text, font_size=18, color=WHITE)
            label.next_to(container, UP, buff=0.15)

            # Percentage
            pct = Text(
                f"each layer: {contribution * 100:.1f}%",
                font_size=14, color=YELLOW
            )
            pct.next_to(container, DOWN, buff=0.15)

            group = VGroup(rest, highlight, container, label, pct)
            all_groups.add(group)

        # Animate each config
        for i, group in enumerate(all_groups):
            self.play(FadeIn(group), run_time=0.6)
            self.wait(0.3)

        # Arrow showing trend
        arrow = Arrow(
            start=LEFT * 3 + DOWN * 2.5,
            end=RIGHT * 3 + DOWN * 2.5,
            color="#ff6b6b", stroke_width=2
        )
        arrow_label = Text(
            "deeper = each layer matters less",
            font_size=18, color="#ff6b6b"
        )
        arrow_label.next_to(arrow, DOWN, buff=0.1)

        self.play(GrowArrow(arrow), Write(arrow_label), run_time=0.6)
        self.wait(2.0)
