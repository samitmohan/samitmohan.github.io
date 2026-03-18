from manim import *
import numpy as np


class PreNormProblem(Scene):
    """Shows how PreNorm bounds each layer's output but the sum still grows."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("The PreNorm Problem", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # Show: running sum (blue bar growing) + new layer contribution (red bar, fixed)
        n_steps = 8
        bar_width = 0.7
        max_total_height = 5.0
        fixed_contrib = 0.5  # Each layer contributes this much height
        base_y = -2.5
        start_x = -4.5

        sum_label = Text("running sum (x)", font_size=16, color="#4ecdc4")
        sum_label.move_to(UP * 2.8 + LEFT * 2.5)
        contrib_label = Text("+ Layer(Norm(x))", font_size=16, color="#ff6b6b")
        contrib_label.next_to(sum_label, DOWN, buff=0.1, aligned_edge=LEFT)

        sum_dot = Dot(color="#4ecdc4", radius=0.08).next_to(sum_label, LEFT, buff=0.1)
        contrib_dot = Dot(color="#ff6b6b", radius=0.08).next_to(contrib_label, LEFT, buff=0.1)
        self.play(
            FadeIn(sum_label), FadeIn(contrib_label),
            FadeIn(sum_dot), FadeIn(contrib_dot),
            run_time=0.4
        )

        # Axis
        axis = Line(
            start=LEFT * 5 + UP * base_y,
            end=RIGHT * 5 + UP * base_y,
            color=GREY, stroke_width=1
        )
        self.play(Create(axis), run_time=0.2)

        running_sum = 1.0  # Start with embedding norm
        for step in range(n_steps):
            x_pos = start_x + step * (bar_width + 0.35)

            # Running sum bar (blue)
            sum_height = min(running_sum * 0.5, max_total_height)
            sum_bar = Rectangle(
                width=bar_width, height=sum_height,
                fill_color="#4ecdc4", fill_opacity=0.5,
                stroke_color="#4ecdc4", stroke_width=1
            )
            sum_bar.move_to(RIGHT * x_pos + UP * (base_y + sum_height / 2))

            # New contribution (red, fixed size)
            contrib_bar = Rectangle(
                width=bar_width, height=fixed_contrib,
                fill_color="#ff6b6b", fill_opacity=0.8,
                stroke_color="#ff6b6b", stroke_width=1
            )
            contrib_bar.move_to(
                RIGHT * x_pos + UP * (base_y + sum_height + fixed_contrib / 2)
            )

            # Layer label
            lbl = Text(f"L{step + 1}", font_size=12, color=GREY_B)
            lbl.next_to(sum_bar, DOWN, buff=0.08)

            # Ratio text
            ratio = fixed_contrib / (sum_height + fixed_contrib)
            ratio_text = Text(
                f"{ratio * 100:.0f}%", font_size=11, color=YELLOW
            )
            ratio_text.next_to(contrib_bar, RIGHT, buff=0.08)

            self.play(
                GrowFromEdge(sum_bar, DOWN),
                GrowFromEdge(contrib_bar, DOWN),
                FadeIn(lbl),
                FadeIn(ratio_text),
                run_time=0.4
            )

            running_sum += 1.0  # Each layer adds ~1.0 to the sum

        # Punchline
        punchline = Text(
            "bounded signal + growing sum = vanishing contribution",
            font_size=18, color="#ff6b6b"
        )
        punchline.to_edge(DOWN, buff=0.3)
        self.play(Write(punchline), run_time=0.6)
        self.wait(2.0)
