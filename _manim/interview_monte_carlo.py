from manim import *
import numpy as np


class MonteCarloPi(Scene):
    """Estimate Pi by throwing darts into a unit square with inscribed quarter circle."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Estimating Pi with Monte Carlo", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # Drawing area - sized to fit within the frame
        sq_side = 3.5
        origin = LEFT * 3.0 + DOWN * 2.0

        # Draw the unit square
        square = Square(
            side_length=sq_side,
            fill_color="#2a2a4e", fill_opacity=0.3,
            stroke_color=GREY, stroke_width=2,
        )
        square.move_to(origin + RIGHT * sq_side / 2 + UP * sq_side / 2)

        # Quarter circle arc (radius = sq_side, centered at origin corner)
        quarter_arc = Arc(
            radius=sq_side,
            start_angle=0,
            angle=PI / 2,
            color="#4ecdc4",
            stroke_width=2,
        )
        quarter_arc.move_arc_center_to(origin)

        # Axis labels
        x_label = Text("1", font_size=14, color=GREY_B)
        x_label.next_to(origin + RIGHT * sq_side, DOWN, buff=0.1)
        y_label = Text("1", font_size=14, color=GREY_B)
        y_label.next_to(origin + UP * sq_side, LEFT, buff=0.1)
        zero_label = Text("0", font_size=14, color=GREY_B)
        zero_label.next_to(origin, DOWN + LEFT, buff=0.05)

        self.play(
            FadeIn(square), Create(quarter_arc),
            FadeIn(x_label), FadeIn(y_label), FadeIn(zero_label),
            run_time=0.5,
        )

        # Formula on the right side
        formula_box = RoundedRectangle(
            width=3.0, height=1.6,
            corner_radius=0.1,
            fill_color="#2a2a4e", fill_opacity=0.4,
            stroke_color="#45b7d1", stroke_width=2,
        )
        formula_box.move_to(RIGHT * 4.0 + UP * 1.8)

        formula_title = Text("Method", font_size=14, color="#45b7d1")
        formula_title.next_to(formula_box, UP, buff=0.08)

        f1 = Text("Area of quarter circle = pi/4", font_size=12, color=WHITE)
        f2 = Text("Area of square = 1", font_size=12, color=WHITE)
        f3 = Text("ratio = inside / total", font_size=12, color=WHITE)
        f4 = Text("pi = 4 * ratio", font_size=15, color="#ffd93d")

        formula_lines = VGroup(f1, f2, f3, f4)
        formula_lines.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        formula_lines.move_to(formula_box.get_center())

        self.play(
            FadeIn(formula_box), FadeIn(formula_title),
            FadeIn(formula_lines),
            run_time=0.5,
        )

        # Pi estimate display
        estimate_label = Text("Pi estimate:", font_size=16, color=GREY_B)
        estimate_label.move_to(RIGHT * 4.0 + DOWN * 0.0)

        pi_text = Text("--", font_size=26, color="#ffd93d")
        pi_text.next_to(estimate_label, DOWN, buff=0.1)

        count_text = Text("0 / 0 dots", font_size=13, color=GREY_B)
        count_text.next_to(pi_text, DOWN, buff=0.12)

        self.play(FadeIn(estimate_label), FadeIn(pi_text), FadeIn(count_text), run_time=0.3)

        # Generate random points
        np.random.seed(42)
        n_total = 200
        xs = np.random.uniform(0, 1, n_total)
        ys = np.random.uniform(0, 1, n_total)
        inside = xs ** 2 + ys ** 2 <= 1.0

        # Animate in batches
        batch_size = 10
        n_batches = n_total // batch_size
        inside_count = 0

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size

            dots = VGroup()
            for i in range(start, end):
                px = origin[0] + xs[i] * sq_side
                py = origin[1] + ys[i] * sq_side

                if inside[i]:
                    color = "#a8e6cf"
                    inside_count += 1
                else:
                    color = "#ff6b6b"

                dot = Dot(
                    point=RIGHT * px + UP * py,
                    radius=0.04,
                    color=color,
                    fill_opacity=0.85,
                )
                dots.add(dot)

            total_so_far = end
            pi_est = 4.0 * inside_count / total_so_far

            new_pi = Text(f"{pi_est:.5f}", font_size=26, color="#ffd93d")
            new_pi.move_to(pi_text.get_center())

            new_count = Text(
                f"{inside_count} / {total_so_far} dots",
                font_size=13, color=GREY_B
            )
            new_count.move_to(count_text.get_center())

            self.play(
                FadeIn(dots),
                Transform(pi_text, new_pi),
                Transform(count_text, new_count),
                run_time=0.3,
            )

        # Final comparison
        actual_pi = Text("Actual: 3.14159...", font_size=15, color="#4ecdc4")
        actual_pi.next_to(count_text, DOWN, buff=0.25)

        error = abs(4.0 * inside_count / n_total - np.pi)
        error_text = Text(f"Error: {error:.5f}", font_size=13, color=GREY_B)
        error_text.next_to(actual_pi, DOWN, buff=0.08)

        self.play(FadeIn(actual_pi), FadeIn(error_text), run_time=0.4)

        self.wait(2.0)
