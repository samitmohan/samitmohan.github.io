from manim import *
import numpy as np


class NormGrowth(Scene):
    """Shows hidden state norm growing with depth in standard residual connections."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        # Title
        title = Text("Hidden State Norm vs Depth", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.4)
        code = Text("x = x + Layer(x)    # repeated 64 times", font_size=18, color=YELLOW)
        code.next_to(title, DOWN, buff=0.2)

        self.play(Write(title), run_time=0.6)
        self.play(FadeIn(code), run_time=0.4)
        self.wait(0.3)

        # Data from actual PyTorch run (torch.manual_seed(42), d=512, nn.Linear)
        layers = [0, 8, 16, 24, 32, 40, 48, 56, 64]
        norms = [71.7, 227.1, 724.9, 2312, 7241, 22891, 72305, 228520, 730954]

        log_norms = [np.log10(n) for n in norms]
        max_log = max(log_norms)

        # Bar dimensions
        n_bars = len(layers)
        bar_width = 0.65
        total_width = n_bars * bar_width + (n_bars - 1) * 0.12
        start_x = -total_width / 2 + bar_width / 2
        max_height = 4.0
        base_y = -2.0

        # Baseline axis
        axis_line = Line(
            start=LEFT * (total_width / 2 + 0.3) + UP * base_y,
            end=RIGHT * (total_width / 2 + 0.3) + UP * base_y,
            color=GREY, stroke_width=1.5
        )
        self.play(Create(axis_line), run_time=0.3)

        # Animate bars one by one
        color_low = ManimColor("#4ecdc4")
        color_high = ManimColor("#ff6b6b")
        bars = VGroup()
        for i, (layer, norm, log_norm) in enumerate(zip(layers, norms, log_norms)):
            height = (log_norm / max_log) * max_height
            t = log_norm / max_log  # 0 to 1 for color

            bar = Rectangle(
                width=bar_width,
                height=height,
                fill_color=interpolate_color(color_low, color_high, t),
                fill_opacity=0.85,
                stroke_color=WHITE,
                stroke_width=1
            )
            x_pos = start_x + i * (bar_width + 0.12)
            bar.move_to(RIGHT * x_pos + UP * (base_y + height / 2))
            bars.add(bar)

            # Layer label below bar
            label = Text(f"L{layer}", font_size=14, color=GREY_B)
            label.next_to(bar, DOWN, buff=0.1)

            # Norm value above bar
            if norm >= 1000:
                val_str = f"{norm / 1000:.0f}K"
            else:
                val_str = f"{norm:.0f}"
            val = Text(val_str, font_size=13, color=YELLOW)
            val.next_to(bar, UP, buff=0.08)

            self.play(
                GrowFromEdge(bar, DOWN),
                FadeIn(label),
                FadeIn(val),
                run_time=0.35
            )

        self.wait(0.4)

        # Highlight the problem
        brace = Brace(bars, DOWN, color="#ff6b6b")
        brace_text = Text(
            "10,000x growth across 64 layers",
            font_size=20, color="#ff6b6b"
        )
        brace_text.next_to(brace, DOWN, buff=0.15)

        self.play(
            GrowFromCenter(brace),
            Write(brace_text),
            run_time=0.6
        )
        self.wait(2.0)
