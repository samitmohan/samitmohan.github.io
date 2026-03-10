from manimlib import *
import numpy as np


class ScalingCurves(Scene):
    def construct(self):
        # White/light background
        self.camera.background_rgba = [0xf5/255, 0xf5/255, 0xf5/255, 1]

        # Create axes (log-log style)
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 5, 1],
            width=8,
            height=5,
            axis_config={
                "color": "#333333",
                "stroke_width": 2,
                "include_ticks": False,
            },
        ).shift(DOWN * 0.3 + LEFT * 0.3)

        x_label = Text("Compute (FLOPs)", font_size=22, color="#333333").next_to(
            axes.x_axis, DOWN, buff=0.3
        )
        y_label = Text("Loss", font_size=22, color="#333333").next_to(
            axes.y_axis, LEFT, buff=0.3
        ).rotate(PI / 2)

        # Log-log tick marks
        x_ticks = VGroup()
        for i in range(1, 6):
            tick = axes.x_axis.get_tick(i, size=0.1)
            tick.set_color("#333333")
            label = Text(f"10^{i+18}", font_size=12, color="#666666").next_to(
                tick, DOWN, buff=0.1
            )
            x_ticks.add(tick, label)

        self.play(
            ShowCreation(axes),
            Write(x_label),
            Write(y_label),
            FadeIn(x_ticks),
            run_time=1.5,
        )

        # Curve definitions using smooth functions
        # Red: Scale parameters only - drops then flattens early
        def red_func(x):
            return 4.2 * np.exp(-0.6 * x) + 1.8

        # Blue: Scale data only - drops then flattens at a different point
        def blue_func(x):
            return 3.8 * np.exp(-0.5 * x) + 1.5

        # Green: Scale both (Chinchilla) - keeps dropping, lower loss
        def green_func(x):
            return 4.5 * np.exp(-0.45 * x) + 0.3

        red_curve = axes.get_graph(
            red_func,
            x_range=[0.3, 5.8],
            color=RED,
            stroke_width=3,
        )
        blue_curve = axes.get_graph(
            blue_func,
            x_range=[0.3, 5.8],
            color=BLUE,
            stroke_width=3,
        )
        green_curve = axes.get_graph(
            green_func,
            x_range=[0.3, 5.8],
            color=GREEN_D,
            stroke_width=3,
        )

        # Labels for curves
        red_label = Text("Scale parameters only", font_size=16, color=RED).next_to(
            axes.c2p(5.8, red_func(5.8)), RIGHT, buff=0.15
        )
        blue_label = Text("Scale data only", font_size=16, color=BLUE).next_to(
            axes.c2p(5.8, blue_func(5.8)), RIGHT, buff=0.15
        )
        green_label = Text(
            "Scale both (Chinchilla)", font_size=16, color=GREEN_D
        ).next_to(axes.c2p(5.8, green_func(5.8)), RIGHT, buff=0.15)

        # Animate curves one by one
        self.play(ShowCreation(red_curve), FadeIn(red_label), run_time=1.5)
        self.play(ShowCreation(blue_curve), FadeIn(blue_label), run_time=1.5)
        self.play(ShowCreation(green_curve), FadeIn(green_label), run_time=1.5)

        # Add labeled dots
        gpt3_x = 4.0
        gpt3_point = axes.c2p(gpt3_x, red_func(gpt3_x))
        gpt3_dot = Dot(gpt3_point, color=RED, radius=0.08)
        gpt3_label = Text(
            "GPT-3\n(overtrained on params,\nundertrained on data)",
            font_size=13,
            color=RED_D,
        ).next_to(gpt3_dot, UP + LEFT, buff=0.15)

        chinchilla_point = axes.c2p(gpt3_x, green_func(gpt3_x))
        chinchilla_dot = Dot(chinchilla_point, color=GREEN_D, radius=0.08)
        chinchilla_label = Text(
            "Chinchilla\n(balanced)",
            font_size=13,
            color=GREEN_D,
        ).next_to(chinchilla_dot, DOWN + LEFT, buff=0.15)

        # Dashed line connecting the two dots to show the gap
        connector = DashedLine(
            gpt3_point, chinchilla_point, color="#888888", stroke_width=1.5
        )

        self.play(
            FadeIn(gpt3_dot),
            Write(gpt3_label),
            FadeIn(chinchilla_dot),
            Write(chinchilla_label),
            ShowCreation(connector),
            run_time=1.5,
        )

        # Bottom text
        bottom_text = Text(
            "Same compute budget, different allocation = different loss",
            font_size=20,
            color="#333333",
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(bottom_text), run_time=1.0)
        self.wait(1.5)
