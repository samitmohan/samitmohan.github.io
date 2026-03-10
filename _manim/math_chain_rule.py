from manim import *
import numpy as np


class ChainRule(Scene):
    """Chain rule visualization: f(g(x)) derivative step by step.

    Shows composed function, symbolic derivative, and numerical example.
    """

    def construct(self):
        self.camera.background_color = WHITE

        title = Text("The Chain Rule", font_size=40, color=BLACK, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)

        # -- Setup: f(g(x)) where g(x) = 3x+1, f(u) = u^2 --
        func_def = VGroup(
            Text("Let g(x) = 3x + 1", font_size=26, color=BLUE_D),
            Text("Let f(u) = u^2", font_size=26, color=RED_D),
            Text("Composed: h(x) = f(g(x)) = (3x+1)^2", font_size=26, color=BLACK),
        )
        func_def.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        func_def.next_to(title, DOWN, buff=0.6)

        for fd in func_def:
            self.play(Write(fd), run_time=0.4)
        self.wait(0.3)

        # -- Visual: box diagram --
        # x -> [g] -> g(x) -> [f] -> f(g(x))
        box_group = VGroup()

        x_label = Text("x", font_size=28, color=BLACK)
        x_label.move_to(LEFT * 5)

        g_box = RoundedRectangle(
            corner_radius=0.15, width=1.5, height=0.8,
            color=BLUE_D, fill_color=BLUE_A, fill_opacity=0.3,
        )
        g_box.move_to(LEFT * 3)
        g_text = Text("g", font_size=28, color=BLUE_D)
        g_text.move_to(g_box)

        mid_label = Text("g(x)", font_size=22, color=GREY_D)
        mid_label.move_to(LEFT * 1)

        f_box = RoundedRectangle(
            corner_radius=0.15, width=1.5, height=0.8,
            color=RED_D, fill_color=RED_A, fill_opacity=0.3,
        )
        f_box.move_to(RIGHT * 1)
        f_text = Text("f", font_size=28, color=RED_D)
        f_text.move_to(f_box)

        out_label = Text("f(g(x))", font_size=22, color=BLACK)
        out_label.move_to(RIGHT * 3.2)

        arrows = VGroup(
            Arrow(x_label.get_right(), g_box.get_left(), color=BLACK, buff=0.1, stroke_width=2),
            Arrow(g_box.get_right(), mid_label.get_left(), color=BLACK, buff=0.1, stroke_width=2),
            Arrow(mid_label.get_right(), f_box.get_left(), color=BLACK, buff=0.1, stroke_width=2),
            Arrow(f_box.get_right(), out_label.get_left(), color=BLACK, buff=0.1, stroke_width=2),
        )

        diagram = VGroup(x_label, g_box, g_text, mid_label, f_box, f_text, out_label, arrows)
        diagram.shift(DOWN * 0.5)

        self.play(
            *[Create(a) for a in arrows],
            Create(g_box), Write(g_text),
            Create(f_box), Write(f_text),
            Write(x_label), Write(mid_label), Write(out_label),
            run_time=1,
        )
        self.wait(0.5)

        # -- Move everything up and show derivative steps below --
        self.play(
            VGroup(title, func_def, diagram).animate.shift(UP * 1.5),
            run_time=0.5,
        )

        # Derivative step by step
        deriv_steps = [
            ("Step 1: Identify outer and inner", GREY_D),
            ("  dh/dx = df/dg * dg/dx", BLACK),
            ("", BLACK),
            ("Step 2: Differentiate outer f(u)=u^2", RED_D),
            ("  df/du = 2u = 2*g(x) = 2(3x+1)", RED_D),
            ("", BLACK),
            ("Step 3: Differentiate inner g(x)=3x+1", BLUE_D),
            ("  dg/dx = 3", BLUE_D),
            ("", BLACK),
            ("Step 4: Multiply (Chain Rule!)", BLACK),
            ("  dh/dx = 2(3x+1) * 3 = 6(3x+1)", BLACK),
        ]

        step_mobs = VGroup()
        for text, color in deriv_steps:
            if text:
                t = Text(text, font_size=20, color=color)
            else:
                t = Text(" ", font_size=10)
            step_mobs.add(t)

        step_mobs.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        step_mobs.to_edge(LEFT, buff=0.5)
        step_mobs.shift(DOWN * 1.8)

        for s in step_mobs:
            self.play(Write(s), run_time=0.25)

        self.wait(0.3)

        # -- Numerical example on the right --
        num_title = Text("Numerical check at x=2:", font_size=22, color=PURPLE_D, weight=BOLD)
        num_steps = VGroup(
            Text("g(2) = 3(2)+1 = 7", font_size=18, color=BLUE_D),
            Text("f(g(2)) = 7^2 = 49", font_size=18, color=RED_D),
            Text("", font_size=8),
            Text("df/dg = 2(7) = 14", font_size=18, color=RED_D),
            Text("dg/dx = 3", font_size=18, color=BLUE_D),
            Text("", font_size=8),
            Text("dh/dx = 14 * 3 = 42", font_size=20, color=BLACK, weight=BOLD),
        )
        num_group = VGroup(num_title, num_steps)
        num_steps.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        num_group.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        num_group.to_edge(RIGHT, buff=0.5)
        num_group.shift(DOWN * 1.8)

        # Surrounding box
        num_box = SurroundingRectangle(num_group, color=PURPLE_D, buff=0.2,
                                        corner_radius=0.1, stroke_width=1.5)

        self.play(Write(num_title), Create(num_box), run_time=0.4)
        for ns in num_steps:
            self.play(Write(ns), run_time=0.2)

        # Final highlight
        result_box = SurroundingRectangle(step_mobs[-1], color=GREEN_D, buff=0.1,
                                           corner_radius=0.05, stroke_width=2)
        self.play(Create(result_box), run_time=0.3)
        self.wait(1)
