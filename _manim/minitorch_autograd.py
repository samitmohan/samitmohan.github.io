from manimlib import *
import numpy as np


class AutogradBackward(Scene):
    """Computation graph forward + backward pass for minitorch autograd.

    Visualizes: a = [1,2] -> b = a**2 = [1,4] -> c = sum(b) = 5.0
    Then backward pass with gradient flow and chain rule.
    """

    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # Colors matching the graphviz in the post
        GRAD_NODE = "#d4edda"    # green for requires_grad tensors
        OP_BOX = "#cce5ff"       # blue for operations
        GRAD_TEXT = "#155724"
        OP_TEXT = "#004085"

        # -- Title --
        title = Text("Autograd: Computation Graph", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # -- Node positions --
        a_pos = LEFT * 4.5
        pow_pos = LEFT * 1.5
        b_pos = RIGHT * 1.5
        sum_pos = RIGHT * 3.5
        c_pos = RIGHT * 5.5

        # -- Tensor nodes (ellipses) --
        def make_tensor_node(label_text, pos, subtitle=""):
            ellipse = Ellipse(width=2.0, height=1.0, color=WHITE,
                              fill_color=GRAD_NODE, fill_opacity=0.85,
                              stroke_width=2)
            ellipse.move_to(pos)
            label = Text(label_text, font_size=18, color=GRAD_TEXT)
            label.move_to(pos)
            if subtitle:
                sub = Text(subtitle, font_size=13, color=GRAD_TEXT)
                sub.next_to(ellipse, DOWN, buff=0.15)
                return ellipse, label, sub
            return ellipse, label, None

        # -- Operation nodes (rectangles) --
        def make_op_node(label_text, pos):
            rect = RoundedRectangle(width=1.2, height=0.6, corner_radius=0.1,
                                    color=WHITE, fill_color=OP_BOX,
                                    fill_opacity=0.85, stroke_width=2)
            rect.move_to(pos)
            label = Text(label_text, font_size=16, color=OP_TEXT, weight=BOLD)
            label.move_to(pos)
            return rect, label

        a_ell, a_lbl, a_sub = make_tensor_node("a", a_pos, "[1.0, 2.0]")
        pow_rect, pow_lbl = make_op_node("pow", pow_pos)
        b_ell, b_lbl, b_sub = make_tensor_node("b", b_pos, "[1.0, 4.0]")
        sum_rect, sum_lbl = make_op_node("sum", sum_pos)
        c_ell, c_lbl, c_sub = make_tensor_node("c", c_pos, "5.0")

        # -- Edges --
        def make_arrow(start_mob, end_mob, color=WHITE):
            return Arrow(start_mob.get_right(), end_mob.get_left(),
                         color=color, buff=0.08, stroke_width=2.5,
                         max_tip_length_to_length_ratio=0.15)

        arr1 = make_arrow(a_ell, pow_rect)
        arr2 = make_arrow(pow_rect, b_ell)
        arr3 = make_arrow(b_ell, sum_rect)
        arr4 = make_arrow(sum_rect, c_ell)

        # == Forward Pass ==
        fwd_label = Text("Forward Pass", font_size=24, color=BLUE_C, weight=BOLD)
        fwd_label.to_edge(DOWN, buff=0.4)
        self.play(Write(fwd_label), run_time=0.4)

        # Animate nodes appearing left to right
        for ell, lbl, sub, arr_in, arr_out in [
            (a_ell, a_lbl, a_sub, None, arr1),
            (None, None, None, None, None),      # pow op
            (b_ell, b_lbl, b_sub, arr2, arr3),
            (None, None, None, None, None),       # sum op
            (c_ell, c_lbl, c_sub, arr4, None),
        ]:
            pass  # will animate below in sequence

        # Appear a
        self.play(FadeIn(a_ell), Write(a_lbl), run_time=0.4)
        if a_sub:
            self.play(FadeIn(a_sub), run_time=0.2)
        self.play(ShowCreation(arr1), run_time=0.3)

        # Appear pow
        self.play(FadeIn(pow_rect), Write(pow_lbl), run_time=0.3)
        self.play(ShowCreation(arr2), run_time=0.3)

        # Appear b
        self.play(FadeIn(b_ell), Write(b_lbl), run_time=0.4)
        if b_sub:
            self.play(FadeIn(b_sub), run_time=0.2)
        self.play(ShowCreation(arr3), run_time=0.3)

        # Appear sum
        self.play(FadeIn(sum_rect), Write(sum_lbl), run_time=0.3)
        self.play(ShowCreation(arr4), run_time=0.3)

        # Appear c
        self.play(FadeIn(c_ell), Write(c_lbl), run_time=0.4)
        if c_sub:
            self.play(FadeIn(c_sub), run_time=0.2)

        self.wait(0.5)
        self.play(FadeOut(fwd_label), run_time=0.3)

        # == Backward Pass ==
        bwd_label = Text("Backward Pass", font_size=24, color=RED_C, weight=BOLD)
        bwd_label.to_edge(DOWN, buff=0.4)
        self.play(Write(bwd_label), run_time=0.4)

        # Backward arrows (below the forward arrows)
        def make_back_arrow(start_mob, end_mob):
            start = start_mob.get_left() + DOWN * 0.15
            end = end_mob.get_right() + DOWN * 0.15
            return Arrow(start, end, color=RED_D, buff=0.08,
                         stroke_width=2.5, max_tip_length_to_length_ratio=0.15)

        back4 = make_back_arrow(c_ell, sum_rect)
        back3 = make_back_arrow(sum_rect, b_ell)
        back2 = make_back_arrow(b_ell, pow_rect)
        back1 = make_back_arrow(pow_rect, a_ell)

        # Step 1: Seed c.grad = 1.0
        c_grad = Text("grad = 1.0", font_size=14, color=RED_C, weight=BOLD)
        c_grad.next_to(c_ell, UP, buff=0.2)
        self.play(
            c_ell.animate.set_fill(RED_A, opacity=0.6),
            Write(c_grad),
            run_time=0.4,
        )
        self.play(c_ell.animate.set_fill(GRAD_NODE, opacity=0.85), run_time=0.2)

        # Step 2: Through sum -> b.grad = [1.0, 1.0]
        self.play(ShowCreation(back4), run_time=0.3)

        sum_grad_note = Text("broadcast 1.0", font_size=12, color=ORANGE)
        sum_grad_note.next_to(back3, DOWN, buff=0.1)

        self.play(ShowCreation(back3), FadeIn(sum_grad_note), run_time=0.3)

        b_grad = Text("grad = [1.0, 1.0]", font_size=14, color=RED_C, weight=BOLD)
        b_grad.next_to(b_ell, UP, buff=0.2)
        self.play(
            b_ell.animate.set_fill(RED_A, opacity=0.6),
            Write(b_grad),
            run_time=0.4,
        )
        self.play(b_ell.animate.set_fill(GRAD_NODE, opacity=0.85), run_time=0.2)

        # Step 3: Through pow -> a.grad = 2*a*b.grad = [2.0, 4.0]
        self.play(ShowCreation(back2), run_time=0.3)

        pow_grad_note = Text("2 * a * grad", font_size=12, color=ORANGE)
        pow_grad_note.next_to(back1, DOWN, buff=0.1)

        self.play(ShowCreation(back1), FadeIn(pow_grad_note), run_time=0.3)

        a_grad = Text("grad = [2.0, 4.0]", font_size=14, color=RED_C, weight=BOLD)
        a_grad.next_to(a_ell, UP, buff=0.2)
        self.play(
            a_ell.animate.set_fill(RED_A, opacity=0.6),
            Write(a_grad),
            run_time=0.4,
        )
        self.play(a_ell.animate.set_fill(GRAD_NODE, opacity=0.85), run_time=0.2)

        self.wait(0.3)
        self.play(FadeOut(bwd_label), FadeOut(sum_grad_note), FadeOut(pow_grad_note),
                  run_time=0.3)

        # Chain rule summary
        chain = Text(
            "Chain Rule: grad_a = 2*a * 1.0 = [2.0, 4.0]",
            font_size=20, color=WHITE, weight=BOLD,
        )
        chain.to_edge(DOWN, buff=0.35)
        self.play(Write(chain), run_time=0.6)
        self.wait(1.5)
