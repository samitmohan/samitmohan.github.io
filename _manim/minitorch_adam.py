from manim import *
import numpy as np


class AdamOptimizer(Scene):
    """Visualizes Adam optimizer internal state evolving over 5 steps
    for a single parameter moving toward its optimum."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Adam Optimizer", font_size=30, color=WHITE)
        subtitle = Text(
            "adaptive learning rates via first and second moment estimates",
            font_size=14, color=GREY_B
        )
        title.to_edge(UP, buff=0.3)
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(Write(title), run_time=0.6)
        self.play(FadeIn(subtitle), run_time=0.3)

        # -- Adam hyperparameters --
        lr = 0.1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        # Simulate: f(x) = (x - 3)^2, start at x = 0
        # grad = 2*(x - 3)
        param = 0.0
        m = 0.0
        v = 0.0
        optimum = 3.0

        # -- Update rule display (always visible) --
        rule_text = Text(
            "param -= lr * m_hat / (sqrt(v_hat) + eps)",
            font_size=16, color="#a8e6cf"
        )
        rule_text.to_edge(DOWN, buff=0.35)
        self.play(Write(rule_text), run_time=0.5)

        # -- Number line for parameter --
        nline = NumberLine(
            x_range=[-0.5, 4.0, 0.5],
            length=8,
            include_numbers=False,
            color=GREY_B,
        )
        nline.move_to(DOWN * 1.5)

        # Manual tick labels (avoids LaTeX dependency)
        tick_labels = VGroup()
        for n in [0, 1, 2, 3, 4]:
            lbl = Text(str(n), font_size=14, color=GREY_B)
            lbl.next_to(nline.n2p(n), DOWN, buff=0.15)
            tick_labels.add(lbl)

        # Optimum marker
        opt_dot = Dot(
            nline.n2p(optimum), color="#a8e6cf", radius=0.1
        )
        opt_label = Text("optimum", font_size=12, color="#a8e6cf")
        opt_label.next_to(opt_dot, DOWN, buff=0.12)

        self.play(Create(nline), FadeIn(tick_labels), run_time=0.4)
        self.play(FadeIn(opt_dot), FadeIn(opt_label), run_time=0.3)

        # Parameter dot
        param_dot = Dot(nline.n2p(param), color="#ffd93d", radius=0.12)
        param_label = Text("param", font_size=12, color="#ffd93d")
        param_label.next_to(param_dot, UP, buff=0.12)
        self.play(FadeIn(param_dot), FadeIn(param_label), run_time=0.3)

        # -- State display panel (right side) --
        panel_x = 3.8
        panel_y = 1.2
        row_spacing = 0.38

        state_names = ["step", "param", "grad", "m", "v", "m_hat", "v_hat"]
        state_colors = [
            GREY_B, "#ffd93d", "#ff6b6b", "#4ecdc4",
            "#45b7d1", "#4ecdc4", "#45b7d1"
        ]

        # Static labels on left of panel
        name_labels = []
        val_texts = []
        for i, (name, color) in enumerate(zip(state_names, state_colors)):
            y = panel_y - i * row_spacing
            nlbl = Text(f"{name}:", font_size=13, color=color, weight=BOLD)
            nlbl.move_to(RIGHT * (panel_x - 1.0) + UP * y)
            nlbl.align_to(RIGHT * (panel_x - 0.6), RIGHT)
            name_labels.append(nlbl)

            vtxt = Text("0", font_size=13, color=WHITE)
            vtxt.move_to(RIGHT * (panel_x + 0.3) + UP * y)
            vtxt.align_to(RIGHT * (panel_x - 0.4), LEFT)
            val_texts.append(vtxt)

        # Panel background
        panel_bg = RoundedRectangle(
            width=3.2, height=len(state_names) * row_spacing + 0.4,
            corner_radius=0.12,
            fill_color=WHITE, fill_opacity=0.05,
            stroke_color=GREY_D, stroke_width=1
        )
        panel_bg.move_to(
            RIGHT * panel_x
            + UP * (panel_y - (len(state_names) - 1) * row_spacing / 2)
        )

        self.play(FadeIn(panel_bg), run_time=0.2)
        self.play(
            *[FadeIn(nl) for nl in name_labels],
            *[FadeIn(vt) for vt in val_texts],
            run_time=0.3
        )

        # -- Helper to format floats --
        def fmt(x):
            if abs(x) < 0.0001:
                return "0.0"
            return f"{x:.4f}"

        # -- Helper to update a value text --
        def update_val(idx, new_val_str):
            new_text = Text(new_val_str, font_size=13, color=WHITE)
            new_text.move_to(val_texts[idx].get_center())
            new_text.align_to(val_texts[idx], LEFT)
            return Transform(val_texts[idx], new_text)

        # -- Step header on left --
        step_header = Text("Step 0", font_size=22, color="#96ceb4", weight=BOLD)
        step_header.move_to(LEFT * 3.5 + UP * 1.2)
        self.play(Write(step_header), run_time=0.3)

        # -- Run 5 Adam steps --
        for t in range(1, 6):
            # Compute gradient: f(x) = (x-3)^2, grad = 2*(x-3)
            grad = 2.0 * (param - optimum)

            # Adam updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

            # Update step header
            new_header = Text(
                f"Step {t}", font_size=22, color="#96ceb4", weight=BOLD
            )
            new_header.move_to(LEFT * 3.5 + UP * 1.2)

            # Build all value update animations
            anims = [
                Transform(step_header, new_header),
                update_val(0, str(t)),
                update_val(1, fmt(param)),
                update_val(2, fmt(grad)),
                update_val(3, fmt(m)),
                update_val(4, fmt(v)),
                update_val(5, fmt(m_hat)),
                update_val(6, fmt(v_hat)),
            ]

            # Move param dot
            new_pos = nline.n2p(param)
            anims.append(param_dot.animate.move_to(new_pos))
            anims.append(param_label.animate.next_to(
                Dot(new_pos), UP, buff=0.12
            ))

            # Gradient arrow from param position pointing left (toward optimum)
            grad_arrow_end = nline.n2p(
                param + min(max(grad * 0.15, -0.8), 0.8)
            )
            grad_arrow = Arrow(
                start=nline.n2p(param),
                end=grad_arrow_end,
                color="#ff6b6b", stroke_width=2, buff=0,
                max_tip_length_to_length_ratio=0.15
            )

            self.play(*anims, run_time=0.7)

            # Show gradient direction briefly
            self.play(GrowArrow(grad_arrow), run_time=0.2)
            self.play(FadeOut(grad_arrow), run_time=0.15)

            self.wait(0.2)

        # -- Final note --
        self.play(FadeOut(rule_text), run_time=0.2)
        final_note = Text(
            "Adam adapts per-parameter: large gradients get smaller steps",
            font_size=16, color="#a8e6cf"
        )
        final_note.to_edge(DOWN, buff=0.35)
        self.play(Write(final_note), run_time=0.5)

        self.wait(2.0)
