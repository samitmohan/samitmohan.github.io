from manim import *
import numpy as np


class DepthAttention(Scene):
    """Visualizes the depth attention mechanism: query scoring checkpoints."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Depth Attention Mechanism", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # Checkpoints on the left
        n_checkpoints = 5
        cp_labels = ["h_0\n(embed)", "h_8'", "h_16'", "h_24'", "h_32"]
        cp_colors = ["#ffd93d", "#4ecdc4", "#45b7d1", "#96ceb4", "#ff6b6b"]

        checkpoints = VGroup()
        cp_rects = []
        for i in range(n_checkpoints):
            rect = RoundedRectangle(
                width=1.6, height=0.6,
                corner_radius=0.1,
                fill_color=cp_colors[i], fill_opacity=0.3,
                stroke_color=cp_colors[i], stroke_width=2
            )
            label = Text(cp_labels[i], font_size=14, color=WHITE)
            label.move_to(rect.get_center())
            cp = VGroup(rect, label)
            cp_rects.append(rect)
            checkpoints.add(cp)

        checkpoints.arrange(DOWN, buff=0.25)
        checkpoints.move_to(LEFT * 4.5)

        self.play(FadeIn(checkpoints), run_time=0.6)

        # RMSNorm arrow
        norm_arrow = Arrow(
            start=LEFT * 3.4, end=LEFT * 2.0,
            color=GREY_B, stroke_width=2
        )
        norm_label = Text("RMSNorm", font_size=14, color=GREY_B)
        norm_label.next_to(norm_arrow, UP, buff=0.05)
        self.play(GrowArrow(norm_arrow), FadeIn(norm_label), run_time=0.3)

        # Query vector (learned parameter)
        query_box = RoundedRectangle(
            width=1.8, height=0.7,
            corner_radius=0.1,
            fill_color="#9b59b6", fill_opacity=0.3,
            stroke_color="#9b59b6", stroke_width=2
        )
        query_box.move_to(RIGHT * 1 + UP * 2.5)
        query_label = Text("w_l (query)\nlearned param", font_size=13, color=WHITE)
        query_label.move_to(query_box.get_center())

        self.play(FadeIn(query_box), FadeIn(query_label), run_time=0.4)

        # Dot product arrows from query to each checkpoint
        # Score box in the middle
        score_title = Text("scores = w . v / sqrt(d)", font_size=15, color=YELLOW)
        score_title.move_to(RIGHT * 1 + UP * 1.3)
        self.play(FadeIn(score_title), run_time=0.3)

        # Show scores
        np.random.seed(42)
        raw_scores = np.array([0.1, 2.5, 0.8, 0.3, 1.2])
        softmax_scores = np.exp(raw_scores) / np.exp(raw_scores).sum()

        score_bars = VGroup()
        score_labels = VGroup()
        for i in range(n_checkpoints):
            bar_width = softmax_scores[i] * 4.0
            bar = Rectangle(
                width=bar_width, height=0.35,
                fill_color=cp_colors[i], fill_opacity=0.7,
                stroke_width=0
            )
            bar.move_to(RIGHT * (1 + bar_width / 2) + DOWN * (0.2 + i * 0.55))

            pct = Text(
                f"{softmax_scores[i] * 100:.1f}%",
                font_size=13, color=WHITE
            )
            pct.next_to(bar, RIGHT, buff=0.1)

            name = Text(cp_labels[i].split("\n")[0], font_size=12, color=GREY_B)
            name.next_to(bar, LEFT, buff=0.1)

            score_bars.add(bar)
            score_labels.add(VGroup(pct, name))

        softmax_label = Text("softmax", font_size=16, color=YELLOW)
        softmax_label.move_to(RIGHT * 1 + UP * 0.4)

        self.play(FadeIn(softmax_label), run_time=0.2)
        for i in range(n_checkpoints):
            self.play(
                GrowFromEdge(score_bars[i], LEFT),
                FadeIn(score_labels[i]),
                run_time=0.25
            )

        # Output
        output_label = Text(
            "output = weighted sum of checkpoints",
            font_size=16, color="#a8e6cf"
        )
        output_label.to_edge(DOWN, buff=0.5)

        # Highlight the winner
        highlight = SurroundingRectangle(
            score_bars[1], color=YELLOW, buff=0.05, stroke_width=2
        )
        winner_text = Text("h_8' dominates here", font_size=14, color=YELLOW)
        winner_text.next_to(highlight, RIGHT, buff=0.3)

        self.play(
            Create(highlight),
            FadeIn(winner_text),
            Write(output_label),
            run_time=0.5
        )
        self.wait(2.0)
