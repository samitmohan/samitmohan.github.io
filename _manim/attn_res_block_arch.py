from manim import *
import numpy as np


class BlockAttnRes(Scene):
    """Architecture diagram of Block Attention Residuals."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Block Attention Residuals", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # Architecture: 4 blocks of 4 layers (simplified from 8x8 for visual clarity)
        n_blocks = 4
        layers_per_block = 4
        block_colors = ["#4ecdc4", "#45b7d1", "#96ceb4", "#ffd93d"]
        block_width = 1.6
        block_height = 2.2
        block_spacing = 0.4

        total_width = n_blocks * block_width + (n_blocks - 1) * block_spacing
        start_x = -total_width / 2 + block_width / 2

        blocks = VGroup()
        block_rects = []
        checkpoint_dots = []
        layer_groups = VGroup()

        # Initial embedding checkpoint
        emb_dot = Dot(
            point=LEFT * (total_width / 2 + 0.8) + DOWN * 0.3,
            color=YELLOW, radius=0.12
        )
        emb_label = Text("h_0", font_size=16, color=YELLOW)
        emb_label.next_to(emb_dot, DOWN, buff=0.1)
        checkpoint_dots.append(emb_dot)

        self.play(FadeIn(emb_dot), FadeIn(emb_label), run_time=0.3)

        # Draw blocks
        for b in range(n_blocks):
            x = start_x + b * (block_width + block_spacing)
            color = block_colors[b]

            # Block rectangle
            block_rect = RoundedRectangle(
                width=block_width, height=block_height,
                corner_radius=0.15,
                fill_color=color, fill_opacity=0.15,
                stroke_color=color, stroke_width=2
            )
            block_rect.move_to(RIGHT * x + DOWN * 0.3)
            block_rects.append(block_rect)

            # Layer labels inside block
            inner_layers = VGroup()
            for l in range(layers_per_block):
                layer_num = b * layers_per_block + l + 1
                layer_rect = Rectangle(
                    width=block_width - 0.3, height=0.35,
                    fill_color=color, fill_opacity=0.3,
                    stroke_color=color, stroke_width=1
                )
                layer_text = Text(
                    f"Layer {layer_num}", font_size=13, color=WHITE
                )
                layer_text.move_to(layer_rect.get_center())
                inner_layers.add(VGroup(layer_rect, layer_text))

            inner_layers.arrange(DOWN, buff=0.08)
            inner_layers.move_to(block_rect.get_center())
            layer_groups.add(inner_layers)

            # Block label
            block_label = Text(
                f"Block {b + 1}", font_size=16, color=color, weight=BOLD
            )
            block_label.next_to(block_rect, UP, buff=0.1)

            # "Standard residuals" annotation inside
            res_label = Text(
                "x = x + F(x)", font_size=11, color=GREY_B
            )
            res_label.next_to(block_rect, DOWN, buff=0.08)

            blocks.add(VGroup(block_rect, inner_layers, block_label, res_label))

            # Checkpoint dot after each block
            cp_dot = Dot(
                point=RIGHT * (x + block_width / 2 + block_spacing / 2) + DOWN * 0.3,
                color=YELLOW, radius=0.12
            )
            cp_label = Text(
                f"h_{(b + 1) * layers_per_block}'",
                font_size=14, color=YELLOW
            )
            cp_label.next_to(cp_dot, DOWN, buff=0.1)
            checkpoint_dots.append(cp_dot)

        # Animate blocks appearing
        for b_group in blocks:
            self.play(FadeIn(b_group), run_time=0.4)

        self.wait(0.3)

        # Show checkpoint dots
        for i, dot in enumerate(checkpoint_dots):
            self.play(FadeIn(dot), run_time=0.15)

        self.wait(0.3)

        # Show depth attention at a boundary
        # Highlight: after Block 3, depth attention over {h_0, h_4', h_8', h_12}
        da_label = Text(
            "Depth Attention", font_size=20, color="#ff6b6b", weight=BOLD
        )
        da_label.to_edge(DOWN, buff=1.2)

        da_box = RoundedRectangle(
            width=2.2, height=0.6,
            corner_radius=0.1,
            fill_color="#ff6b6b", fill_opacity=0.2,
            stroke_color="#ff6b6b", stroke_width=2
        )
        da_box.move_to(da_label.get_center())

        self.play(
            FadeIn(da_box),
            Write(da_label),
            run_time=0.5
        )

        # Draw arrows from checkpoints to depth attention
        arrows = VGroup()
        for i, dot in enumerate(checkpoint_dots[:4]):  # h_0 through h_12
            arrow = Arrow(
                start=dot.get_center(),
                end=da_box.get_top(),
                color="#ff6b6b",
                stroke_width=2,
                buff=0.15,
                max_tip_length_to_length_ratio=0.15
            )
            arrows.add(arrow)

        self.play(*[GrowArrow(a) for a in arrows], run_time=0.6)

        # Show the formula
        formula = Text(
            "h' = softmax(w . checkpoints) x checkpoints",
            font_size=16, color=WHITE
        )
        formula.next_to(da_box, DOWN, buff=0.2)
        self.play(FadeIn(formula), run_time=0.4)

        # Cost annotation
        cost = Text(
            "Cost: ~2% overhead", font_size=18, color="#a8e6cf"
        )
        cost.next_to(formula, DOWN, buff=0.15)
        self.play(Write(cost), run_time=0.4)

        self.wait(2.0)
