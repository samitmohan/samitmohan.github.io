from manim import *
import numpy as np


class BlockAttnRes(Scene):
    """Architecture diagram of Block Attention Residuals."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Block Attention Residuals", font_size=30, color=WHITE)
        subtitle = Text(
            "x = x + F(x) within blocks  |  depth attention at boundaries",
            font_size=14, color=GREY_B
        )
        title.to_edge(UP, buff=0.3)
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(Write(title), run_time=0.6)
        self.play(FadeIn(subtitle), run_time=0.3)

        # Architecture: 4 blocks of 4 layers (simplified from 8x8)
        n_blocks = 4
        layers_per_block = 4
        block_colors = ["#4ecdc4", "#45b7d1", "#96ceb4", "#ffd93d"]
        block_width = 1.5
        block_height = 2.0
        block_spacing = 0.5
        block_center_y = 0.7

        total_width = n_blocks * block_width + (n_blocks - 1) * block_spacing
        start_x = -total_width / 2 + block_width / 2

        blocks = VGroup()
        checkpoint_dots = []
        checkpoint_labels = []

        # Initial embedding checkpoint
        emb_x = start_x - block_width / 2 - 0.4
        emb_dot = Dot(
            point=RIGHT * emb_x + UP * block_center_y,
            color=YELLOW, radius=0.1
        )
        emb_label = Text("h_0", font_size=13, color=YELLOW)
        emb_label.next_to(emb_dot, UP, buff=0.06)
        checkpoint_dots.append(emb_dot)
        checkpoint_labels.append(emb_label)

        self.play(FadeIn(emb_dot), FadeIn(emb_label), run_time=0.3)

        # Draw blocks
        for b in range(n_blocks):
            x = start_x + b * (block_width + block_spacing)
            color = block_colors[b]

            block_rect = RoundedRectangle(
                width=block_width, height=block_height,
                corner_radius=0.15,
                fill_color=color, fill_opacity=0.15,
                stroke_color=color, stroke_width=2
            )
            block_rect.move_to(RIGHT * x + UP * block_center_y)

            # Layer labels inside block
            inner_layers = VGroup()
            for l in range(layers_per_block):
                layer_num = b * layers_per_block + l + 1
                layer_rect = Rectangle(
                    width=block_width - 0.3, height=0.3,
                    fill_color=color, fill_opacity=0.3,
                    stroke_color=color, stroke_width=1
                )
                layer_text = Text(f"L{layer_num}", font_size=12, color=WHITE)
                layer_text.move_to(layer_rect.get_center())
                inner_layers.add(VGroup(layer_rect, layer_text))

            inner_layers.arrange(DOWN, buff=0.06)
            inner_layers.move_to(block_rect.get_center())

            block_label = Text(
                f"Block {b + 1}", font_size=14, color=color, weight=BOLD
            )
            block_label.next_to(block_rect, UP, buff=0.06)

            blocks.add(VGroup(block_rect, inner_layers, block_label))

            # Checkpoint dot after each block
            if b < n_blocks - 1:
                cp_x = x + block_width / 2 + block_spacing / 2
            else:
                cp_x = x + block_width / 2 + 0.4

            cp_dot = Dot(
                point=RIGHT * cp_x + UP * block_center_y,
                color=YELLOW, radius=0.1
            )
            cp_label = Text(
                f"h_{(b + 1) * layers_per_block}'",
                font_size=11, color=YELLOW
            )
            cp_label.next_to(cp_dot, UP, buff=0.06)
            checkpoint_dots.append(cp_dot)
            checkpoint_labels.append(cp_label)

        # Animate blocks appearing
        for b_group in blocks:
            self.play(FadeIn(b_group), run_time=0.35)

        # Show checkpoint dots and labels
        for dot, label in zip(checkpoint_dots[1:], checkpoint_labels[1:]):
            self.play(FadeIn(dot), FadeIn(label), run_time=0.15)

        self.wait(0.3)

        # Depth attention box below blocks (clear space, no labels in the way)
        da_label = Text(
            "Depth Attention", font_size=20, color="#ff6b6b", weight=BOLD
        )
        da_label.move_to(DOWN * 2.0)

        da_box = RoundedRectangle(
            width=2.8, height=0.6,
            corner_radius=0.1,
            fill_color="#ff6b6b", fill_opacity=0.2,
            stroke_color="#ff6b6b", stroke_width=2
        )
        da_box.move_to(da_label.get_center())

        self.play(FadeIn(da_box), Write(da_label), run_time=0.5)

        # Arrows from first 4 checkpoints to depth attention box
        arrows = VGroup()
        for dot in checkpoint_dots[:4]:
            arrow = Arrow(
                start=dot.get_center(),
                end=da_box.get_top(),
                color="#ff6b6b",
                stroke_width=1.5,
                buff=0.15,
                max_tip_length_to_length_ratio=0.1
            )
            arrows.add(arrow)

        self.play(*[GrowArrow(a) for a in arrows], run_time=0.6)

        # Formula and cost
        formula = Text(
            "h' = sum(softmax(w . v_i) * v_i)",
            font_size=14, color=WHITE
        )
        formula.next_to(da_box, DOWN, buff=0.2)
        self.play(FadeIn(formula), run_time=0.3)

        cost = Text(
            "~2% overhead  |  <0.2% params", font_size=16, color="#a8e6cf"
        )
        cost.next_to(formula, DOWN, buff=0.12)
        self.play(Write(cost), run_time=0.3)

        self.wait(2.0)
