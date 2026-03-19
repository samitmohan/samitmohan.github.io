from manim import *


class SkipConnection(Scene):
    """Flow diagram of a BasicBlock with skip connection."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("The Skip Connection (BasicBlock)", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # Layout constants
        main_x = 0
        box_w = 2.0
        box_h = 0.5
        spacing = 0.7

        # Main path boxes (top to bottom)
        stages = [
            ("Input x", "#a8e6cf"),
            ("Conv -> BN", "#4ecdc4"),
            ("ReLU", "#45b7d1"),
            ("Conv -> BN", "#4ecdc4"),
        ]

        boxes = []
        labels = []
        y_start = 1.8

        for i, (text, color) in enumerate(stages):
            y = y_start - i * (box_h + spacing)
            box = RoundedRectangle(
                width=box_w, height=box_h,
                corner_radius=0.1,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2
            )
            box.move_to(RIGHT * main_x + UP * y)
            label = Text(text, font_size=16, color=WHITE)
            label.move_to(box.get_center())
            boxes.append(box)
            labels.append(label)

        # Addition node
        add_y = y_start - len(stages) * (box_h + spacing)
        add_circle = Circle(
            radius=0.3,
            fill_color="#96ceb4", fill_opacity=0.3,
            stroke_color="#96ceb4", stroke_width=2
        )
        add_circle.move_to(RIGHT * main_x + UP * add_y)
        add_text = Text("+", font_size=28, color="#96ceb4", weight=BOLD)
        add_text.move_to(add_circle.get_center())

        # ReLU after addition
        relu_y = add_y - (box_h + spacing)
        relu_box = RoundedRectangle(
            width=box_w, height=box_h,
            corner_radius=0.1,
            fill_color="#45b7d1", fill_opacity=0.2,
            stroke_color="#45b7d1", stroke_width=2
        )
        relu_box.move_to(RIGHT * main_x + UP * relu_y)
        relu_label = Text("ReLU", font_size=16, color=WHITE)
        relu_label.move_to(relu_box.get_center())

        # Output label
        out_label = Text("H(x) = F(x) + x", font_size=18, color=YELLOW)
        out_label.next_to(relu_box, DOWN, buff=0.3)

        # Animate main path boxes
        for box, label in zip(boxes, labels):
            self.play(FadeIn(box), FadeIn(label), run_time=0.25)

        # Main path arrows
        main_arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrow = Arrow(
                start=boxes[i].get_bottom(),
                end=boxes[i + 1].get_top(),
                color=WHITE, stroke_width=2, buff=0.08,
                max_tip_length_to_length_ratio=0.15
            )
            main_arrows.add(arrow)
            self.play(GrowArrow(arrow), run_time=0.15)

        # Arrow from last conv->bn to addition
        arrow_to_add = Arrow(
            start=boxes[-1].get_bottom(),
            end=add_circle.get_top(),
            color=WHITE, stroke_width=2, buff=0.08,
            max_tip_length_to_length_ratio=0.15
        )

        self.play(FadeIn(add_circle), FadeIn(add_text), run_time=0.3)
        self.play(GrowArrow(arrow_to_add), run_time=0.2)

        # F(x) label on main path
        fx_label = Text("F(x)", font_size=16, color="#4ecdc4")
        fx_label.next_to(arrow_to_add, LEFT, buff=0.15)
        self.play(FadeIn(fx_label), run_time=0.2)

        # Skip connection - the key visual
        skip_start = boxes[0].get_right() + RIGHT * 0.05
        skip_end = add_circle.get_right() + RIGHT * 0.05
        skip_offset = 2.2

        # Draw skip path: right from input, down, then left to addition
        skip_path = VGroup(
            Line(
                skip_start,
                skip_start + RIGHT * skip_offset,
                color="#ffd93d", stroke_width=3
            ),
            Line(
                skip_start + RIGHT * skip_offset,
                skip_end + RIGHT * skip_offset,
                color="#ffd93d", stroke_width=3
            ),
            Arrow(
                start=skip_end + RIGHT * skip_offset,
                end=skip_end,
                color="#ffd93d", stroke_width=3, buff=0.08,
                max_tip_length_to_length_ratio=0.1
            ),
        )

        skip_label = Text("x (identity)", font_size=16, color="#ffd93d")
        skip_label.move_to(
            skip_start + RIGHT * skip_offset + DOWN * 1.2
        )
        skip_label.rotate(PI / 2)

        self.play(
            *[Create(seg) for seg in skip_path],
            FadeIn(skip_label),
            run_time=0.8
        )

        self.wait(0.3)

        # Arrow from addition to final relu
        arrow_to_relu = Arrow(
            start=add_circle.get_bottom(),
            end=relu_box.get_top(),
            color=WHITE, stroke_width=2, buff=0.08,
            max_tip_length_to_length_ratio=0.15
        )
        self.play(
            FadeIn(relu_box), FadeIn(relu_label),
            GrowArrow(arrow_to_relu),
            run_time=0.3
        )

        # Output formula
        self.play(Write(out_label), run_time=0.5)

        # Highlight insight
        insight = Text(
            "If F(x) = 0, output = x  (identity, no harm done)",
            font_size=16, color="#a8e6cf"
        )
        insight.next_to(out_label, DOWN, buff=0.2)
        self.play(FadeIn(insight), run_time=0.4)

        self.wait(2.0)
