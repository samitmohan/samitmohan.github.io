from manim import *


class DecoratorStacking(Scene):
    """Visualizes how Python decorators wrap functions - the onion model."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Decorator Wrapping", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # --- Phase 1: Show the original function ---
        phase1_label = Text("1. Define the original function", font_size=16, color=GREY_B)
        phase1_label.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(phase1_label), run_time=0.3)

        # Inner box: greet()
        inner_box = RoundedRectangle(
            width=2.4, height=0.8,
            corner_radius=0.1,
            fill_color="#4ecdc4", fill_opacity=0.2,
            stroke_color="#4ecdc4", stroke_width=2
        )
        inner_label = Text("greet()", font_size=20, color="#4ecdc4", weight=BOLD)
        inner_label.move_to(inner_box.get_center())
        inner_group = VGroup(inner_box, inner_label)
        inner_group.move_to(UP * 0.3)

        returns_label = Text('returns "yo"', font_size=14, color=GREY_B)
        returns_label.next_to(inner_box, DOWN, buff=0.15)

        self.play(FadeIn(inner_box), FadeIn(inner_label), run_time=0.4)
        self.play(FadeIn(returns_label), run_time=0.3)
        self.wait(0.3)

        # --- Phase 2: Apply @uppercase decorator ---
        self.play(FadeOut(phase1_label), run_time=0.2)
        phase2_label = Text("2. @uppercase wraps greet()", font_size=16, color=GREY_B)
        phase2_label.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(phase2_label), run_time=0.3)

        # Decorator code snippet at left
        code_lines = VGroup(
            Text("def uppercase(func):", font_size=13, color="#a8e6cf"),
            Text("    def wrapper():", font_size=13, color="#a8e6cf"),
            Text("        result = func()", font_size=13, color=GREY_B),
            Text("        return result.upper()", font_size=13, color="#ffd93d"),
            Text("    return wrapper", font_size=13, color=GREY_B),
        )
        code_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        code_lines.move_to(LEFT * 4.2 + DOWN * 0.8)

        code_bg = RoundedRectangle(
            width=code_lines.width + 0.4, height=code_lines.height + 0.3,
            corner_radius=0.1,
            fill_color="#2a2a4e", fill_opacity=0.5,
            stroke_color="#555580", stroke_width=1
        )
        code_bg.move_to(code_lines.get_center())
        self.play(FadeIn(code_bg), FadeIn(code_lines), run_time=0.5)

        # Outer box: wrapper wraps around greet
        outer_box = RoundedRectangle(
            width=3.6, height=1.6,
            corner_radius=0.15,
            fill_color="#ffd93d", fill_opacity=0.08,
            stroke_color="#ffd93d", stroke_width=2.5
        )
        outer_box.move_to(inner_box.get_center() + DOWN * 0.1)

        wrapper_label = Text("wrapper()", font_size=16, color="#ffd93d")
        wrapper_label.next_to(outer_box, UP, buff=0.08)

        decorator_tag = Text("@uppercase", font_size=16, color="#ffd93d", weight=BOLD)
        decorator_tag.next_to(wrapper_label, UP, buff=0.05)

        self.play(
            Create(outer_box),
            FadeIn(wrapper_label),
            FadeIn(decorator_tag),
            run_time=0.6
        )
        self.wait(0.4)

        # --- Phase 3: Show the call flow ---
        self.play(
            FadeOut(phase2_label),
            FadeOut(code_bg), FadeOut(code_lines),
            run_time=0.3
        )
        phase3_label = Text("3. Call flow when you call greet()", font_size=16, color=GREY_B)
        phase3_label.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(phase3_label), run_time=0.3)

        # Shift the decorator visual up-left to make room
        decorator_visual = VGroup(
            inner_box, inner_label, returns_label,
            outer_box, wrapper_label, decorator_tag
        )
        self.play(decorator_visual.animate.scale(0.75).move_to(LEFT * 3.5 + DOWN * 0.5), run_time=0.4)

        # Build call flow diagram on the right
        flow_x = RIGHT * 1.8
        step_spacing = 0.95

        steps = [
            ("greet()", "#a8e6cf", "you call greet()"),
            ("wrapper()", "#ffd93d", "actually calls wrapper()"),
            ("original greet()", "#4ecdc4", 'runs original -> "yo"'),
            ('.upper()', "#ff6b6b", 'transforms -> "YO"'),
            ("return", "#a8e6cf", 'returns "YO"'),
        ]

        flow_boxes = []
        flow_labels = []
        flow_descs = []
        y_start = 1.6

        for i, (text, color, desc) in enumerate(steps):
            y = y_start - i * step_spacing
            box = RoundedRectangle(
                width=2.2, height=0.5,
                corner_radius=0.1,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2
            )
            box.move_to(flow_x + UP * y)
            label = Text(text, font_size=16, color=color, weight=BOLD)
            label.move_to(box.get_center())
            desc_text = Text(desc, font_size=12, color=GREY_B)
            desc_text.next_to(box, RIGHT, buff=0.2)
            flow_boxes.append(box)
            flow_labels.append(label)
            flow_descs.append(desc_text)

        # Animate flow step by step
        flow_arrows = []
        for i in range(len(steps)):
            self.play(
                FadeIn(flow_boxes[i]),
                FadeIn(flow_labels[i]),
                FadeIn(flow_descs[i]),
                run_time=0.3
            )
            if i < len(steps) - 1:
                arrow = Arrow(
                    start=flow_boxes[i].get_bottom(),
                    end=flow_boxes[i + 1].get_top(),
                    color=WHITE, stroke_width=2, buff=0.06,
                    max_tip_length_to_length_ratio=0.15
                )
                flow_arrows.append(arrow)
                self.play(GrowArrow(arrow), run_time=0.15)

        self.wait(0.4)

        # --- Phase 4: The transformation highlight ---
        self.play(FadeOut(phase3_label), run_time=0.2)

        # Show "yo" -> "YO" transformation with a bold arrow
        transform_group = VGroup()
        yo_text = Text('"yo"', font_size=24, color="#4ecdc4")
        arrow_transform = Arrow(
            LEFT * 0.6, RIGHT * 0.6,
            color="#ffd93d", stroke_width=3, buff=0.05,
            max_tip_length_to_length_ratio=0.15
        )
        yo_upper = Text('"YO"', font_size=24, color="#ff6b6b", weight=BOLD)

        transform_group.add(yo_text, arrow_transform, yo_upper)
        transform_group.arrange(RIGHT, buff=0.2)
        transform_group.to_edge(DOWN, buff=0.55)

        upper_note = Text(".upper() applied by decorator", font_size=14, color="#ffd93d")
        upper_note.next_to(transform_group, DOWN, buff=0.15)

        self.play(FadeIn(yo_text), run_time=0.3)
        self.play(GrowArrow(arrow_transform), run_time=0.3)
        self.play(FadeIn(yo_upper), run_time=0.3)
        self.play(FadeIn(upper_note), run_time=0.3)

        self.wait(2.0)
