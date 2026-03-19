from manim import *


class ModuleSystem(Scene):
    """Module hierarchy tree for a CNN model, showing parameters()
    collection and state_dict serialization."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("nn.Module Hierarchy", font_size=30, color=WHITE)
        subtitle = Text(
            "parameters() collects all learnable weights recursively",
            font_size=14, color=GREY_B
        )
        title.to_edge(UP, buff=0.3)
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(Write(title), run_time=0.6)
        self.play(FadeIn(subtitle), run_time=0.3)

        # -- Tree structure --
        root_color = "#ffd93d"
        child_colors = [
            "#4ecdc4", "#45b7d1", "#4ecdc4", "#45b7d1", "#96ceb4", "#ff6b6b"
        ]
        children = [
            "Conv2d(1,8)", "MaxPool2d", "Conv2d(8,16)",
            "MaxPool2d", "Flatten", "Linear(256,10)"
        ]
        has_params = [True, False, True, False, False, True]
        param_counts = ["72", "-", "1168", "-", "-", "2570"]

        # Root node
        root_box = RoundedRectangle(
            width=2.2, height=0.55, corner_radius=0.1,
            fill_color=root_color, fill_opacity=0.2,
            stroke_color=root_color, stroke_width=2
        )
        root_box.move_to(UP * 1.6)
        root_label = Text("CNN Model", font_size=18, color=root_color, weight=BOLD)
        root_label.move_to(root_box.get_center())

        self.play(FadeIn(root_box), Write(root_label), run_time=0.4)

        # Child nodes - arranged in two rows of 3
        child_boxes = []
        child_labels = []
        child_groups = []

        row1_y = 0.2
        row2_y = -1.2
        col_spacing = 3.0
        row1_start_x = -col_spacing
        row2_start_x = -col_spacing

        positions = []
        for i in range(3):
            positions.append(RIGHT * (row1_start_x + i * col_spacing) + UP * row1_y)
        for i in range(3):
            positions.append(RIGHT * (row2_start_x + i * col_spacing) + UP * row2_y)

        for i, (name, color, pos) in enumerate(zip(children, child_colors, positions)):
            box = RoundedRectangle(
                width=2.2, height=0.50, corner_radius=0.1,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2
            )
            box.move_to(pos)
            label = Text(name, font_size=14, color=WHITE)
            label.move_to(box.get_center())

            child_boxes.append(box)
            child_labels.append(label)
            child_groups.append(VGroup(box, label))

        # Arrows from root to children
        arrows = VGroup()
        for i, box in enumerate(child_boxes):
            arrow = Arrow(
                start=root_box.get_bottom(),
                end=box.get_top(),
                color=GREY_B, stroke_width=2, buff=0.06,
                max_tip_length_to_length_ratio=0.15
            )
            arrows.add(arrow)

        # Animate children appearing with arrows
        for i in range(len(children)):
            self.play(
                GrowArrow(arrows[i]),
                FadeIn(child_groups[i]),
                run_time=0.25
            )

        self.wait(0.4)

        # -- Parameter count badges --
        param_badges = []
        for i, (box, count, hp) in enumerate(
            zip(child_boxes, param_counts, has_params)
        ):
            if hp:
                badge_color = "#a8e6cf"
                badge = Text(f"{count} params", font_size=11, color=badge_color)
            else:
                badge_color = GREY
                badge = Text("no params", font_size=11, color=badge_color)
            badge.next_to(box, DOWN, buff=0.08)
            param_badges.append(badge)

        self.play(
            *[FadeIn(b) for b in param_badges],
            run_time=0.4
        )

        self.wait(0.3)

        # -- Animate parameters() collection: highlight flows from leaves to root --
        collect_label = Text(
            "model.parameters()", font_size=20, color="#a8e6cf", weight=BOLD
        )
        collect_label.move_to(DOWN * 2.5)
        self.play(Write(collect_label), run_time=0.4)

        # Highlight param-bearing modules one by one
        highlight_order = [0, 2, 5]  # Conv2d(1,8), Conv2d(8,16), Linear(256,10)
        for idx in highlight_order:
            # Flash the child box
            box = child_boxes[idx]
            color = child_colors[idx]
            self.play(
                box.animate.set_fill(color, opacity=0.5),
                run_time=0.2
            )
            # Send a dot flowing up the arrow to root
            dot = Dot(color="#a8e6cf", radius=0.06)
            dot.move_to(box.get_top())
            self.add(dot)
            self.play(
                dot.animate.move_to(root_box.get_bottom()),
                run_time=0.35
            )
            self.play(
                box.animate.set_fill(color, opacity=0.2),
                FadeOut(dot),
                run_time=0.15
            )

        # Flash root to show all params collected
        self.play(
            root_box.animate.set_fill(root_color, opacity=0.5),
            run_time=0.3
        )
        total_label = Text(
            "Total: 3810 parameters", font_size=16, color="#ffd93d"
        )
        total_label.next_to(collect_label, DOWN, buff=0.15)
        self.play(
            Write(total_label),
            root_box.animate.set_fill(root_color, opacity=0.2),
            run_time=0.4
        )

        self.wait(0.5)

        # -- Phase 2: state_dict table --
        # Fade out the tree
        tree_elements = VGroup(
            root_box, root_label, arrows,
            *child_groups, *param_badges,
            collect_label, total_label
        )
        self.play(FadeOut(tree_elements), run_time=0.5)

        sd_title = Text("state_dict()", font_size=24, color="#ffd93d", weight=BOLD)
        sd_title.move_to(UP * 1.8)
        sd_subtitle = Text(
            "serializable snapshot of all parameters",
            font_size=14, color=GREY_B
        )
        sd_subtitle.next_to(sd_title, DOWN, buff=0.1)
        self.play(Write(sd_title), FadeIn(sd_subtitle), run_time=0.4)

        # Table of key-value pairs
        entries = [
            ("conv1.weight", "Tensor(8, 1, 3, 3)"),
            ("conv1.bias", "Tensor(8)"),
            ("conv2.weight", "Tensor(16, 8, 3, 3)"),
            ("conv2.bias", "Tensor(16)"),
            ("fc.weight", "Tensor(10, 256)"),
            ("fc.bias", "Tensor(10)"),
        ]
        entry_colors = [
            "#4ecdc4", "#4ecdc4", "#4ecdc4", "#4ecdc4", "#ff6b6b", "#ff6b6b"
        ]

        table_start_y = 0.8
        row_height = 0.45
        key_x = -2.0
        val_x = 2.0

        # Header
        key_header = Text("Key", font_size=14, color=GREY, weight=BOLD)
        val_header = Text("Value", font_size=14, color=GREY, weight=BOLD)
        key_header.move_to(RIGHT * key_x + UP * (table_start_y + 0.4))
        val_header.move_to(RIGHT * val_x + UP * (table_start_y + 0.4))
        header_line = Line(
            LEFT * 3.5 + UP * (table_start_y + 0.2),
            RIGHT * 3.5 + UP * (table_start_y + 0.2),
            color=GREY_D, stroke_width=1
        )
        self.play(FadeIn(key_header), FadeIn(val_header), Create(header_line),
                  run_time=0.3)

        for i, (key, val) in enumerate(entries):
            y = table_start_y - i * row_height
            color = entry_colors[i]

            key_text = Text(key, font_size=13, color=color)
            key_text.move_to(RIGHT * key_x + UP * y)

            val_text = Text(val, font_size=13, color=WHITE)
            val_text.move_to(RIGHT * val_x + UP * y)

            # Separator line
            sep = Line(
                LEFT * 3.5 + UP * (y - row_height / 2 + 0.04),
                RIGHT * 3.5 + UP * (y - row_height / 2 + 0.04),
                color=GREY_D, stroke_width=0.5
            )

            self.play(
                FadeIn(key_text), FadeIn(val_text), Create(sep),
                run_time=0.2
            )

        # Save / load note
        save_note = Text(
            "torch.save(model.state_dict(), 'model.pt')",
            font_size=16, color="#a8e6cf"
        )
        save_note.to_edge(DOWN, buff=0.5)
        self.play(Write(save_note), run_time=0.5)

        self.wait(2.0)
