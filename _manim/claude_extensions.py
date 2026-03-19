from manim import *
import numpy as np


class Extensions(Scene):
    """Shows how Skills, Hooks, and MCPs plug into the agentic loop."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Extension Points", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # Horizontal loop layout - avoids crossing arrows
        loop_labels = ["Read", "Plan", "Code", "Run"]
        loop_colors = ["#4ecdc4", "#ffd93d", "#9b59b6", "#ff6b6b"]
        n = len(loop_labels)
        spacing = 2.0
        total_w = (n - 1) * spacing
        start_x = -total_w / 2
        loop_y = 0.3

        nodes = VGroup()
        positions = []
        for i, (label, color) in enumerate(zip(loop_labels, loop_colors)):
            x = start_x + i * spacing
            pos = RIGHT * x + UP * loop_y
            positions.append(pos)

            rect = RoundedRectangle(
                width=1.2, height=0.55,
                corner_radius=0.08,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2
            )
            rect.move_to(pos)
            txt = Text(label, font_size=15, color=color, weight=BOLD)
            txt.move_to(pos)
            nodes.add(VGroup(rect, txt))

        # Forward arrows: Read -> Plan -> Code -> Run
        fwd_arrows = VGroup()
        for i in range(n - 1):
            arrow = Arrow(
                start=positions[i], end=positions[i + 1],
                color=GREY_B, stroke_width=1.5,
                buff=0.4,
                max_tip_length_to_length_ratio=0.12
            )
            fwd_arrows.add(arrow)

        # Return arrow: Run loops back to Read (curved underneath)
        return_arrow = CurvedArrow(
            start_point=positions[3] + DOWN * 0.35,
            end_point=positions[0] + DOWN * 0.35,
            color=GREY_B, stroke_width=1.5,
            angle=-TAU / 3
        )

        self.play(
            *[FadeIn(node) for node in nodes],
            *[GrowArrow(a) for a in fwd_arrows],
            run_time=0.6
        )
        self.play(Create(return_arrow), run_time=0.3)

        loop_label = Text("agentic loop", font_size=11, color=GREY)
        loop_label.move_to(UP * loop_y + DOWN * 1.2)
        self.play(FadeIn(loop_label), run_time=0.2)
        self.wait(0.3)

        # Extension boxes - each directly above/below its target node
        extensions = [
            {
                "name": "Skills",
                "color": "#4ecdc4",
                "desc": "/review-pr, /commit",
                "target": 1,  # Plan
                "side": UP,
            },
            {
                "name": "Hooks",
                "color": "#ff6b6b",
                "desc": "before/after tool calls",
                "target": 2,  # Code
                "side": UP,
            },
            {
                "name": "MCPs",
                "color": "#ffd93d",
                "desc": "external tool servers",
                "target": 0,  # Read
                "side": UP,
            },
        ]

        box_offset = 1.6

        for ext in extensions:
            color = ext["color"]
            target_pos = positions[ext["target"]]
            box_pos = target_pos + ext["side"] * box_offset

            box = RoundedRectangle(
                width=2.2, height=0.8,
                corner_radius=0.1,
                fill_color=color, fill_opacity=0.15,
                stroke_color=color, stroke_width=2
            )
            box.move_to(box_pos)

            name_txt = Text(ext["name"], font_size=15, color=color, weight=BOLD)
            name_txt.move_to(box.get_center() + UP * 0.1)

            desc_txt = Text(ext["desc"], font_size=9, color=GREY_B)
            desc_txt.move_to(box.get_center() + DOWN * 0.16)

            # Straight vertical arrow from box to node
            connector = Arrow(
                start=box.get_bottom() if ext["side"][1] > 0 else box.get_top(),
                end=nodes[ext["target"]][0].get_top() if ext["side"][1] > 0 else nodes[ext["target"]][0].get_bottom(),
                color=color,
                stroke_width=2,
                buff=0.06,
                max_tip_length_to_length_ratio=0.12
            )

            self.play(
                FadeIn(box, shift=ext["side"] * -0.2),
                FadeIn(name_txt),
                FadeIn(desc_txt),
                run_time=0.3
            )
            self.play(GrowArrow(connector), run_time=0.25)

        # Punchline
        punchline = Text(
            "skills = on-demand instructions | hooks = event triggers | MCPs = new tools",
            font_size=11, color=GREY_B
        )
        punchline.to_edge(DOWN, buff=0.3)
        self.play(Write(punchline), run_time=0.5)
        self.wait(2.0)
