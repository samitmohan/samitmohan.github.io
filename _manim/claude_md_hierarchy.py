from manim import *
import numpy as np


class ClaudeMdHierarchy(Scene):
    """Shows the CLAUDE.md hierarchy: global -> project -> subdirectory."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("CLAUDE.md Hierarchy", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # Three layers
        layer_data = [
            {
                "label": "~/.claude/CLAUDE.md",
                "color": "#4ecdc4",
                "rules": ["always use uv", "no emojis"],
                "scope": "global"
            },
            {
                "label": "project/CLAUDE.md",
                "color": "#ffd93d",
                "rules": ["bundle exec jekyll serve", "use ruff for linting"],
                "scope": "project"
            },
            {
                "label": "project/src/CLAUDE.md",
                "color": "#ff6b6b",
                "rules": ["components use .tsx", "no default exports"],
                "scope": "subdirectory"
            },
        ]

        layers = VGroup()
        layer_width = 5.0
        layer_height = 1.05
        spacing = 0.35
        start_y = 2.2

        for i, data in enumerate(layer_data):
            y = start_y - i * (layer_height + spacing)
            color = data["color"]

            rect = RoundedRectangle(
                width=layer_width, height=layer_height,
                corner_radius=0.12,
                fill_color=color, fill_opacity=0.12,
                stroke_color=color, stroke_width=2
            )
            rect.move_to(UP * y)

            # Label on left
            lbl = Text(data["label"], font_size=13, color=color, weight=BOLD)
            lbl.align_to(rect, LEFT).shift(RIGHT * 0.2 + DOWN * 0.02)
            lbl.align_to(rect, UP).shift(DOWN * 0.15)

            # Scope badge on right
            scope = Text(data["scope"], font_size=11, color=GREY_B)
            scope.align_to(rect, RIGHT).shift(LEFT * 0.2)
            scope.align_to(rect, UP).shift(DOWN * 0.15)

            # Rule previews
            rules_group = VGroup()
            for rule in data["rules"]:
                rule_txt = Text(f"  {rule}", font_size=10, color=GREY_A)
                rules_group.add(rule_txt)
            rules_group.arrange(DOWN, buff=0.03, aligned_edge=LEFT)
            rules_group.align_to(rect, LEFT).shift(RIGHT * 0.25)
            rules_group.align_to(rect, DOWN).shift(UP * 0.12)

            layer_group = VGroup(rect, lbl, scope, rules_group)
            layers.add(layer_group)

        # Animate layers appearing top-to-bottom
        for layer in layers:
            self.play(FadeIn(layer, shift=DOWN * 0.3), run_time=0.4)

        # Arrows between layers
        for i in range(len(layers) - 1):
            arrow = Arrow(
                start=layers[i][0].get_bottom(),
                end=layers[i + 1][0].get_top(),
                color=GREY_B,
                stroke_width=2,
                buff=0.06,
                max_tip_length_to_length_ratio=0.15
            )
            override_txt = Text("overrides", font_size=10, color=GREY)
            override_txt.next_to(arrow, RIGHT, buff=0.1)
            self.play(GrowArrow(arrow), FadeIn(override_txt), run_time=0.25)

        self.wait(0.5)

        # Merge animation: single "effective config" box well below the layers
        merge_label = Text("effective config", font_size=16, color=WHITE, weight=BOLD)
        merge_box = RoundedRectangle(
            width=5.0, height=0.7,
            corner_radius=0.12,
            fill_color="#9b59b6", fill_opacity=0.2,
            stroke_color="#9b59b6", stroke_width=2.5
        )
        merge_box.move_to(DOWN * 2.8)
        merge_label.move_to(merge_box.get_center())

        self.play(FadeIn(merge_box), Write(merge_label), run_time=0.4)

        # Single arrow from last layer to merged box
        merge_arrow = Arrow(
            start=layers[2][0].get_bottom(),
            end=merge_box.get_top(),
            color="#9b59b6",
            stroke_width=2,
            buff=0.06,
            max_tip_length_to_length_ratio=0.12
        )
        self.play(GrowArrow(merge_arrow), run_time=0.4)

        punchline = Text(
            "subdirectory wins - most specific scope takes priority",
            font_size=14, color=GREY_B
        )
        punchline.to_edge(DOWN, buff=0.3)
        self.play(Write(punchline), run_time=0.4)
        self.wait(2.0)
