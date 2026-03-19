from manim import *
import numpy as np


class ContextWindow(Scene):
    """Context window filling up: green -> yellow -> red, then smart compaction."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("The Context Window is a Budget", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        bar_width = 10.0
        bar_height = 0.8
        bar_y = 1.0

        # Background bar (empty)
        bg_bar = RoundedRectangle(
            width=bar_width, height=bar_height,
            corner_radius=0.1,
            fill_color=GREY_E, fill_opacity=0.3,
            stroke_color=GREY, stroke_width=1
        )
        bg_bar.move_to(UP * bar_y)

        capacity_label = Text("200k tokens", font_size=14, color=GREY_B)
        capacity_label.next_to(bg_bar, RIGHT, buff=0.2)
        self.play(FadeIn(bg_bar), FadeIn(capacity_label), run_time=0.3)

        # Segments to fill the bar
        segments = [
            ("system prompt", 0.05, "#4ecdc4"),
            ("CLAUDE.md", 0.08, "#4ecdc4"),
            ("file reads", 0.20, "#ffd93d"),
            ("tool outputs", 0.18, "#ffd93d"),
            ("conversation", 0.25, "#ff6b6b"),
            ("more files...", 0.15, "#ff6b6b"),
        ]

        current_x = bg_bar.get_left()[0]
        left_edge = current_x
        filled_segments = VGroup()

        for label, fraction, color in segments:
            seg_width = bar_width * fraction
            seg = Rectangle(
                width=seg_width, height=bar_height - 0.04,
                fill_color=color, fill_opacity=0.6,
                stroke_width=0
            )
            seg.move_to(
                RIGHT * (current_x + seg_width / 2) + UP * bar_y
            )

            seg_label = Text(label, font_size=9, color=WHITE)
            if seg_width > 0.8:
                seg_label.move_to(seg.get_center())
            else:
                seg_label.move_to(seg.get_center())
                seg_label.scale(0.8)

            filled_segments.add(VGroup(seg, seg_label))
            current_x += seg_width

            self.play(
                GrowFromEdge(seg, LEFT),
                FadeIn(seg_label),
                run_time=0.3
            )

        # Percentage indicator
        total_pct = sum(s[1] for s in segments)
        pct_text = Text(
            f"{total_pct * 100:.0f}% used", font_size=16, color="#ff6b6b"
        )
        pct_text.next_to(bg_bar, DOWN, buff=0.15)
        self.play(FadeIn(pct_text), run_time=0.2)
        self.wait(0.5)

        # Warning text
        warning = Text(
            "early instructions get fuzzy after compression",
            font_size=14, color="#ff6b6b"
        )
        warning.next_to(pct_text, DOWN, buff=0.15)
        self.play(Write(warning), run_time=0.4)
        self.wait(0.5)

        # --- Smart approach ---
        smart_title = Text("after /compact", font_size=20, color="#a8e6cf", weight=BOLD)
        smart_title.move_to(DOWN * 1.2)
        self.play(Write(smart_title), run_time=0.3)

        bar_y2 = -2.0
        bg_bar2 = RoundedRectangle(
            width=bar_width, height=bar_height,
            corner_radius=0.1,
            fill_color=GREY_E, fill_opacity=0.3,
            stroke_color=GREY, stroke_width=1
        )
        bg_bar2.move_to(UP * bar_y2)
        self.play(FadeIn(bg_bar2), run_time=0.2)

        # Compacted segments - much smaller
        compact_segments = [
            ("system", 0.05, "#4ecdc4"),
            ("CLAUDE.md", 0.08, "#4ecdc4"),
            ("summary", 0.10, "#9b59b6"),
            ("current task", 0.12, "#ffd93d"),
        ]

        current_x2 = bg_bar2.get_left()[0]
        for label, fraction, color in compact_segments:
            seg_width = bar_width * fraction
            seg = Rectangle(
                width=seg_width, height=bar_height - 0.04,
                fill_color=color, fill_opacity=0.6,
                stroke_width=0
            )
            seg.move_to(
                RIGHT * (current_x2 + seg_width / 2) + UP * bar_y2
            )
            seg_label = Text(label, font_size=9, color=WHITE)
            seg_label.move_to(seg.get_center())

            self.play(
                GrowFromEdge(seg, LEFT),
                FadeIn(seg_label),
                run_time=0.2
            )
            current_x2 += seg_width

        compact_pct = sum(s[1] for s in compact_segments)
        pct_text2 = Text(
            f"{compact_pct * 100:.0f}% used - room to work",
            font_size=16, color="#a8e6cf"
        )
        pct_text2.next_to(bg_bar2, DOWN, buff=0.15)
        self.play(FadeIn(pct_text2), run_time=0.2)

        punchline = Text(
            "focused sprints > marathon sessions",
            font_size=16, color=GREY_B
        )
        punchline.to_edge(DOWN, buff=0.3)
        self.play(Write(punchline), run_time=0.4)
        self.wait(2.0)
