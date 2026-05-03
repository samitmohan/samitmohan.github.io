from manim import *
import numpy as np


class FrameworkEvolution(Scene):
    """Three generations of agent frameworks on an animated timeline."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        # -- coordinate helpers --
        TIMELINE_Y = 1.5
        LABEL_Y = 2.7
        X_LEFT, X_RIGHT = -5.5, 5.5

        def year_x(year):
            """Map a year (float) to x position."""
            return X_LEFT + (year - 2022) / (2026 - 2022) * (X_RIGHT - X_LEFT)

        # -- 1  Title --
        title = Text("The Three Generations", font_size=30, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=0.6)

        # -- 2  Timeline axis --
        timeline = Arrow(
            start=LEFT * 5.8 + UP * TIMELINE_Y,
            end=RIGHT * 5.8 + UP * TIMELINE_Y,
            color=GREY_B, stroke_width=2, tip_length=0.15,
        )
        years = [2022, 2023, 2024, 2025, 2026]
        ticks = VGroup()
        year_labels = VGroup()
        for y in years:
            x = year_x(y)
            tick = Line(
                UP * (TIMELINE_Y + 0.1), UP * (TIMELINE_Y - 0.1),
                stroke_width=1.5, color=GREY_B,
            ).shift(RIGHT * x)
            label = Text(str(y), font_size=14, color=GREY_B).next_to(tick, DOWN, buff=0.1)
            ticks.add(tick)
            year_labels.add(label)

        self.play(
            Create(timeline),
            *[FadeIn(t) for t in ticks],
            *[FadeIn(l) for l in year_labels],
            run_time=0.6,
        )

        # -- helpers --
        def make_pill(text, x, row, fill="#444441", text_color=WHITE):
            pill = RoundedRectangle(
                width=1.4, height=0.35, corner_radius=0.15,
                fill_color=fill, fill_opacity=0.85,
                stroke_width=0, stroke_color=fill,
            )
            txt = Text(text, font_size=13, color=text_color)
            txt.move_to(pill.get_center())
            group = VGroup(pill, txt)
            y_offset = TIMELINE_Y - 0.55 - row * 0.45
            group.move_to(RIGHT * x + UP * y_offset)
            return group

        def make_region(x_start, x_end, color, opacity=0.15):
            w = x_end - x_start
            cx = (x_start + x_end) / 2
            rect = Rectangle(
                width=w, height=2.8,
                fill_color=color, fill_opacity=opacity,
                stroke_width=0,
            )
            rect.move_to(RIGHT * cx + UP * (TIMELINE_Y - 0.5))
            return rect

        # -- 3  Generation 1: Chains (2022-2023) --
        g1_region = make_region(year_x(2022), year_x(2023.9), "#5F5E5A")
        g1_label = Text("Chains", font_size=20, color="#B4B2A9", weight=BOLD)
        g1_label.move_to(
            RIGHT * (year_x(2022) + year_x(2023.9)) / 2 + UP * LABEL_Y
        )

        g1_pills = [
            make_pill("ReAct paper", year_x(2022.8), 0),
            make_pill("AutoGPT", year_x(2023.2), 1),
            make_pill("LangChain v0", year_x(2023.5), 0),
        ]

        self.play(FadeIn(g1_region), FadeIn(g1_label), run_time=0.3)
        for pill in g1_pills:
            self.play(FadeIn(pill, shift=DOWN * 0.3), run_time=0.2)
        self.wait(0.5)

        # -- 4  Generation 2: Stateful Agents (2024) --
        g2_region = make_region(year_x(2023.9), year_x(2024.95), "#534AB7")
        g2_label = Text("Stateful Agents", font_size=20, color="#a78bfa", weight=BOLD)
        g2_label.move_to(
            RIGHT * (year_x(2023.9) + year_x(2024.95)) / 2 + UP * LABEL_Y
        )

        g2_pills = [
            make_pill("LangGraph", year_x(2024.1), 0),
            make_pill("CrewAI", year_x(2024.4), 1),
            make_pill("AutoGen v0.4", year_x(2024.7), 0),
            make_pill("MCP", year_x(2024.85), 1, fill="#fbbf24", text_color=BLACK),
        ]

        self.play(FadeIn(g2_region), FadeIn(g2_label), run_time=0.3)
        for pill in g2_pills:
            self.play(FadeIn(pill, shift=DOWN * 0.3), run_time=0.2)
        self.wait(0.5)

        # -- 5  Generation 3: Harnesses (2025-2026) --
        g3_region = make_region(year_x(2024.95), year_x(2026), "#0F6E56")
        g3_label = Text("Harnesses", font_size=20, color="#4ecdc4", weight=BOLD)
        g3_label.move_to(
            RIGHT * (year_x(2024.95) + year_x(2026)) / 2 + UP * LABEL_Y
        )

        g3_pills = [
            make_pill("Agents SDK", year_x(2025.2), 0),
            make_pill("Claude SDK", year_x(2025.45), 1),
            make_pill("Strands", year_x(2025.6), 0),
            make_pill("Google ADK", year_x(2025.8), 1),
            make_pill("Agno", year_x(2026.0) - 0.2, 0),
        ]

        self.play(FadeIn(g3_region), FadeIn(g3_label), run_time=0.3)
        for pill in g3_pills:
            self.play(FadeIn(pill, shift=DOWN * 0.3), run_time=0.15)
        self.wait(0.5)

        # -- 6  Trend arrow --
        arrow_y = TIMELINE_Y - 2.6
        trend_arrow = Arrow(
            start=LEFT * 5.0 + UP * arrow_y,
            end=RIGHT * 5.0 + UP * arrow_y,
            color="#4ecdc4", stroke_width=3, tip_length=0.2,
        )
        trend_text = Text(
            "Less scaffolding, more model capability",
            font_size=14, color=GREY_B,
        )
        trend_text.next_to(trend_arrow, DOWN, buff=0.15)

        self.play(
            GrowArrow(trend_arrow),
            FadeIn(trend_text, shift=UP * 0.1),
            run_time=0.8,
        )
        self.wait(2)
