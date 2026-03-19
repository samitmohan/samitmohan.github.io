from manim import *
import numpy as np


class AgenticLoop(Scene):
    """Visualizes Claude Code's agentic loop: Read -> Plan -> Code -> Run -> Check."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("The Agentic Loop", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # Node setup - circular layout
        labels = ["Read", "Plan", "Code", "Run", "Check"]
        colors = ["#4ecdc4", "#ffd93d", "#9b59b6", "#ff6b6b", "#4ecdc4"]
        n = len(labels)
        radius = 1.8
        center = DOWN * 0.3

        # Positions (starting from top, going clockwise)
        angles = [PI / 2 - i * 2 * PI / n for i in range(n)]
        positions = [
            center + radius * np.array([np.cos(a), np.sin(a), 0])
            for a in angles
        ]

        # Create nodes
        nodes = VGroup()
        node_texts = VGroup()
        for i, (label, color, pos) in enumerate(zip(labels, colors, positions)):
            rect = RoundedRectangle(
                width=1.3, height=0.6,
                corner_radius=0.1,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2
            )
            rect.move_to(pos)
            txt = Text(label, font_size=18, color=color, weight=BOLD)
            txt.move_to(pos)
            nodes.add(rect)
            node_texts.add(txt)

        self.play(
            *[FadeIn(n) for n in nodes],
            *[FadeIn(t) for t in node_texts],
            run_time=0.6
        )

        # Draw arrows between nodes (clockwise)
        arrows = VGroup()
        for i in range(n):
            j = (i + 1) % n
            arrow = Arrow(
                start=positions[i],
                end=positions[j],
                color=GREY_B,
                stroke_width=2,
                buff=0.45,
                max_tip_length_to_length_ratio=0.12
            )
            arrows.add(arrow)

        self.play(*[GrowArrow(a) for a in arrows], run_time=0.5)
        self.wait(0.3)

        # Animated dot traversing the loop - offset above node centers so it
        # doesn't cover text
        dot_offset = UP * 0.45
        dot = Dot(color=YELLOW, radius=0.08)
        dot.move_to(positions[0] + dot_offset)
        self.play(FadeIn(dot), run_time=0.2)

        # Two full iterations
        for iteration in range(2):
            for i in range(n):
                j = (i + 1) % n
                # Highlight current node
                self.play(
                    nodes[i].animate.set_fill(opacity=0.5),
                    run_time=0.15
                )

                # Move dot to next node
                self.play(
                    dot.animate.move_to(positions[j] + dot_offset),
                    run_time=0.25
                )

                # Reset node highlight
                self.play(
                    nodes[i].animate.set_fill(opacity=0.2),
                    run_time=0.1
                )

                # At Check node: show branch decision
                if j == 4:  # Check node
                    if iteration == 0:
                        # Error - loop back (red flash)
                        err = Text("error", font_size=14, color="#ff6b6b")
                        err.next_to(nodes[4], LEFT, buff=0.3)
                        self.play(
                            FadeIn(err),
                            nodes[4].animate.set_stroke(color="#ff6b6b"),
                            run_time=0.2
                        )
                        self.play(
                            FadeOut(err),
                            nodes[4].animate.set_stroke(color="#4ecdc4"),
                            run_time=0.2
                        )
                    else:
                        # Success - done (green flash)
                        ok = Text("done", font_size=14, color="#a8e6cf")
                        ok.next_to(nodes[4], LEFT, buff=0.3)
                        self.play(
                            FadeIn(ok),
                            nodes[4].animate.set_fill(color="#a8e6cf", opacity=0.5),
                            run_time=0.3
                        )

        # Fade out dot before punchline
        self.play(FadeOut(dot), run_time=0.2)

        # Final label
        punchline = Text(
            "not a chatbot - a while loop with an LLM",
            font_size=16, color=GREY_B
        )
        punchline.to_edge(DOWN, buff=0.4)
        self.play(Write(punchline), run_time=0.5)
        self.wait(2.0)
