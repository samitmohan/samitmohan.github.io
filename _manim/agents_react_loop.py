from manim import *
import numpy as np


class ReactLoop(Scene):
    """Visualizes the ReAct loop: Think -> Act -> Observe cycle."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("The ReAct Loop", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # Three nodes in a triangle layout
        labels = ["Think", "Act", "Observe"]
        colors = ["#4ecdc4", "#ffd93d", "#ff6b6b"]
        # Top, bottom-left, bottom-right
        positions = [
            UP * 0.8,
            DOWN * 1.2 + LEFT * 2.0,
            DOWN * 1.2 + RIGHT * 2.0,
        ]

        # Create nodes
        nodes = VGroup()
        node_texts = VGroup()
        for label, color, pos in zip(labels, colors, positions):
            rect = RoundedRectangle(
                width=1.6, height=0.7,
                corner_radius=0.12,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2,
            )
            rect.move_to(pos)
            txt = Text(label, font_size=20, color=color, weight=BOLD)
            txt.move_to(pos)
            nodes.add(rect)
            node_texts.add(txt)

        self.play(
            *[FadeIn(n) for n in nodes],
            *[FadeIn(t) for t in node_texts],
            run_time=0.5,
        )

        # Curved arrows: Think -> Act -> Observe -> Think
        arrows = VGroup()
        for i in range(3):
            j = (i + 1) % 3
            arrow = CurvedArrow(
                start_point=positions[i],
                end_point=positions[j],
                color=GREY_B,
                stroke_width=2,
                angle=0.4,
                tip_length=0.15,
            )
            # Shorten by placing start/end with buff
            direction = positions[j] - positions[i]
            unit = direction / np.linalg.norm(direction)
            arrow = CurvedArrow(
                start_point=positions[i] + unit * 0.45,
                end_point=positions[j] - unit * 0.45,
                color=GREY_B,
                stroke_width=2,
                angle=0.4,
                tip_length=0.15,
            )
            arrows.add(arrow)

        self.play(*[Create(a) for a in arrows], run_time=0.4)
        self.wait(0.2)

        # Traversal dot
        dot = Dot(color=YELLOW, radius=0.09)
        dot.move_to(positions[0] + UP * 0.5)
        self.play(FadeIn(dot), run_time=0.15)

        # Ephemeral text positions: Think above, Act left, Observe right
        ephemeral_dirs = [UP, LEFT, RIGHT]
        ephemeral_buffs = [0.55, 0.6, 0.6]

        # Two iterations of the loop
        iteration_texts = [
            ["I need the weather API", 'call get_weather("SF")', "72F, sunny"],
            ["format the response", "call format_response()", "Done"],
        ]

        for iteration, texts in enumerate(iteration_texts):
            for step in range(3):
                node_idx = step
                # Move dot to this node
                self.play(
                    dot.animate.move_to(positions[node_idx] + UP * 0.5),
                    run_time=0.2,
                )

                # Highlight node
                self.play(
                    nodes[node_idx].animate.set_fill(opacity=0.5),
                    run_time=0.15,
                )

                # Show ephemeral text
                eph = Text(
                    texts[step], font_size=13, color=GREY_A,
                    font="Courier New",
                )
                eph.next_to(
                    nodes[node_idx], ephemeral_dirs[node_idx],
                    buff=ephemeral_buffs[node_idx],
                )

                # Special case: green flash on Observe "Done" in iteration 2
                if iteration == 1 and step == 2:
                    eph.set_color("#a8e6cf")
                    self.play(FadeIn(eph), run_time=0.15)
                    self.play(
                        nodes[node_idx].animate.set_fill(
                            color="#a8e6cf", opacity=0.6
                        ),
                        run_time=0.2,
                    )
                    self.wait(0.3)
                    self.play(
                        FadeOut(eph),
                        nodes[node_idx].animate.set_fill(
                            color=colors[node_idx], opacity=0.2
                        ),
                        run_time=0.2,
                    )
                else:
                    self.play(FadeIn(eph), run_time=0.15)
                    self.wait(0.25)
                    self.play(
                        FadeOut(eph),
                        nodes[node_idx].animate.set_fill(opacity=0.2),
                        run_time=0.15,
                    )

        # Fade out dot
        self.play(FadeOut(dot), run_time=0.2)

        # Punchline
        punchline = Text(
            "LLM decides what to do next - that's the whole idea",
            font_size=16, color=GREY_B,
        )
        punchline.to_edge(DOWN, buff=0.4)
        self.play(Write(punchline), run_time=0.5)
        self.wait(2.0)
