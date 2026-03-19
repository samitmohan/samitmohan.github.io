from manim import *


class CompGraphBackward(Scene):
    """Computation graph with forward pass and backward gradient flow."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Autograd: Forward + Backward Pass", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # Node positions (hand-crafted DAG layout)
        # Inputs at left, loss at right
        #   a --\
        #        mul --> c --\
        #   b --/             add --> e (loss)
        #                d --/

        node_radius = 0.4
        positions = {
            "a": LEFT * 5.0 + UP * 1.2,
            "b": LEFT * 5.0 + DOWN * 1.2,
            "mul": LEFT * 2.0 + ORIGIN,
            "c": RIGHT * 0.5 + ORIGIN,
            "d": RIGHT * 0.5 + DOWN * 2.0,
            "add": RIGHT * 3.0 + DOWN * 0.8,
            "e": RIGHT * 5.5 + DOWN * 0.8,
        }

        # Node info: (label, color, value_forward, gradient)
        node_info = {
            "a":   ("a=3",    "#a8e6cf", "3",   "de/da=4"),
            "b":   ("b=4",    "#a8e6cf", "4",   "de/db=3"),
            "mul": ("a * b",  "#4ecdc4", "mul", "de/dc=1"),
            "c":   ("c=12",   "#45b7d1", "12",  "de/dc=1"),
            "d":   ("d=5",    "#a8e6cf", "5",   "de/dd=1"),
            "add": ("c + d",  "#4ecdc4", "add", "de/de=1"),
            "e":   ("e=17",   "#ffd93d", "17",  "de/de=1"),
        }

        edges = [
            ("a", "mul"),
            ("b", "mul"),
            ("mul", "c"),
            ("c", "add"),
            ("d", "add"),
            ("add", "e"),
        ]

        # ==================================================================
        # FORWARD PASS - Build the graph
        # ==================================================================
        phase_label = Text("Forward Pass", font_size=22, color="#4ecdc4")
        phase_label.move_to(UP * 2.5)
        self.play(FadeIn(phase_label), run_time=0.3)

        # Create nodes in topological order
        node_objects = {}
        node_labels = {}
        forward_order = ["a", "b", "d", "mul", "c", "add", "e"]

        for name in forward_order:
            pos = positions[name]
            label_text, color, _, _ = node_info[name]

            circle = Circle(
                radius=node_radius,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2,
            )
            circle.move_to(pos)

            label = Text(label_text, font_size=14, color=WHITE)
            label.move_to(pos)

            node_objects[name] = circle
            node_labels[name] = label

            self.play(FadeIn(circle), FadeIn(label), run_time=0.2)

        # Draw edges (arrows)
        arrow_objects = {}
        for src, dst in edges:
            arrow = Arrow(
                start=positions[src],
                end=positions[dst],
                color=WHITE, stroke_width=2, buff=node_radius + 0.05,
                max_tip_length_to_length_ratio=0.15,
            )
            arrow_objects[(src, dst)] = arrow
            self.play(GrowArrow(arrow), run_time=0.15)

        self.wait(0.5)

        # ==================================================================
        # Show computed values propagating forward
        # ==================================================================
        # Flash nodes as values are computed
        compute_order = ["a", "b", "d", "mul", "c", "add", "e"]
        values = {"a": 3, "b": 4, "d": 5}
        values["mul"] = values["a"] * values["b"]  # 12
        values["c"] = values["mul"]  # 12
        values["add"] = values["c"] + values["d"]  # 17
        values["e"] = values["add"]  # 17

        for name in compute_order:
            _, color, _, _ = node_info[name]
            self.play(
                node_objects[name].animate.set_fill(color, opacity=0.6),
                run_time=0.15,
            )

        self.wait(0.4)

        # ==================================================================
        # BACKWARD PASS - Gradient flows backward
        # ==================================================================
        # Fade forward label, show backward label
        backward_label = Text("Backward Pass (Autograd)", font_size=22, color="#ff6b6b")
        backward_label.move_to(UP * 2.5)
        self.play(Transform(phase_label, backward_label), run_time=0.4)

        # Reset node colors
        for name in node_objects:
            _, color, _, _ = node_info[name]
            node_objects[name].set_fill(color, opacity=0.15)

        # Gradient annotations
        grad_labels = {}
        grad_values = {
            "e":   "de/de = 1",
            "add": "grad = 1",
            "c":   "grad = 1",
            "d":   "grad = 1",
            "mul": "grad = 1",
            "a":   "grad = b = 4",
            "b":   "grad = a = 3",
        }

        backward_order = ["e", "add", "c", "d", "mul", "a", "b"]

        # Backward edge order (reversed arrows)
        backward_edges = [
            ("e", "add"),
            ("add", "c"),
            ("add", "d"),
            ("c", "mul"),
            ("mul", "a"),
            ("mul", "b"),
        ]

        # Animate gradient flowing backward
        for name in backward_order:
            _, color, _, _ = node_info[name]

            # Light up the node in red/orange tint
            self.play(
                node_objects[name].animate.set_fill("#ff6b6b", opacity=0.5).set_stroke("#ff6b6b", width=3),
                run_time=0.25,
            )

            # Show gradient label
            grad_text = Text(grad_values[name], font_size=11, color="#ffd93d")
            # Place above for top nodes, below for bottom nodes
            if positions[name][1] >= 0:
                grad_text.next_to(node_objects[name], UP, buff=0.08)
            else:
                grad_text.next_to(node_objects[name], DOWN, buff=0.08)

            grad_labels[name] = grad_text
            self.play(FadeIn(grad_text), run_time=0.15)

        self.wait(0.3)

        # Highlight the backward arrows in red
        for src, dst in backward_edges:
            # Find the forward arrow and flash it red
            if (dst, src) in arrow_objects:
                self.play(
                    arrow_objects[(dst, src)].animate.set_color("#ff6b6b"),
                    run_time=0.12,
                )

        self.wait(0.3)

        # Final insight
        insight_box = RoundedRectangle(
            width=8, height=0.7,
            corner_radius=0.1,
            fill_color="#2a2a4e", fill_opacity=0.5,
            stroke_color="#96ceb4", stroke_width=2,
        )
        insight_box.to_edge(DOWN, buff=0.25)
        insight_text = Text(
            "Chain rule: each node computes local gradient, multiplies by upstream",
            font_size=16, color="#96ceb4",
        )
        insight_text.move_to(insight_box.get_center())

        self.play(FadeIn(insight_box), Write(insight_text), run_time=0.6)

        self.wait(2.0)
