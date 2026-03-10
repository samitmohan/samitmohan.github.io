from manim import *
import numpy as np


class Backpropagation(Scene):
    """Backpropagation flow through a computation graph.

    Shows forward pass (blue, left-to-right) and backward pass (red, right-to-left)
    with partial derivative labels and chain rule at each node.
    """

    def construct(self):
        self.camera.background_color = WHITE

        title = Text("Backpropagation", font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # -- Computation Graph Nodes --
        # x -> [*w] -> [+b] -> [L2 loss]
        # Computation: z = w*x, a = z + b, L = a^2

        node_radius = 0.5
        node_color = BLUE_E
        node_positions = [
            LEFT * 5,      # x (input)
            LEFT * 2,      # multiply by w
            RIGHT * 1,     # add b
            RIGHT * 4,     # loss (square)
        ]

        node_labels = ["x", "* w", "+ b", "L=a^2"]
        node_values_fwd = ["2", "2*3=6", "6+1=7", "7^2=49"]

        nodes = VGroup()
        labels = VGroup()
        for i, (pos, label_text) in enumerate(zip(node_positions, node_labels)):
            if i == 0:
                # Input is a simple circle
                node = Circle(radius=node_radius, color=node_color,
                              fill_color=BLUE_A, fill_opacity=0.3, stroke_width=2)
            elif i == 3:
                # Loss node is slightly different
                node = Circle(radius=node_radius, color=RED_D,
                              fill_color=RED_A, fill_opacity=0.3, stroke_width=2)
            else:
                node = Circle(radius=node_radius, color=node_color,
                              fill_color=BLUE_A, fill_opacity=0.3, stroke_width=2)
            node.move_to(pos)
            label = Text(label_text, font_size=20, color=BLACK)
            label.move_to(pos)
            nodes.add(node)
            labels.add(label)

        # Edges (forward arrows)
        edges = VGroup()
        for i in range(len(node_positions) - 1):
            start = node_positions[i] + RIGHT * node_radius
            end = node_positions[i + 1] + LEFT * node_radius
            edge = Arrow(start, end, color=BLUE_D, buff=0.05,
                         stroke_width=3, max_tip_length_to_length_ratio=0.15)
            edges.add(edge)

        # Variable names above edges
        var_names = ["z = w*x", "a = z+b", "L = a^2"]
        var_labels = VGroup()
        for i, name in enumerate(var_names):
            mid = (node_positions[i] + node_positions[i + 1]) / 2 + UP * 0.6
            vl = Text(name, font_size=16, color=GREY_D)
            vl.move_to(mid)
            var_labels.add(vl)

        # Parameter labels below nodes
        param_text = ["x=2", "w=3", "b=1", ""]
        param_labels = VGroup()
        for i, pt in enumerate(param_text):
            if pt:
                pl = Text(pt, font_size=16, color=PURPLE_D)
                pl.next_to(nodes[i], DOWN, buff=0.3)
                param_labels.add(pl)

        self.play(
            *[Create(n) for n in nodes],
            *[Write(l) for l in labels],
            *[Create(e) for e in edges],
            *[Write(vl) for vl in var_labels],
            *[Write(pl) for pl in param_labels],
            run_time=1.5,
        )
        self.wait(0.3)

        # ===================== FORWARD PASS =====================
        fwd_title = Text("Forward Pass", font_size=28, color=BLUE_D)
        fwd_title.to_edge(DOWN, buff=0.3)
        self.play(Write(fwd_title), run_time=0.4)

        # Show values flowing through
        fwd_values = ["2", "6", "7", "49"]
        value_mobs = VGroup()
        for i, val in enumerate(fwd_values):
            v = Text(val, font_size=22, color=BLUE_D, weight=BOLD)
            v.next_to(nodes[i], UP, buff=0.6)
            value_mobs.add(v)

        # Animate forward flow with a traveling dot
        for i in range(len(nodes)):
            if i > 0:
                # Moving pulse along edge
                pulse = Dot(color=BLUE, radius=0.08)
                start = node_positions[i - 1] + RIGHT * node_radius
                end = node_positions[i] + LEFT * node_radius
                pulse.move_to(start)
                self.play(
                    pulse.animate.move_to(end),
                    run_time=0.3,
                )
                self.remove(pulse)

            # Show computed value
            self.play(Write(value_mobs[i]), run_time=0.3)
            # Flash the node
            self.play(
                nodes[i].animate.set_fill(BLUE, opacity=0.6),
                run_time=0.15,
            )
            self.play(
                nodes[i].animate.set_fill(BLUE_A, opacity=0.3),
                run_time=0.15,
            )

        self.wait(0.3)
        self.play(FadeOut(fwd_title), run_time=0.3)

        # ===================== BACKWARD PASS =====================
        bwd_title = Text("Backward Pass (Chain Rule)", font_size=28, color=RED_D)
        bwd_title.to_edge(DOWN, buff=0.3)
        self.play(Write(bwd_title), run_time=0.4)

        # Backward edges (red, right to left)
        back_edges = VGroup()
        for i in range(len(node_positions) - 1, 0, -1):
            start = node_positions[i] + LEFT * node_radius + DOWN * 0.15
            end = node_positions[i - 1] + RIGHT * node_radius + DOWN * 0.15
            be = Arrow(start, end, color=RED_D, buff=0.05,
                       stroke_width=3, max_tip_length_to_length_ratio=0.15)
            back_edges.add(be)

        self.play(*[Create(be) for be in back_edges], run_time=0.5)

        # Gradients: dL/da = 2a = 14, dL/dz = dL/da * da/dz = 14*1 = 14, dL/dx = dL/dz * dz/dx = 14*3 = 42
        grad_labels = [
            "dL/da = 2a = 14",
            "dL/dz = 14 * 1 = 14",
            "dL/dx = 14 * 3 = 42",
        ]
        partial_labels = [
            "da/dz = 1",
            "dz/dx = w = 3",
        ]

        # Show gradients flowing backward
        grad_mobs = VGroup()
        for i, (gl, be) in enumerate(zip(grad_labels, back_edges)):
            g = Text(gl, font_size=16, color=RED_D)
            # Position below the backward edge
            mid_pos = be.get_center() + DOWN * 0.5
            g.move_to(mid_pos)
            grad_mobs.add(g)

            # Backward pulse
            pulse = Dot(color=RED, radius=0.08)
            pulse.move_to(be.get_start())
            self.play(
                pulse.animate.move_to(be.get_end()),
                Write(g),
                run_time=0.4,
            )
            self.remove(pulse)

            # Flash the receiving node
            target_idx = len(node_positions) - 2 - i
            self.play(
                nodes[target_idx].animate.set_fill(RED, opacity=0.5),
                run_time=0.15,
            )
            self.play(
                nodes[target_idx].animate.set_fill(BLUE_A, opacity=0.3),
                run_time=0.15,
            )

        # Show partial derivative labels on edges
        for i, pl_text in enumerate(partial_labels):
            pl = Text(pl_text, font_size=14, color=ORANGE)
            edge_mid = back_edges[i + 1].get_center() if i + 1 < len(back_edges) else back_edges[i].get_center()
            pl.next_to(edge_mid, DOWN, buff=0.4)

        # Chain rule summary
        chain = Text(
            "Chain Rule: dL/dx = dL/da * da/dz * dz/dx = 14 * 1 * 3 = 42",
            font_size=18, color=BLACK, weight=BOLD,
        )
        chain.to_edge(DOWN, buff=0.1)
        self.play(FadeOut(bwd_title), run_time=0.2)
        self.play(Write(chain), run_time=0.8)
        self.wait(1)
