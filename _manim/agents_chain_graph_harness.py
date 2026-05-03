from manim import *
import numpy as np


class ChainGraphHarness(Scene):
    """Compares three agent architecture patterns: Chain, Graph, and Harness."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Three Architectures", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # ── PANEL 1: CHAIN ──────────────────────────────────────────────
        chain_panel = self.build_chain_panel()
        self.play(FadeIn(chain_panel), run_time=0.4)
        self.wait(0.2)

        # Green dot flows linearly through the chain
        chain_dot = Dot(color="#2ecc71", radius=0.08)
        chain_positions = [box.get_center() for box in chain_panel[1]]  # node group
        start = chain_positions[0] + LEFT * 2
        end = chain_positions[-1] + RIGHT * 2
        chain_dot.move_to(start)
        self.play(FadeIn(chain_dot), run_time=0.15)

        # Flow: enter -> Parse -> Embed -> Respond -> exit
        for pos in chain_positions:
            self.play(chain_dot.animate.move_to(pos), run_time=0.35)
            self.wait(0.1)
        self.play(chain_dot.animate.move_to(end), run_time=0.3)
        self.play(FadeOut(chain_dot), run_time=0.15)
        self.wait(0.3)

        # ── PANEL 2: GRAPH ──────────────────────────────────────────────
        graph_panel, graph_node_map = self.build_graph_panel()
        self.play(
            chain_panel.animate.set_opacity(0.0),
            FadeIn(graph_panel),
            run_time=0.4,
        )
        self.remove(chain_panel)

        # Amber dot traverses the graph with a cycle
        graph_dot = Dot(color="#fbbf24", radius=0.08)
        route_pos = graph_node_map["Route"]
        search_pos = graph_node_map["Search"]
        generate_pos = graph_node_map["Generate"]
        check_pos = graph_node_map["Check"]

        graph_dot.move_to(route_pos + UP * 0.8)
        self.play(FadeIn(graph_dot), run_time=0.15)

        # Path: enter Route -> Search -> Check -> Route (cycle!) -> Generate -> Check -> exit
        path = [
            route_pos, search_pos, check_pos,
            route_pos,  # cycle back
            generate_pos, check_pos,
        ]
        for pos in path:
            self.play(graph_dot.animate.move_to(pos), run_time=0.3)
            self.wait(0.08)

        exit_pos = check_pos + DOWN * 0.8
        self.play(graph_dot.animate.move_to(exit_pos), run_time=0.25)
        self.play(FadeOut(graph_dot), run_time=0.15)
        self.wait(0.3)

        # ── PANEL 3: HARNESS ────────────────────────────────────────────
        harness_panel, llm_box, tool_boxes = self.build_harness_panel()
        self.play(
            graph_panel.animate.set_opacity(0.0),
            FadeIn(harness_panel),
            run_time=0.4,
        )
        self.remove(graph_panel)

        # LLM pulses while dynamically reaching out to tools
        pulse = llm_box.animate.set_fill(opacity=0.45)
        self.play(pulse, run_time=0.3)
        self.play(llm_box.animate.set_fill(opacity=0.25), run_time=0.3)

        # Reach out to tools one/two at a time
        tool_order = [
            [tool_boxes[0]],          # Search
            [tool_boxes[1]],          # Code
            [tool_boxes[2], tool_boxes[3]],  # Files + API simultaneously
            [tool_boxes[4]],          # Browser
        ]
        for tool_group in tool_order:
            arrows = []
            for tb in tool_group:
                arr = Arrow(
                    start=llm_box.get_center(),
                    end=tb.get_center(),
                    color="#2ecc71",
                    stroke_width=2,
                    buff=0.35,
                    max_tip_length_to_length_ratio=0.15,
                )
                arrows.append(arr)
            self.play(*[GrowArrow(a) for a in arrows], run_time=0.3)
            self.wait(0.15)
            self.play(*[FadeOut(a) for a in arrows], run_time=0.2)

        self.wait(0.3)

        # ── FINAL SUMMARY ──────────────────────────────────────────────
        self.play(
            FadeOut(harness_panel),
            run_time=0.4,
        )

        fixed_path = Text("Fixed path", font_size=22, color="#4f9ef8")
        fixed_topo = Text("Fixed topology", font_size=22, color="#a78bfa")
        no_fixed = Text("No fixed structure", font_size=22, color="#2ecc71")

        summary = VGroup(fixed_path, fixed_topo, no_fixed)
        summary.arrange(RIGHT, buff=1.5)
        summary.move_to(ORIGIN)

        # Thin separator lines
        sep1 = Line(UP * 0.3, DOWN * 0.3, color=GREY_B, stroke_width=1)
        sep1.move_to((fixed_path.get_right() + fixed_topo.get_left()) / 2)
        sep2 = Line(UP * 0.3, DOWN * 0.3, color=GREY_B, stroke_width=1)
        sep2.move_to((fixed_topo.get_right() + no_fixed.get_left()) / 2)

        self.play(FadeIn(summary), FadeIn(sep1), FadeIn(sep2), run_time=0.4)
        self.wait(2.0)

    # ── PANEL BUILDERS ──────────────────────────────────────────────────

    def make_node(self, label, color, width=1.2, height=0.5, font_size=16):
        rect = RoundedRectangle(
            width=width, height=height, corner_radius=0.1,
            fill_color=color, fill_opacity=0.2,
            stroke_color=color, stroke_width=2,
        )
        txt = Text(label, font_size=font_size, color=color, weight=BOLD)
        txt.move_to(rect.get_center())
        return VGroup(rect, txt)

    def build_chain_panel(self):
        header = Text("CHAIN", font_size=26, color="#4f9ef8", weight=BOLD)
        header.move_to(UP * 2.2 + LEFT * 4.5)

        labels = ["Parse", "Embed", "Respond"]
        color = "#4f9ef8"
        nodes = VGroup()
        for label in labels:
            nodes.add(self.make_node(label, color))
        nodes.arrange(RIGHT, buff=1.2)
        nodes.move_to(ORIGIN + UP * 0.3)

        arrows = VGroup()
        for i in range(len(labels) - 1):
            arr = Arrow(
                start=nodes[i].get_right(),
                end=nodes[i + 1].get_left(),
                color=GREY_B, stroke_width=2, buff=0.08,
                max_tip_length_to_length_ratio=0.2,
            )
            arrows.add(arr)

        subtitle = Text(
            "Developer defines every step",
            font_size=16, color=GREY_B,
        )
        subtitle.move_to(DOWN * 1.5)

        return VGroup(header, nodes, arrows, subtitle)

    def build_graph_panel(self):
        header = Text("GRAPH", font_size=26, color="#a78bfa", weight=BOLD)
        header.move_to(UP * 2.2 + LEFT * 4.5)

        color = "#a78bfa"
        route = self.make_node("Route", color)
        search = self.make_node("Search", color)
        generate = self.make_node("Generate", color)
        check = self.make_node("Check", color)

        # Position: Route top center, Search bottom-left, Generate bottom-right, Check bottom center
        route.move_to(UP * 1.0)
        search.move_to(LEFT * 2.0 + DOWN * 0.3)
        generate.move_to(RIGHT * 2.0 + DOWN * 0.3)
        check.move_to(DOWN * 1.5)

        node_group = VGroup(route, search, generate, check)

        # State box in center
        state_box = RoundedRectangle(
            width=0.8, height=0.35, corner_radius=0.05,
            fill_color="#fbbf24", fill_opacity=0.15,
            stroke_color="#fbbf24", stroke_width=1.5,
        )
        state_txt = Text("State", font_size=12, color="#fbbf24")
        state_txt.move_to(state_box.get_center())
        state_group = VGroup(state_box, state_txt)
        state_group.move_to(DOWN * 0.3)

        # Thin lines from state to each node
        state_lines = VGroup()
        for node in [route, search, generate, check]:
            ln = DashedLine(
                state_group.get_center(), node.get_center(),
                color="#fbbf24", stroke_width=1, stroke_opacity=0.4,
                dash_length=0.08,
            )
            state_lines.add(ln)

        # Conditional edges from Route (dashed)
        edge_to_search = DashedLine(
            route.get_bottom(), search.get_top(),
            color=GREY_B, stroke_width=2, dash_length=0.1,
        )
        lbl_query = Text("if query", font_size=10, color=GREY_B)
        lbl_query.move_to((route.get_bottom() + search.get_top()) / 2 + LEFT * 0.5)

        edge_to_generate = DashedLine(
            route.get_bottom(), generate.get_top(),
            color=GREY_B, stroke_width=2, dash_length=0.1,
        )
        lbl_command = Text("if command", font_size=10, color=GREY_B)
        lbl_command.move_to((route.get_bottom() + generate.get_top()) / 2 + RIGHT * 0.6)

        # Solid edges: Search -> Check, Generate -> Check, Check -> Route (cycle)
        edge_search_check = Arrow(
            search.get_bottom(), check.get_left(),
            color=GREY_B, stroke_width=2, buff=0.08,
            max_tip_length_to_length_ratio=0.15,
        )
        edge_gen_check = Arrow(
            generate.get_bottom(), check.get_right(),
            color=GREY_B, stroke_width=2, buff=0.08,
            max_tip_length_to_length_ratio=0.15,
        )
        # Cycle edge: Check back to Route (curved)
        cycle_arrow = CurvedArrow(
            check.get_left() + LEFT * 0.1,
            route.get_left() + LEFT * 0.1,
            color="#ff6b6b", stroke_width=2,
            angle=-TAU / 4,
        )
        cycle_label = Text("cycle", font_size=10, color="#ff6b6b")
        cycle_label.next_to(cycle_arrow, LEFT, buff=0.1)

        edges = VGroup(
            edge_to_search, lbl_query,
            edge_to_generate, lbl_command,
            edge_search_check, edge_gen_check,
            cycle_arrow, cycle_label,
        )

        subtitle = Text(
            "Developer defines topology, model steers within it",
            font_size=16, color=GREY_B,
        )
        subtitle.move_to(DOWN * 2.5)

        panel = VGroup(header, node_group, state_group, state_lines, edges, subtitle)

        # Build position map for dot animation
        node_map = {
            "Route": route.get_center(),
            "Search": search.get_center(),
            "Generate": generate.get_center(),
            "Check": check.get_center(),
        }
        return panel, node_map

    def build_harness_panel(self):
        header = Text("HARNESS", font_size=26, color="#2ecc71", weight=BOLD)
        header.move_to(UP * 2.2 + LEFT * 4.5)

        # Central LLM box (larger)
        llm_rect = RoundedRectangle(
            width=1.5, height=0.7, corner_radius=0.1,
            fill_color="#2ecc71", fill_opacity=0.25,
            stroke_color="#2ecc71", stroke_width=2.5,
        )
        llm_txt = Text("LLM", font_size=20, color="#2ecc71", weight=BOLD)
        llm_txt.move_to(llm_rect.get_center())
        llm_box = VGroup(llm_rect, llm_txt)
        llm_box.move_to(UP * 0.2)

        # Tool boxes in a loose cloud
        tool_labels = ["Search", "Code", "Files", "API", "Browser"]
        tool_color = GREY_B
        angles = [PI / 2 + i * (2 * PI / 5) for i in range(5)]
        cloud_radius = 2.0
        center = llm_box.get_center()

        tool_boxes = []
        tool_group = VGroup()
        for i, (label, angle) in enumerate(zip(tool_labels, angles)):
            node = self.make_node(label, tool_color, width=0.9, height=0.4, font_size=13)
            pos = center + cloud_radius * np.array([np.cos(angle), np.sin(angle), 0])
            node.move_to(pos)
            tool_boxes.append(node)
            tool_group.add(node)

        subtitle = Text(
            "Model drives everything, framework provides tools",
            font_size=16, color=GREY_B,
        )
        subtitle.move_to(DOWN * 2.5)

        panel = VGroup(header, llm_box, tool_group, subtitle)
        return panel, llm_rect, tool_boxes
