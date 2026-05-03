from manim import *
import numpy as np


class MCPFlow(Scene):
    """Visualizes MCP solving the N x M integration problem."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        # ── Title ──
        title = Text("MCP: The Integration Standard", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # ── Box definitions ──
        client_names = ["Claude", "GPT", "Gemini"]
        client_colors = ["#a78bfa", "#4ecdc4", "#4f9ef8"]
        tool_names = ["Slack", "GitHub", "Postgres", "Drive"]
        tool_colors = ["#ff6b6b", "#ffd93d", "#2ecc71", "#f08c4b"]

        box_w, box_h, corner_r = 1.3, 0.5, 0.1
        left_x, right_x = -4.5, 4.5

        def make_box(label, color, pos):
            rect = RoundedRectangle(
                width=box_w, height=box_h, corner_radius=corner_r,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=2,
            )
            rect.move_to(pos)
            txt = Text(label, font_size=18, color=color, weight=BOLD)
            txt.move_to(pos)
            return VGroup(rect, txt)

        # Position clients (3 boxes, centered vertically)
        client_positions = [
            np.array([left_x, 1.2, 0]),
            np.array([left_x, 0.0, 0]),
            np.array([left_x, -1.2, 0]),
        ]
        # Position tools (4 boxes, centered vertically)
        tool_positions = [
            np.array([right_x, 1.8, 0]),
            np.array([right_x, 0.6, 0]),
            np.array([right_x, -0.6, 0]),
            np.array([right_x, -1.8, 0]),
        ]

        client_boxes = [make_box(n, c, p) for n, c, p in zip(client_names, client_colors, client_positions)]
        tool_boxes = [make_box(n, c, p) for n, c, p in zip(tool_names, tool_colors, tool_positions)]

        # Fade in all boxes
        self.play(
            *[FadeIn(b) for b in client_boxes],
            *[FadeIn(b) for b in tool_boxes],
            run_time=0.5,
        )

        # ── Before MCP: 12 tangled lines ──
        tangled_lines = []
        # Varying arc angles to create crossing/chaos
        arc_angles = [
            0.6, -0.4, 0.8, -0.7,
            -0.5, 0.9, -0.3, 0.5,
            0.7, -0.8, 0.4, -0.6,
        ]
        tangle_color = "#885544"
        idx = 0
        for i, c_pos in enumerate(client_positions):
            for j, t_pos in enumerate(tool_positions):
                start = c_pos + RIGHT * (box_w / 2)
                end = t_pos + LEFT * (box_w / 2)
                line = ArcBetweenPoints(
                    start, end,
                    angle=arc_angles[idx],
                    color=tangle_color,
                    stroke_width=1.5,
                    stroke_opacity=0.7,
                )
                tangled_lines.append(line)
                idx += 1

        # Animate lines in rapid groups of 3
        for batch_start in range(0, 12, 3):
            batch = tangled_lines[batch_start:batch_start + 3]
            self.play(*[Create(l) for l in batch], run_time=0.1)

        # Label: "N x M = 12 custom integrations"
        before_label = VGroup(
            Text("N x M = ", font_size=18, color=GREY_B),
            Text("12", font_size=18, color="#ff6b6b", weight=BOLD),
            Text(" custom integrations", font_size=18, color=GREY_B),
        ).arrange(RIGHT, buff=0.08)
        before_label.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(before_label), run_time=0.3)
        self.wait(0.5)

        # ── Transition: fade tangled lines, grow MCP bar ──
        self.play(
            *[FadeOut(l) for l in tangled_lines],
            FadeOut(before_label),
            run_time=0.5,
        )

        # MCP bar in center
        mcp_bar = RoundedRectangle(
            width=0.8, height=4.2, corner_radius=0.12,
            fill_color="#fbbf24", fill_opacity=0.8,
            stroke_color="#fbbf24", stroke_width=2,
        )
        mcp_bar.move_to(ORIGIN)
        mcp_label = Text("MCP", font_size=22, color="#1a1a2e", weight=BOLD)
        mcp_label.move_to(ORIGIN)

        # Grow from zero height
        mcp_bar.save_state()
        mcp_bar.stretch(0.01, 1)
        mcp_label.set_opacity(0)
        self.play(
            Restore(mcp_bar),
            mcp_label.animate.set_opacity(1),
            run_time=0.5,
        )

        # ── After MCP: 7 clean lines ──
        clean_lines = []
        mcp_left_x = -0.4  # left edge of MCP bar
        mcp_right_x = 0.4  # right edge of MCP bar

        # 3 lines from clients to MCP bar
        for i, c_pos in enumerate(client_positions):
            start = c_pos + RIGHT * (box_w / 2)
            end = np.array([mcp_left_x, c_pos[1], 0])
            line = Line(start, end, color=client_colors[i], stroke_width=2, stroke_opacity=0.9)
            clean_lines.append(line)

        # 4 lines from MCP bar to tools
        for j, t_pos in enumerate(tool_positions):
            start = np.array([mcp_right_x, t_pos[1], 0])
            end = t_pos + LEFT * (box_w / 2)
            line = Line(start, end, color=tool_colors[j], stroke_width=2, stroke_opacity=0.9)
            clean_lines.append(line)

        # Animate clean lines
        for line in clean_lines:
            self.play(Create(line), run_time=0.15)

        # Label: "N + M = 7 standard connections"
        after_label = VGroup(
            Text("N + M = ", font_size=18, color=GREY_B),
            Text("7", font_size=18, color="#2ecc71", weight=BOLD),
            Text(" standard connections", font_size=18, color=GREY_B),
        ).arrange(RIGHT, buff=0.08)
        after_label.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(after_label), run_time=0.3)
        self.wait(0.3)

        # ── Punchline ──
        punchline = Text(
            "Write once, works with any MCP-compatible client",
            font_size=16, color=GREY_B,
        )
        punchline.next_to(after_label, UP, buff=0.25)
        self.play(Write(punchline), run_time=0.5)
        self.wait(2.0)
