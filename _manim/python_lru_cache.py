from manim import *


class LRUCache(Scene):
    """Visualizes fibonacci with and without @lru_cache - call tree comparison."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("@lru_cache: Fibonacci", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # --- Layout: two columns ---
        left_title = Text("Without cache", font_size=18, color="#ff6b6b", weight=BOLD)
        right_title = Text("With @lru_cache", font_size=18, color="#a8e6cf", weight=BOLD)
        left_title.move_to(LEFT * 3.5 + UP * 2.3)
        right_title.move_to(RIGHT * 1.8 + UP * 2.3)
        self.play(FadeIn(left_title), FadeIn(right_title), run_time=0.4)

        # Divider
        divider = DashedLine(
            UP * 2.5, DOWN * 3.5,
            color=GREY_D, stroke_width=1, dash_length=0.1
        )
        divider.move_to(LEFT * 0.5)
        self.play(Create(divider), run_time=0.3)

        # ============================================================
        # LEFT SIDE: Naive fib(5) call tree (exponential)
        # Show partial tree to illustrate explosion of repeated calls
        # ============================================================
        def make_node(label, color, pos, font_size=13, w=0.55, h=0.35):
            box = RoundedRectangle(
                width=w, height=h, corner_radius=0.06,
                fill_color=color, fill_opacity=0.2,
                stroke_color=color, stroke_width=1.5
            )
            box.move_to(pos)
            txt = Text(label, font_size=font_size, color=color)
            txt.move_to(pos)
            return VGroup(box, txt)

        def make_edge(start_pos, end_pos, color=GREY_B):
            return Line(
                start_pos, end_pos,
                color=color, stroke_width=1.2
            )

        # Left tree layout for fib(5) - positions manually placed
        # Only show enough to demonstrate the explosion + duplicates
        left_anchor = LEFT * 3.5
        ly = UP * 1.6

        # Tree structure (abbreviated but showing key duplicates)
        # Level 0: fib(5)
        # Level 1: fib(4), fib(3)
        # Level 2: fib(3), fib(2), fib(2), fib(1)
        # Level 3: fib(2), fib(1), fib(1), fib(0), fib(1), fib(0)
        # Level 4: fib(1), fib(0)

        left_nodes_data = [
            # (label, x_offset, y_offset, is_duplicate)
            ("f(5)", 0, 0, False),
            ("f(4)", -1.2, -0.7, False),
            ("f(3)", 1.2, -0.7, True),
            ("f(3)", -1.9, -1.4, True),
            ("f(2)", -0.5, -1.4, True),
            ("f(2)", 0.6, -1.4, True),
            ("f(1)", 1.8, -1.4, False),
            ("f(2)", -2.3, -2.1, True),
            ("f(1)", -1.5, -2.1, False),
            ("f(1)", -0.8, -2.1, False),
            ("f(0)", -0.2, -2.1, False),
            ("f(1)", 0.3, -2.1, False),
            ("f(0)", 0.9, -2.1, False),
            ("f(1)", -2.5, -2.8, False),
            ("f(0)", -2.0, -2.8, False),
        ]

        # Edges: (parent_idx, child_idx)
        left_edges_data = [
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5), (2, 6),
            (3, 7), (3, 8),
            (4, 9), (4, 10),
            (5, 11), (5, 12),
            (7, 13), (7, 14),
        ]

        left_node_objs = []
        for label, xo, yo, is_dup in left_nodes_data:
            color = "#ff6b6b" if is_dup else "#45b7d1"
            pos = left_anchor + RIGHT * xo + ly + DOWN * (-yo)
            node = make_node(label, color, pos, font_size=11, w=0.5, h=0.3)
            left_node_objs.append((node, pos))

        left_edge_objs = []
        for pi, ci in left_edges_data:
            _, p_pos = left_node_objs[pi]
            _, c_pos = left_node_objs[ci]
            edge = make_edge(p_pos + DOWN * 0.15, c_pos + UP * 0.15)
            left_edge_objs.append(edge)

        # Animate left tree level by level
        levels = [
            [0],
            [1, 2],
            [3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14],
        ]
        edge_by_level = [
            [0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13],
        ]

        for li, level in enumerate(levels):
            node_anims = [FadeIn(left_node_objs[i][0]) for i in level]
            self.play(*node_anims, run_time=0.35)
            if li < len(edge_by_level):
                edge_anims = [Create(left_edge_objs[i]) for i in edge_by_level[li]]
                self.play(*edge_anims, run_time=0.2)

        # Duplicate count callout
        dup_note = Text("6 duplicate calls (red)", font_size=12, color="#ff6b6b")
        dup_note.move_to(left_anchor + DOWN * 3.3)
        self.play(FadeIn(dup_note), run_time=0.3)
        self.wait(0.3)

        # ============================================================
        # RIGHT SIDE: Cached fib(5) - linear chain + cache table
        # ============================================================
        right_anchor = RIGHT * 0.8
        ry = UP * 1.6

        # With cache, each fib(n) is computed once: fib(0)..fib(5) bottom-up
        # Show as a linear top-down chain
        cached_calls = ["f(5)", "f(4)", "f(3)", "f(2)", "f(1)", "f(0)"]
        cached_labels = ["miss", "miss", "miss", "miss", "miss", "miss"]
        cached_colors_call = ["#ffd93d"] * 6  # all yellow (miss) on first compute

        right_node_objs = []
        spacing = 0.6

        for i, (label, status) in enumerate(zip(cached_calls, cached_labels)):
            pos = right_anchor + ry + DOWN * (i * spacing)
            node = make_node(label, "#ffd93d", pos, font_size=13, w=0.6, h=0.35)
            right_node_objs.append((node, pos))

        right_edge_objs = []
        for i in range(len(cached_calls) - 1):
            _, p_pos = right_node_objs[i]
            _, c_pos = right_node_objs[i + 1]
            edge = make_edge(p_pos + DOWN * 0.18, c_pos + UP * 0.18, color="#ffd93d")
            right_edge_objs.append(edge)

        # Cache table on the far right
        cache_title = Text("Cache Table", font_size=14, color="#96ceb4", weight=BOLD)
        cache_title.move_to(RIGHT * 4.5 + UP * 1.8)
        self.play(FadeIn(cache_title), run_time=0.2)

        cache_entries = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 5}
        table_x = RIGHT * 4.5
        table_y_start = UP * 1.4
        row_spacing = 0.35

        cache_row_objs = []

        # Animate right tree: each call goes down, then returns up filling cache
        # First show the calls going down (all misses)
        for i in range(len(cached_calls)):
            node, pos = right_node_objs[i]
            anims = [FadeIn(node)]
            if i > 0:
                anims.append(Create(right_edge_objs[i - 1]))
            # Show "miss" tag
            miss_tag = Text("miss", font_size=10, color="#ffd93d")
            miss_tag.next_to(node, RIGHT, buff=0.12)
            anims.append(FadeIn(miss_tag))
            self.play(*anims, run_time=0.25)

        self.wait(0.2)

        # Now show cache filling bottom-up (base cases first)
        fill_order = [5, 4, 3, 2, 1, 0]  # index in right_node_objs (fib(0) is index 5)
        fib_values = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 5}

        for step, idx in enumerate(fill_order):
            n = 5 - idx  # actual fib number
            val = fib_values[n]
            node, pos = right_node_objs[idx]

            # Turn node green (computed)
            new_node = make_node(
                f"f({n})={val}", "#a8e6cf", pos,
                font_size=11, w=0.75, h=0.35
            )
            self.play(
                FadeOut(node),
                FadeIn(new_node),
                run_time=0.2
            )

            # Add to cache table
            row_y = table_y_start + DOWN * (step * row_spacing)
            key_text = Text(f"f({n})", font_size=12, color="#4ecdc4")
            arrow_text = Text(":", font_size=12, color=GREY_B)
            val_text = Text(str(val), font_size=12, color="#a8e6cf", weight=BOLD)
            row = VGroup(key_text, arrow_text, val_text).arrange(RIGHT, buff=0.1)
            row.move_to(table_x + row_y)

            row_bg = RoundedRectangle(
                width=row.width + 0.25, height=0.28,
                corner_radius=0.05,
                fill_color="#96ceb4", fill_opacity=0.1,
                stroke_color="#96ceb4", stroke_width=1
            )
            row_bg.move_to(row.get_center())
            cache_row_objs.append(VGroup(row_bg, row))
            self.play(FadeIn(row_bg), FadeIn(row), run_time=0.15)

        self.wait(0.3)

        # Show what happens on re-call: highlight cache hit
        hit_note = Text("Next call to f(3)?", font_size=14, color=WHITE)
        hit_note.move_to(RIGHT * 2.5 + DOWN * 2.8)
        self.play(FadeIn(hit_note), run_time=0.3)

        # Flash the f(3) cache row green
        # f(3) is the 4th entry added (step=3 in fill_order, which is f(2), f(1), f(0), then f(3) is step 3)
        # Actually fill_order gives: f(0), f(1), f(2), f(3), f(4), f(5) - step 3 = f(3)
        hit_row = cache_row_objs[3]  # f(3) row
        self.play(
            hit_row[0].animate.set_fill(color="#a8e6cf", opacity=0.5),
            run_time=0.3
        )

        hit_result = Text("Cache HIT - O(1)!", font_size=14, color="#a8e6cf", weight=BOLD)
        hit_result.next_to(hit_note, DOWN, buff=0.15)
        self.play(FadeIn(hit_result), run_time=0.3)

        self.wait(0.3)

        # --- Bottom comparison ---
        comparison = VGroup(
            Text("O(2^n) calls", font_size=16, color="#ff6b6b"),
            Text("  vs  ", font_size=16, color=GREY_B),
            Text("O(n) calls", font_size=16, color="#a8e6cf"),
        ).arrange(RIGHT, buff=0.1)
        comparison.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(comparison), run_time=0.4)

        self.wait(2.0)
