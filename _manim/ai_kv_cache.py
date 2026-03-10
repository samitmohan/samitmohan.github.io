from manimlib import *
import numpy as np


class KVCacheGrowth(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        tokens = ["I", "am", "going", "home"]
        token_colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#ffd93d"]

        # --- Title ---
        title = Text(
            "KV Cache: Why It Grows Quadratically",
            font_size=30, color=WHITE, weight=BOLD
        ).to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # Layout: left side = attention matrix, right side = KV cache
        left_anchor = LEFT * 3.2
        right_anchor = RIGHT * 2.8

        # Labels
        attn_label = Text("Attention Matrix", font_size=20, color="#dda0dd")
        attn_label.move_to(left_anchor + UP * 2.2)

        kv_label = Text("KV Cache", font_size=20, color="#a8e6cf")
        kv_label.move_to(right_anchor + UP * 2.2)

        self.play(FadeIn(attn_label), FadeIn(kv_label), run_time=0.4)

        # Token display at top center
        token_display = VGroup()
        token_display_label = Text("Tokens:", font_size=18, color=GREY_B)
        token_display_label.move_to(UP * 2.5 + LEFT * 0.2)
        self.play(FadeIn(token_display_label), run_time=0.3)

        # Size indicator
        size_text = None

        # Track current objects for replacement
        current_matrix = None
        current_token_row = None
        current_kv_group = None

        def make_attention_matrix(n, highlight_new=True):
            """Build an n x n attention matrix with cells."""
            cell_size = min(0.6, 2.4 / max(n, 1))
            cells = VGroup()
            for i in range(n):
                for j in range(n):
                    is_new = (i == n - 1 or j == n - 1) and highlight_new
                    fill_col = "#ff6b6b" if is_new else "#2a2a4e"
                    stroke_col = "#ff6b6b" if is_new else "#555580"
                    fill_op = 0.5 if is_new else 0.3
                    rect = Rectangle(
                        width=cell_size, height=cell_size,
                        stroke_color=stroke_col, stroke_width=1.5,
                        fill_color=fill_col, fill_opacity=fill_op,
                    )
                    cells.add(rect)
            grid = cells.arrange_in_grid(n_rows=n, n_cols=n, buff=0.04)
            return grid

        def make_kv_cache(n):
            """Build KV cache visualization: two columns of cached vectors."""
            k_col_label = Text("Keys", font_size=16, color="#4ecdc4", weight=BOLD)
            v_col_label = Text("Values", font_size=16, color="#45b7d1", weight=BOLD)

            row_h = 0.35
            row_w = 1.0
            k_cells = VGroup()
            v_cells = VGroup()

            for i in range(n):
                is_new = (i == n - 1)
                # Key cell
                k_rect = Rectangle(
                    width=row_w, height=row_h,
                    stroke_color="#4ecdc4", stroke_width=1.5,
                    fill_color="#4ecdc4",
                    fill_opacity=0.5 if is_new else 0.2,
                )
                k_text = Text(tokens[i], font_size=14, color=WHITE)
                k_text.move_to(k_rect.get_center())
                k_cells.add(VGroup(k_rect, k_text))

                # Value cell
                v_rect = Rectangle(
                    width=row_w, height=row_h,
                    stroke_color="#45b7d1", stroke_width=1.5,
                    fill_color="#45b7d1",
                    fill_opacity=0.5 if is_new else 0.2,
                )
                v_text = Text(tokens[i], font_size=14, color=WHITE)
                v_text.move_to(v_rect.get_center())
                v_cells.add(VGroup(v_rect, v_text))

            k_cells.arrange(DOWN, buff=0.06)
            v_cells.arrange(DOWN, buff=0.06)

            k_col_label.next_to(k_cells, UP, buff=0.15)
            v_col_label.next_to(v_cells, UP, buff=0.15)

            k_group = VGroup(k_col_label, k_cells)
            v_group = VGroup(v_col_label, v_cells)

            cache = VGroup(k_group, v_group).arrange(RIGHT, buff=0.4)
            return cache

        for step, token in enumerate(tokens):
            n = step + 1

            # Update token display
            new_tok = Text(token, font_size=22, color=token_colors[step])
            token_display.add(new_tok)
            arranged_tokens = token_display.copy().arrange(RIGHT, buff=0.3)
            arranged_tokens.move_to(UP * 2.5 + RIGHT * 1.2)

            # Build attention matrix
            new_matrix = make_attention_matrix(n, highlight_new=(step > 0))
            new_matrix.move_to(left_anchor + DOWN * 0.3)

            # Row/col labels for the matrix
            label_group = VGroup()
            cell_size = min(0.6, 2.4 / max(n, 1))
            for i in range(n):
                # Row label (left side)
                rl = Text(tokens[i], font_size=12, color=token_colors[i])
                row_y = new_matrix[i * n].get_center()[1]
                rl.move_to(
                    new_matrix.get_left() + LEFT * 0.35 + UP * (row_y - new_matrix.get_center()[1])
                )
                label_group.add(rl)

                # Col label (top)
                cl = Text(tokens[i], font_size=12, color=token_colors[i])
                col_x = new_matrix[i].get_center()[0]
                cl.move_to(
                    new_matrix.get_top() + UP * 0.25 + RIGHT * (col_x - new_matrix.get_center()[0])
                )
                label_group.add(cl)

            matrix_with_labels = VGroup(new_matrix, label_group)

            # Size text
            new_size = Text(f"{n}x{n}", font_size=20, color="#ffd93d")
            new_size.next_to(new_matrix, DOWN, buff=0.25)

            # Build KV cache
            new_kv = make_kv_cache(n)
            new_kv.move_to(right_anchor + DOWN * 0.3)

            # Animate
            if step == 0:
                # First token: fade everything in
                self.play(
                    FadeIn(arranged_tokens),
                    FadeIn(matrix_with_labels, shift=UP * 0.2),
                    FadeIn(new_size),
                    FadeIn(new_kv, shift=UP * 0.2),
                    run_time=0.7,
                )
                current_matrix = matrix_with_labels
                current_token_row = arranged_tokens
                current_kv_group = new_kv
                size_text = new_size
            else:
                # Subsequent tokens: transform
                self.play(
                    ReplacementTransform(current_token_row, arranged_tokens),
                    ReplacementTransform(current_matrix, matrix_with_labels),
                    ReplacementTransform(size_text, new_size),
                    ReplacementTransform(current_kv_group, new_kv),
                    run_time=0.7,
                )
                current_matrix = matrix_with_labels
                current_token_row = arranged_tokens
                current_kv_group = new_kv
                size_text = new_size

            self.wait(0.4)

        # --- Growth summary ---
        self.wait(0.3)

        # Show growth: 1 -> 4 -> 9 -> 16
        growth_text = Text(
            "Matrix cells: 1 -> 4 -> 9 -> 16  (n^2 growth)",
            font_size=20, color="#ffd93d"
        )
        growth_text.next_to(size_text, DOWN, buff=0.3)
        self.play(Write(growth_text), run_time=0.6)
        self.wait(0.5)

        # --- Final comparison text ---
        self.play(
            FadeOut(current_matrix),
            FadeOut(current_token_row),
            FadeOut(current_kv_group),
            FadeOut(size_text),
            FadeOut(growth_text),
            FadeOut(attn_label),
            FadeOut(kv_label),
            FadeOut(token_display_label),
            run_time=0.5,
        )

        without_text = Text(
            "Without cache: recompute all K,V every step",
            font_size=22, color="#ff6b6b"
        )
        with_text = Text(
            "With cache: store and reuse",
            font_size=22, color="#a8e6cf"
        )

        comparison = VGroup(without_text, with_text).arrange(DOWN, buff=0.4)
        comparison.move_to(ORIGIN)

        self.play(Write(without_text), run_time=0.6)
        self.wait(0.2)
        self.play(Write(with_text), run_time=0.6)
        self.wait(1.5)
