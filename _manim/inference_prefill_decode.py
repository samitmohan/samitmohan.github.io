from manimlib import *
import numpy as np


class PrefillDecode(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        PREFILL_COLOR = "#2ecc71"
        DECODE_COLOR = "#ff6b6b"
        MATRIX_BG = "#2a2a4e"
        HIGHLIGHT = "#ffd93d"

        # --- Title ---
        title = Text(
            "LLM Inference: Prefill vs Decode",
            font_size=30, color=WHITE, weight=BOLD
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # --- Divider ---
        divider = DashedLine(
            start=UP * 2.5, end=DOWN * 3.2,
            color=GREY, stroke_width=1, dash_length=0.1
        )
        self.play(ShowCreation(divider), run_time=0.3)

        # --- Column headers ---
        prefill_header = Text("PREFILL", font_size=24, color=PREFILL_COLOR, weight=BOLD)
        decode_header = Text("DECODE", font_size=24, color=DECODE_COLOR, weight=BOLD)
        prefill_header.move_to(LEFT * 3.5 + UP * 2.2)
        decode_header.move_to(RIGHT * 3.5 + UP * 2.2)
        self.play(FadeIn(prefill_header), FadeIn(decode_header), run_time=0.4)

        # =====================================================================
        # LEFT SIDE: PREFILL - Matrix x Matrix
        # =====================================================================
        cell_size = 0.42
        n = 4

        def make_matrix(rows, cols, color, center_pos):
            cells = VGroup()
            for r in range(rows):
                for c in range(cols):
                    rect = Rectangle(
                        width=cell_size, height=cell_size,
                        stroke_color=color, stroke_width=1.2,
                        fill_color=MATRIX_BG, fill_opacity=0.4,
                    )
                    cells.add(rect)
            grid = cells.arrange_in_grid(n_rows=rows, n_cols=cols, buff=0.04)
            grid.move_to(center_pos)
            return grid, cells

        # Input matrix A (4x4)
        prefill_center = LEFT * 3.5 + UP * 0.3
        mat_a, mat_a_cells = make_matrix(n, n, PREFILL_COLOR, prefill_center + LEFT * 1.6)

        # Multiplication symbol
        times_left = Text("x", font_size=24, color=WHITE)
        times_left.move_to(prefill_center)

        # Weight matrix B (4x4)
        mat_b, mat_b_cells = make_matrix(n, n, PREFILL_COLOR, prefill_center + RIGHT * 1.6)

        # Label
        prefill_op = Text("Matrix x Matrix", font_size=16, color=PREFILL_COLOR)
        prefill_op.next_to(VGroup(mat_a, mat_b), UP, buff=0.2)

        self.play(
            FadeIn(mat_a), FadeIn(mat_b), FadeIn(times_left),
            FadeIn(prefill_op),
            run_time=0.5,
        )
        self.wait(0.3)

        # Animate: all cells fill simultaneously (parallel processing)
        all_a_anims = [
            cell.animate.set_fill(PREFILL_COLOR, opacity=0.55)
            for cell in mat_a_cells
        ]
        all_b_anims = [
            cell.animate.set_fill(PREFILL_COLOR, opacity=0.55)
            for cell in mat_b_cells
        ]
        self.play(*all_a_anims, *all_b_anims, run_time=0.8)

        # Token labels showing parallel processing
        token_labels_prefill = VGroup()
        token_names = ["tok1", "tok2", "tok3", "tok4"]
        for i, name in enumerate(token_names):
            lbl = Text(name, font_size=11, color=PREFILL_COLOR)
            lbl.next_to(mat_a_cells[i * n], LEFT, buff=0.15)
            token_labels_prefill.add(lbl)

        # All tokens appear at once
        self.play(*[FadeIn(lbl) for lbl in token_labels_prefill], run_time=0.4)

        parallel_text = Text("All tokens in parallel", font_size=14, color=PREFILL_COLOR)
        parallel_text.next_to(VGroup(mat_a, mat_b), DOWN, buff=0.25)
        self.play(FadeIn(parallel_text), run_time=0.3)

        # COMPUTE-BOUND label
        compute_label = Text("COMPUTE-BOUND", font_size=18, color=PREFILL_COLOR, weight=BOLD)
        compute_label.next_to(parallel_text, DOWN, buff=0.2)

        compute_box = SurroundingRectangle(
            compute_label, buff=0.1,
            stroke_color=PREFILL_COLOR, stroke_width=2,
            fill_color=PREFILL_COLOR, fill_opacity=0.1,
        )
        self.play(FadeIn(compute_label), ShowCreation(compute_box), run_time=0.5)
        self.wait(0.5)

        # =====================================================================
        # RIGHT SIDE: DECODE - Vector x Matrix
        # =====================================================================
        decode_center = RIGHT * 3.5 + UP * 0.3

        # Single vector (1 column, 4 rows)
        vec, vec_cells = make_matrix(n, 1, DECODE_COLOR, decode_center + LEFT * 1.6)

        # Multiplication symbol
        times_right = Text("x", font_size=24, color=WHITE)
        times_right.move_to(decode_center)

        # Weight matrix (4x4)
        mat_w, mat_w_cells = make_matrix(n, n, DECODE_COLOR, decode_center + RIGHT * 1.4)

        # Label
        decode_op = Text("Vector x Matrix", font_size=16, color=DECODE_COLOR)
        decode_op.next_to(VGroup(vec, mat_w), UP, buff=0.2)

        self.play(
            FadeIn(vec), FadeIn(mat_w), FadeIn(times_right),
            FadeIn(decode_op),
            run_time=0.5,
        )
        self.wait(0.3)

        # Animate: one cell at a time (sequential processing)
        decode_tok_labels = VGroup()
        for i, cell in enumerate(vec_cells):
            tok_lbl = Text(f"tok{i+1}", font_size=11, color=DECODE_COLOR)
            tok_lbl.next_to(cell, LEFT, buff=0.15)
            decode_tok_labels.add(tok_lbl)
            self.play(
                cell.animate.set_fill(DECODE_COLOR, opacity=0.55),
                FadeIn(tok_lbl),
                run_time=0.3,
            )

        # Fill weight matrix dimly (data loaded but GPU underutilized)
        dim_anims = [
            cell.animate.set_fill(DECODE_COLOR, opacity=0.2)
            for cell in mat_w_cells
        ]
        self.play(*dim_anims, run_time=0.4)

        sequential_text = Text("One token at a time", font_size=14, color=DECODE_COLOR)
        sequential_text.next_to(VGroup(vec, mat_w), DOWN, buff=0.25)
        self.play(FadeIn(sequential_text), run_time=0.3)

        # MEMORY-BOUND label
        memory_label = Text("MEMORY-BOUND", font_size=18, color=DECODE_COLOR, weight=BOLD)
        memory_label.next_to(sequential_text, DOWN, buff=0.2)

        memory_box = SurroundingRectangle(
            memory_label, buff=0.1,
            stroke_color=DECODE_COLOR, stroke_width=2,
            fill_color=DECODE_COLOR, fill_opacity=0.1,
        )
        self.play(FadeIn(memory_label), ShowCreation(memory_box), run_time=0.5)
        self.wait(0.5)

        # =====================================================================
        # FINAL COMPARISON
        # =====================================================================
        all_objects = VGroup(
            mat_a, mat_b, times_left, prefill_op, parallel_text,
            compute_label, compute_box, token_labels_prefill,
            vec, mat_w, times_right, decode_op, sequential_text,
            memory_label, memory_box, divider, prefill_header,
            decode_header, title, decode_tok_labels,
        )
        self.play(FadeOut(all_objects), run_time=0.5)

        final_prefill = Text(
            "Prefill: GPU cores saturated",
            font_size=24, color=PREFILL_COLOR
        )
        final_decode = Text(
            "Decode: GPU idle, waiting for data",
            font_size=24, color=DECODE_COLOR
        )
        final_group = VGroup(final_prefill, final_decode).arrange(DOWN, buff=0.5)
        final_group.move_to(ORIGIN)

        self.play(Write(final_prefill), run_time=0.6)
        self.wait(0.2)
        self.play(Write(final_decode), run_time=0.6)
        self.wait(1.5)
