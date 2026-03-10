from manimlib import *
import numpy as np


class FlashAttention(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Color palette ---
        HBM_COLOR = "#4a4a6a"
        SRAM_COLOR = "#2d6a4f"
        Q_COLOR = "#ff6b6b"
        KT_COLOR = "#4ecdc4"
        RESULT_COLOR = "#ffd93d"
        TILE_HIGHLIGHT = "#e07cff"

        # =====================================================================
        # STEP 1: Title
        # =====================================================================
        title = Text("Flash Attention: Tiling Strategy", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # =====================================================================
        # STEP 2: HBM region with Q and K^T matrices
        # =====================================================================
        hbm_box = Rectangle(
            width=12.5, height=2.8,
            stroke_color=HBM_COLOR, stroke_width=2,
            fill_color=HBM_COLOR, fill_opacity=0.15,
        )
        hbm_box.next_to(title, DOWN, buff=0.3)
        hbm_label = Text("HBM (Slow Memory)", font_size=18, color="#8888aa")
        hbm_label.next_to(hbm_box, UP, buff=0.08, aligned_edge=LEFT)

        N = 6  # grid size NxN

        def make_grid(rows, cols, cell_w, cell_h, color, label_text):
            cells = VGroup()
            for i in range(rows):
                for j in range(cols):
                    rect = Rectangle(
                        width=cell_w, height=cell_h,
                        stroke_color=color, stroke_width=1.2,
                        fill_color=color, fill_opacity=0.2,
                    )
                    cells.add(rect)
            grid = cells.arrange_in_grid(n_rows=rows, n_cols=cols, buff=0.03)
            label = Text(label_text, font_size=18, color=color, weight=BOLD)
            label.next_to(grid, UP, buff=0.12)
            return VGroup(label, grid), cells

        q_group, q_cells = make_grid(N, N, 0.32, 0.28, Q_COLOR, "Q")
        kt_group, kt_cells = make_grid(N, N, 0.32, 0.28, KT_COLOR, "K^T")

        # Output matrix placeholder in HBM
        out_group, out_cells = make_grid(N, N, 0.32, 0.28, RESULT_COLOR, "Output O")

        matrices_row = VGroup(q_group, kt_group, out_group).arrange(RIGHT, buff=0.6)
        matrices_row.move_to(hbm_box.get_center())

        self.play(FadeIn(hbm_box), Write(hbm_label), run_time=0.4)
        self.play(
            FadeIn(q_group, shift=DOWN * 0.2),
            FadeIn(kt_group, shift=DOWN * 0.2),
            FadeIn(out_group, shift=DOWN * 0.2),
            run_time=0.6,
        )
        self.wait(0.3)

        # =====================================================================
        # STEP 3: SRAM region
        # =====================================================================
        sram_box = Rectangle(
            width=8.0, height=2.2,
            stroke_color=SRAM_COLOR, stroke_width=2,
            fill_color=SRAM_COLOR, fill_opacity=0.15,
        )
        sram_box.next_to(hbm_box, DOWN, buff=0.7)
        sram_label = Text("SRAM (Fast On-Chip Memory)", font_size=18, color="#52b788")
        sram_label.next_to(sram_box, UP, buff=0.08, aligned_edge=LEFT)

        self.play(FadeIn(sram_box), Write(sram_label), run_time=0.4)
        self.wait(0.2)

        # =====================================================================
        # STEP 4: Tiling loop - animate tiles moving from HBM to SRAM
        # =====================================================================
        tile_size = 3  # tiles are 3x3 sub-blocks
        num_tile_steps = (N // tile_size)  # 2 tiles along each dimension

        step_label = Text("", font_size=16, color=WHITE)
        step_label.next_to(sram_box, DOWN, buff=0.15)

        iteration = 0
        for qi in range(num_tile_steps):
            for ki in range(num_tile_steps):
                iteration += 1

                # Identify Q tile cells (rows qi*tile_size .. (qi+1)*tile_size)
                q_tile_indices = []
                for r in range(qi * tile_size, (qi + 1) * tile_size):
                    for c in range(N):
                        q_tile_indices.append(r * N + c)

                # Identify K^T tile cells (cols ki*tile_size .. (ki+1)*tile_size)
                kt_tile_indices = []
                for r in range(N):
                    for c in range(ki * tile_size, (ki + 1) * tile_size):
                        kt_tile_indices.append(r * N + c)

                # Highlight tiles in HBM
                q_highlights = VGroup(*[
                    q_cells[i].copy().set_fill(TILE_HIGHLIGHT, opacity=0.5)
                    for i in q_tile_indices
                ])
                kt_highlights = VGroup(*[
                    kt_cells[i].copy().set_fill(TILE_HIGHLIGHT, opacity=0.5)
                    for i in kt_tile_indices
                ])

                # Build small tiles for SRAM
                q_tile_sram, _ = make_grid(
                    tile_size, N, 0.28, 0.24, Q_COLOR, f"Q tile {qi+1}"
                )
                kt_tile_sram, _ = make_grid(
                    N, tile_size, 0.28, 0.24, KT_COLOR, f"K^T tile {ki+1}"
                )
                partial_result, partial_cells = make_grid(
                    tile_size, tile_size, 0.35, 0.28, RESULT_COLOR, "Partial Attn"
                )

                sram_contents = VGroup(q_tile_sram, kt_tile_sram, partial_result)
                sram_contents.arrange(RIGHT, buff=0.35)
                sram_contents.move_to(sram_box.get_center())

                # Update step label
                new_step_label = Text(
                    f"Step {iteration}/{num_tile_steps**2}: "
                    f"Load Q-tile {qi+1}, K^T-tile {ki+1} into SRAM, compute partial attention",
                    font_size=14, color="#cccccc",
                )
                new_step_label.next_to(sram_box, DOWN, buff=0.15)

                # Animate: highlight in HBM
                self.play(
                    FadeIn(q_highlights),
                    FadeIn(kt_highlights),
                    run_time=0.35,
                )

                # Animate: move tiles down into SRAM
                # Create copies at HBM positions, then animate to SRAM positions
                q_fly = q_tile_sram.copy().move_to(q_group.get_center())
                kt_fly = kt_tile_sram.copy().move_to(kt_group.get_center())

                self.play(
                    ReplacementTransform(q_fly, q_tile_sram),
                    ReplacementTransform(kt_fly, kt_tile_sram),
                    FadeTransform(step_label, new_step_label),
                    run_time=0.5,
                )
                step_label = new_step_label

                # Show compute happening - flash partial result
                self.play(
                    FadeIn(partial_result),
                    run_time=0.4,
                )

                # Identify output cells that get updated
                out_tile_indices = []
                for r in range(qi * tile_size, (qi + 1) * tile_size):
                    for c in range(ki * tile_size, (ki + 1) * tile_size):
                        out_tile_indices.append(r * N + c)

                # Animate: write result back to HBM output
                out_fly = partial_result.copy()
                out_target_center = VGroup(*[out_cells[i] for i in out_tile_indices]).get_center()

                # Flash the output cells
                out_highlights = VGroup(*[
                    out_cells[i].copy().set_fill(RESULT_COLOR, opacity=0.6)
                    for i in out_tile_indices
                ])

                self.play(
                    FadeOut(out_fly),
                    FadeIn(out_highlights),
                    run_time=0.4,
                )

                # Update the actual output cells to show they are filled
                for i in out_tile_indices:
                    out_cells[i].set_fill(RESULT_COLOR, opacity=0.35)

                # Clean up SRAM and highlights for next iteration
                self.play(
                    FadeOut(q_highlights),
                    FadeOut(kt_highlights),
                    FadeOut(out_highlights),
                    FadeOut(q_tile_sram),
                    FadeOut(kt_tile_sram),
                    FadeOut(partial_result),
                    run_time=0.3,
                )

        self.wait(0.3)

        # =====================================================================
        # STEP 5: Show completed output matrix
        # =====================================================================
        done_label = Text(
            "All tiles processed - Output O complete",
            font_size=16, color="#52b788",
        )
        done_label.move_to(step_label.get_center())
        self.play(FadeTransform(step_label, done_label), run_time=0.4)

        # Flash the full output matrix
        full_out_flash = SurroundingRectangle(
            out_group, color=RESULT_COLOR, buff=0.08, stroke_width=3
        )
        self.play(ShowCreation(full_out_flash), run_time=0.4)
        self.play(FadeOut(full_out_flash), run_time=0.3)
        self.wait(0.3)

        # =====================================================================
        # STEP 6: Memory comparison
        # =====================================================================
        self.play(
            FadeOut(hbm_box), FadeOut(hbm_label),
            FadeOut(sram_box), FadeOut(sram_label),
            FadeOut(q_group), FadeOut(kt_group), FadeOut(out_group),
            FadeOut(done_label), FadeOut(title),
            run_time=0.5,
        )

        compare_title = Text("Memory Comparison", font_size=28, color=WHITE)
        compare_title.to_edge(UP, buff=0.5)

        # Naive attention
        naive_box = Rectangle(
            width=4.5, height=2.5,
            stroke_color="#ff4444", stroke_width=2,
            fill_color="#ff4444", fill_opacity=0.1,
        )
        naive_title = Text("Naive Attention", font_size=22, color="#ff6b6b", weight=BOLD)
        naive_mem = Text("O(N^2) memory", font_size=20, color="#ff6b6b")
        naive_detail = Text(
            "Materializes full N x N\nattention matrix in HBM",
            font_size=14, color="#cc9999",
        )
        naive_content = VGroup(naive_title, naive_mem, naive_detail).arrange(DOWN, buff=0.2)
        naive_content.move_to(naive_box.get_center())
        naive_group = VGroup(naive_box, naive_content)

        # Flash attention
        flash_box = Rectangle(
            width=4.5, height=2.5,
            stroke_color="#44ff44", stroke_width=2,
            fill_color="#44ff44", fill_opacity=0.1,
        )
        flash_title = Text("Flash Attention", font_size=22, color="#52b788", weight=BOLD)
        flash_mem = Text("O(N) memory", font_size=20, color="#52b788")
        flash_detail = Text(
            "Tiles loaded into SRAM\nNever stores full N x N",
            font_size=14, color="#99cc99",
        )
        flash_content = VGroup(flash_title, flash_mem, flash_detail).arrange(DOWN, buff=0.2)
        flash_content.move_to(flash_box.get_center())
        flash_group = VGroup(flash_box, flash_content)

        comparison = VGroup(naive_group, flash_group).arrange(RIGHT, buff=1.0)
        comparison.next_to(compare_title, DOWN, buff=0.6)

        vs_text = Text("vs", font_size=24, color=WHITE, weight=BOLD)
        vs_text.move_to(
            (naive_group.get_right() + flash_group.get_left()) / 2
        )

        self.play(Write(compare_title), run_time=0.4)
        self.play(
            FadeIn(naive_group, shift=RIGHT * 0.3),
            FadeIn(flash_group, shift=LEFT * 0.3),
            FadeIn(vs_text),
            run_time=0.7,
        )

        # Emphasize flash with a glow
        flash_glow = SurroundingRectangle(
            flash_group, color="#52b788", buff=0.1, stroke_width=3
        )
        self.play(ShowCreation(flash_glow), run_time=0.4)

        # Bottom summary
        summary = Text(
            "Same exact result, dramatically less memory",
            font_size=18, color="#dda0dd",
        )
        summary.next_to(comparison, DOWN, buff=0.5)
        self.play(Write(summary), run_time=0.5)
        self.wait(1.5)
