from manim import *
import numpy as np


class FlashAttention(Scene):
    """Standard Attention (full NxN in HBM) vs Flash Attention (tiled in SRAM)."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("Flash Attention: HBM vs SRAM", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # --- Column headers ---
        std_label = Text("Standard Attention", font_size=20, color="#ff6b6b")
        flash_label = Text("Flash Attention", font_size=20, color="#a8e6cf")
        std_label.move_to(LEFT * 3.5 + UP * 2.3)
        flash_label.move_to(RIGHT * 3.5 + UP * 2.3)
        self.play(FadeIn(std_label), FadeIn(flash_label), run_time=0.4)

        divider = DashedLine(
            start=UP * 2.6, end=DOWN * 3.5,
            color=GREY, stroke_width=1, dash_length=0.1
        )
        self.play(Create(divider), run_time=0.2)

        # ==================================================================
        # LEFT SIDE: Standard Attention - full NxN matrix in HBM
        # ==================================================================
        n = 6
        cell = 0.32
        std_center = np.array([-3.5, 0.3, 0])

        # HBM memory block
        hbm_box = RoundedRectangle(
            width=n * cell + 0.6, height=n * cell + 1.0,
            corner_radius=0.15,
            fill_color="#ff6b6b", fill_opacity=0.08,
            stroke_color="#ff6b6b", stroke_width=2,
        )
        hbm_box.move_to(RIGHT * std_center[0] + UP * std_center[1])

        hbm_label = Text("HBM (slow)", font_size=14, color="#ff6b6b")
        hbm_label.next_to(hbm_box, UP, buff=0.08)

        self.play(FadeIn(hbm_box), FadeIn(hbm_label), run_time=0.3)

        # Build full NxN attention matrix
        std_grid = VGroup()
        for r in range(n):
            for c in range(n):
                sq = Square(
                    side_length=cell,
                    fill_color="#ff6b6b", fill_opacity=0.0,
                    stroke_color="#ff6b6b", stroke_width=0.8,
                )
                x = std_center[0] + (c - n / 2 + 0.5) * cell
                y = std_center[1] + (n / 2 - 0.5 - r) * cell
                sq.move_to(RIGHT * x + UP * y)
                std_grid.add(sq)

        nxn_label = Text("N x N attention matrix", font_size=12, color=GREY_B)
        nxn_label.next_to(hbm_box, DOWN, buff=0.08)

        self.play(FadeIn(std_grid), FadeIn(nxn_label), run_time=0.3)

        # Fill the entire matrix row by row (materializes everything)
        for row in range(n):
            row_anims = []
            for col in range(n):
                idx = row * n + col
                row_anims.append(
                    std_grid[idx].animate.set_fill("#ff6b6b", opacity=0.6)
                )
            self.play(*row_anims, run_time=0.2)

        # HBM read/write arrows
        hbm_rw = Text("read/write full matrix", font_size=11, color="#ff6b6b")
        hbm_rw.next_to(nxn_label, DOWN, buff=0.06)
        self.play(FadeIn(hbm_rw), run_time=0.2)

        self.wait(0.3)

        # ==================================================================
        # RIGHT SIDE: Flash Attention - tiled in SRAM
        # ==================================================================
        flash_center = np.array([3.5, 0.3, 0])
        tile_size = 3  # 3x3 tiles out of 6x6

        # SRAM memory block (small, fast)
        sram_box = RoundedRectangle(
            width=tile_size * cell + 0.6, height=tile_size * cell + 0.6,
            corner_radius=0.15,
            fill_color="#a8e6cf", fill_opacity=0.08,
            stroke_color="#a8e6cf", stroke_width=2,
        )
        sram_box.move_to(RIGHT * flash_center[0] + UP * flash_center[1])

        sram_label = Text("SRAM (fast)", font_size=14, color="#a8e6cf")
        sram_label.next_to(sram_box, UP, buff=0.08)

        self.play(FadeIn(sram_box), FadeIn(sram_label), run_time=0.3)

        # Show tile-by-tile processing
        tile_label = Text("tile (block)", font_size=12, color=GREY_B)
        tile_label.next_to(sram_box, DOWN, buff=0.08)
        self.play(FadeIn(tile_label), run_time=0.2)

        tile_positions = [
            (0, 0), (0, 1),
            (1, 0), (1, 1),
        ]  # 4 tiles covering 6x6 with 3x3 blocks

        tile_colors = ["#a8e6cf", "#4ecdc4", "#45b7d1", "#96ceb4"]

        for t_idx, (tr, tc) in enumerate(tile_positions):
            color = tile_colors[t_idx]

            # Create a small tile grid inside the SRAM box
            tile_grid = VGroup()
            for r in range(tile_size):
                for c in range(tile_size):
                    sq = Square(
                        side_length=cell,
                        fill_color=color, fill_opacity=0.0,
                        stroke_color=color, stroke_width=0.8,
                    )
                    x = flash_center[0] + (c - tile_size / 2 + 0.5) * cell
                    y = flash_center[1] + (tile_size / 2 - 0.5 - r) * cell
                    sq.move_to(RIGHT * x + UP * y)
                    tile_grid.add(sq)

            # Label which tile
            tile_num = Text(
                f"Tile ({tr},{tc})", font_size=12, color=color
            )
            tile_num.next_to(sram_box, LEFT, buff=0.15).shift(DOWN * 0.8)

            # Flash the tile: appear, fill, then fade for next
            fill_anims = [
                sq.animate.set_fill(color, opacity=0.6) for sq in tile_grid
            ]
            self.play(
                FadeIn(tile_grid), FadeIn(tile_num),
                run_time=0.15,
            )
            self.play(*fill_anims, run_time=0.25)

            if t_idx < len(tile_positions) - 1:
                self.play(
                    FadeOut(tile_grid), FadeOut(tile_num),
                    run_time=0.15,
                )
            else:
                # Keep last tile visible
                pass

        no_materialize = Text("never materialize full N x N", font_size=11, color="#a8e6cf")
        no_materialize.next_to(tile_label, DOWN, buff=0.06)
        self.play(FadeIn(no_materialize), run_time=0.2)

        self.wait(0.3)

        # ==================================================================
        # Memory comparison at bottom
        # ==================================================================
        mem_box = RoundedRectangle(
            width=10, height=1.0,
            corner_radius=0.15,
            fill_color="#2a2a4e", fill_opacity=0.5,
            stroke_color="#ffd93d", stroke_width=2,
        )
        mem_box.to_edge(DOWN, buff=0.2)

        # Bar comparison
        bar_y = mem_box.get_center()[1] + 0.1
        bar_max_width = 4.0

        # Standard: O(N^2) memory
        std_bar = Rectangle(
            width=bar_max_width, height=0.25,
            fill_color="#ff6b6b", fill_opacity=0.7,
            stroke_color="#ff6b6b", stroke_width=1,
        )
        std_bar.move_to(LEFT * 0.5 + UP * (bar_y + 0.15))
        std_bar_label = Text("Standard: O(N^2) memory", font_size=13, color="#ff6b6b")
        std_bar_label.next_to(std_bar, LEFT, buff=0.15)

        # Flash: O(N) memory (much smaller bar)
        flash_bar = Rectangle(
            width=bar_max_width * 0.15, height=0.25,
            fill_color="#a8e6cf", fill_opacity=0.7,
            stroke_color="#a8e6cf", stroke_width=1,
        )
        flash_bar.align_to(std_bar, LEFT)
        flash_bar.shift(DOWN * 0.35)
        flash_bar_label = Text("Flash: O(N) memory", font_size=13, color="#a8e6cf")
        flash_bar_label.next_to(flash_bar, LEFT, buff=0.15)

        self.play(
            FadeIn(mem_box),
            GrowFromEdge(std_bar, LEFT), FadeIn(std_bar_label),
            run_time=0.4,
        )
        self.play(
            GrowFromEdge(flash_bar, LEFT), FadeIn(flash_bar_label),
            run_time=0.4,
        )

        self.wait(2.0)
