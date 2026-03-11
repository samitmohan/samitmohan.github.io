from manimlib import *
import numpy as np


class Im2Col(Scene):
    """Visualizes the im2col trick: extracting patches from a 4x4 input
    with a 2x2 kernel and stacking them into rows for matrix multiplication."""

    def construct(self):
        self.camera.background_rgba = [1, 1, 1, 1]

        # -- Constants --
        CELL = 0.55
        SMALL_CELL = 0.45
        FONT = 16
        SMALL = 14
        ACCENT = "#e94560"
        PATCH_COLORS = [
            "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
            "#00BCD4", "#F44336", "#8BC34A", "#FF5722",
            "#3F51B5",
        ]

        # -- Data --
        np.random.seed(7)
        image = np.arange(1, 17).reshape(4, 4)  # 1..16 for clarity
        kh, kw = 2, 2
        OH, OW = 3, 3  # output spatial dims (stride=1)

        # -- Title --
        title = Text("The im2col Trick", font_size=30, color=BLACK, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # -- Helper: build grid --
        def make_grid(data, rows, cols, cell_size, fill_color, origin,
                      font_size=FONT, text_color=BLACK, fill_opacity=0.15):
            squares = VGroup()
            texts = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(side_length=cell_size, stroke_width=1.5,
                                stroke_color=GREY_D,
                                fill_color=fill_color, fill_opacity=fill_opacity)
                    sq.move_to(origin + RIGHT * c * cell_size
                               + DOWN * r * cell_size)
                    val = data[r, c]
                    txt = Text(str(int(val)), font_size=font_size,
                               color=text_color)
                    txt.move_to(sq.get_center())
                    squares.add(sq)
                    texts.add(txt)
            return squares, texts

        # -- Positions --
        img_origin = LEFT * 5.5 + UP * 1.0

        # -- Input image --
        img_label = Text("Input (4x4)", font_size=SMALL, color=BLUE_D)
        img_label.next_to(img_origin + RIGHT * 1.5 * CELL + UP * 0.1,
                          UP, buff=0.2)
        img_sq, img_tx = make_grid(image, 4, 4, CELL, BLUE_A, img_origin)

        self.play(FadeIn(img_label), FadeIn(img_sq), FadeIn(img_tx),
                  run_time=0.6)

        # -- Kernel label --
        kern_info = Text("Kernel: 2x2, stride=1", font_size=SMALL, color=GREY_D)
        kern_info.next_to(img_label, RIGHT, buff=1.0)
        self.play(FadeIn(kern_info), run_time=0.3)

        # -- Columns matrix area --
        col_origin = RIGHT * 0.5 + UP * 1.0
        col_label = Text("Columns Matrix (9x4)", font_size=SMALL, color="#155724")
        col_label.next_to(col_origin + RIGHT * 1.5 * SMALL_CELL + UP * 0.1,
                          UP, buff=0.2)
        self.play(FadeIn(col_label), run_time=0.3)

        # -- Sliding window highlight --
        def get_highlight_rect(row, col):
            top_left = img_origin + RIGHT * col * CELL + DOWN * row * CELL
            rect = Rectangle(
                width=kw * CELL, height=kh * CELL,
                stroke_color=ACCENT, stroke_width=3.5,
                fill_color=ACCENT, fill_opacity=0.15,
            )
            rect.move_to(top_left + RIGHT * (kw - 1) * CELL / 2
                         + DOWN * (kh - 1) * CELL / 2)
            return rect

        # -- Extract patches and animate first 4, then fill rest --
        highlight = get_highlight_rect(0, 0)
        self.play(ShowCreation(highlight), run_time=0.3)

        col_rows = []  # store the row VGroups
        positions = [(r, c) for r in range(OH) for c in range(OW)]

        for step_i, (pr, pc) in enumerate(positions[:4]):
            # Move highlight
            new_hl = get_highlight_rect(pr, pc)
            self.play(Transform(highlight, new_hl), run_time=0.3)

            # Extract patch
            patch = image[pr:pr+kh, pc:pc+kw].flatten()  # 4 values
            color = PATCH_COLORS[step_i % len(PATCH_COLORS)]

            # Build row in columns matrix
            row_group = VGroup()
            for ci, val in enumerate(patch):
                sq = Square(side_length=SMALL_CELL, stroke_width=1.2,
                            stroke_color=GREY_D,
                            fill_color=color, fill_opacity=0.25)
                sq.move_to(col_origin + RIGHT * ci * SMALL_CELL
                           + DOWN * step_i * SMALL_CELL)
                txt = Text(str(int(val)), font_size=SMALL, color=BLACK)
                txt.move_to(sq.get_center())
                row_group.add(sq, txt)

            # Animate: flash patch cells on input, then show row
            patch_indices = []
            for dr in range(kh):
                for dc in range(kw):
                    patch_indices.append((pr + dr) * 4 + (pc + dc))

            flash_anims = []
            for idx in patch_indices:
                flash_anims.append(
                    img_sq[idx].animate.set_fill(color, opacity=0.5))

            self.play(*flash_anims, run_time=0.2)
            self.play(FadeIn(row_group), run_time=0.3)

            # Reset input colors
            reset_anims = []
            for idx in patch_indices:
                reset_anims.append(
                    img_sq[idx].animate.set_fill(BLUE_A, opacity=0.15))
            self.play(*reset_anims, run_time=0.15)

            col_rows.append(row_group)

        # Skip text and fill remaining rows
        self.play(FadeOut(highlight), run_time=0.2)
        skip_text = Text("... 5 more patches ...", font_size=13, color=GREY_D)
        skip_text.move_to(col_origin + RIGHT * 1.5 * SMALL_CELL
                          + DOWN * 6 * SMALL_CELL)
        self.play(FadeIn(skip_text), run_time=0.3)
        self.wait(0.3)

        fill_anims = []
        for step_i in range(4, len(positions)):
            pr, pc = positions[step_i]
            patch = image[pr:pr+kh, pc:pc+kw].flatten()
            color = PATCH_COLORS[step_i % len(PATCH_COLORS)]
            row_group = VGroup()
            for ci, val in enumerate(patch):
                sq = Square(side_length=SMALL_CELL, stroke_width=1.2,
                            stroke_color=GREY_D,
                            fill_color=color, fill_opacity=0.25)
                sq.move_to(col_origin + RIGHT * ci * SMALL_CELL
                           + DOWN * step_i * SMALL_CELL)
                txt = Text(str(int(val)), font_size=SMALL, color=BLACK)
                txt.move_to(sq.get_center())
                row_group.add(sq, txt)
            fill_anims.append(FadeIn(row_group))
            col_rows.append(row_group)

        self.play(*fill_anims, run_time=0.5)
        self.play(FadeOut(skip_text), run_time=0.2)

        # -- Show the matrix multiply equation --
        eq_text = Text(
            "output = columns @ weights.T",
            font_size=18, color=BLACK, weight=BOLD,
        )
        eq_text.to_edge(DOWN, buff=0.6)

        shape_text = Text(
            "(9 x 4) @ (4 x out_ch) = (9 x out_ch) -> reshape to (3 x 3 x out_ch)",
            font_size=13, color=GREY_D,
        )
        shape_text.next_to(eq_text, DOWN, buff=0.15)

        self.play(Write(eq_text), run_time=0.5)
        self.play(FadeIn(shape_text), run_time=0.3)

        # -- Final note --
        note = Text(
            "Convolution becomes a single matrix multiply",
            font_size=16, color=GREY_D,
        )
        note.to_edge(DOWN, buff=0.15)
        self.play(FadeOut(shape_text), FadeIn(note), run_time=0.4)
        self.wait(1.5)
