from manim import *
import numpy as np


class Im2colConv(Scene):
    """Visualizes the im2col trick: converting convolution into matrix
    multiplication by extracting patches from a 4x4 input with a 3x3
    kernel, flattening them into rows, and multiplying by the flattened
    kernel column vector."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("im2col: Convolution as Matrix Multiply", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # -- Constants --
        CELL = 0.48
        SMALL = 0.38
        INPUT_COLOR = "#4ecdc4"
        KERNEL_COLOR = "#ffd93d"
        PATCH_COLORS = ["#ff6b6b", "#45b7d1", "#a8e6cf", "#96ceb4"]
        MATRIX_COLOR = "#4ecdc4"

        # -- Data --
        np.random.seed(42)
        image = np.array([
            [1,  2,  3,  4],
            [5,  6,  7,  8],
            [9,  10, 11, 12],
            [13, 14, 15, 16],
        ])
        kernel = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ])
        kh, kw = 3, 3
        OH, OW = 2, 2  # output 2x2 with stride=1

        # -- Helper: build a grid of squares with numbers --
        def make_grid(data, rows, cols, cell_size, fill_color,
                      origin, font_size=14, fill_opacity=0.15):
            squares = VGroup()
            texts = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(
                        side_length=cell_size, stroke_width=1.5,
                        stroke_color=GREY_B,
                        fill_color=fill_color, fill_opacity=fill_opacity
                    )
                    sq.move_to(origin + RIGHT * c * cell_size + DOWN * r * cell_size)
                    val = data[r, c]
                    txt = Text(str(int(val)), font_size=font_size, color=WHITE)
                    txt.move_to(sq.get_center())
                    squares.add(sq)
                    texts.add(txt)
            return squares, texts

        # ============================
        # PHASE 1: Show input + kernel
        # ============================
        img_origin = LEFT * 5.0 + UP * 1.0
        img_label = Text("Input (4x4)", font_size=14, color=INPUT_COLOR)
        img_sq, img_tx = make_grid(image, 4, 4, CELL, INPUT_COLOR, img_origin)
        img_label.next_to(
            img_origin + RIGHT * 1.5 * CELL + UP * 0.05, UP, buff=0.15
        )

        self.play(FadeIn(img_label), FadeIn(img_sq), FadeIn(img_tx), run_time=0.5)

        # Kernel grid
        kern_origin = LEFT * 1.8 + UP * 1.0
        kern_label = Text("Kernel (3x3)", font_size=14, color=KERNEL_COLOR)
        kern_sq, kern_tx = make_grid(
            kernel, 3, 3, CELL, KERNEL_COLOR, kern_origin, fill_opacity=0.25
        )
        kern_label.next_to(
            kern_origin + RIGHT * 1.0 * CELL + UP * 0.05, UP, buff=0.15
        )

        self.play(FadeIn(kern_label), FadeIn(kern_sq), FadeIn(kern_tx), run_time=0.4)
        self.wait(0.3)

        # ==================================
        # PHASE 2: Slide kernel, extract patches
        # ==================================
        step_label = Text(
            "Step 1: Extract patches where kernel slides",
            font_size=16, color="#a8e6cf"
        )
        step_label.to_edge(DOWN, buff=0.35)
        self.play(Write(step_label), run_time=0.4)

        # Highlight rectangle for sliding window
        def get_highlight(row, col):
            top_left = img_origin + RIGHT * col * CELL + DOWN * row * CELL
            rect = Rectangle(
                width=kw * CELL, height=kh * CELL,
                stroke_color=KERNEL_COLOR, stroke_width=3,
                fill_color=KERNEL_COLOR, fill_opacity=0.1
            )
            rect.move_to(
                top_left + RIGHT * (kw - 1) * CELL / 2 + DOWN * (kh - 1) * CELL / 2
            )
            return rect

        highlight = get_highlight(0, 0)
        self.play(Create(highlight), run_time=0.3)

        # Extract 4 patches and show them as labels
        positions_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        patch_labels = []
        for pi, (pr, pc) in enumerate(positions_list):
            new_hl = get_highlight(pr, pc)
            self.play(Transform(highlight, new_hl), run_time=0.3)

            # Flash the patch cells
            patch_indices = []
            for dr in range(kh):
                for dc in range(kw):
                    patch_indices.append((pr + dr) * 4 + (pc + dc))

            flash_anims = [
                img_sq[idx].animate.set_fill(PATCH_COLORS[pi], opacity=0.5)
                for idx in patch_indices
            ]
            self.play(*flash_anims, run_time=0.2)

            # Small patch label
            patch = image[pr:pr+kh, pc:pc+kw].flatten()
            vals_str = ",".join(str(int(x)) for x in patch)
            plbl = Text(
                f"P{pi+1}: [{vals_str}]",
                font_size=10, color=PATCH_COLORS[pi]
            )
            plbl.move_to(LEFT * 5.0 + DOWN * (1.8 + pi * 0.3))
            patch_labels.append(plbl)
            self.play(FadeIn(plbl), run_time=0.15)

            # Reset colors
            reset_anims = [
                img_sq[idx].animate.set_fill(INPUT_COLOR, opacity=0.15)
                for idx in patch_indices
            ]
            self.play(*reset_anims, run_time=0.1)

        self.play(FadeOut(highlight), run_time=0.2)
        self.wait(0.3)

        # ===================================
        # PHASE 3: Build im2col matrix
        # ===================================
        new_step = Text(
            "Step 2: Stack patches as rows -> im2col matrix",
            font_size=16, color="#a8e6cf"
        )
        self.play(Transform(step_label, new_step), run_time=0.3)

        # Fade out patch labels, kernel grid
        self.play(
            *[FadeOut(pl) for pl in patch_labels],
            FadeOut(kern_sq), FadeOut(kern_tx), FadeOut(kern_label),
            FadeOut(img_sq), FadeOut(img_tx), FadeOut(img_label),
            run_time=0.4
        )

        # im2col matrix (4 rows x 9 cols) - left side
        mat_origin = LEFT * 5.8 + UP * 1.2
        mat_label = Text("im2col Matrix (4x9)", font_size=14, color=MATRIX_COLOR)
        mat_label.next_to(mat_origin + RIGHT * 4.0 * SMALL, UP, buff=0.2)
        self.play(FadeIn(mat_label), run_time=0.2)

        # Build each row from patches
        all_row_groups = []
        for pi, (pr, pc) in enumerate(positions_list):
            patch = image[pr:pr+kh, pc:pc+kw].flatten()
            color = PATCH_COLORS[pi]
            row_group = VGroup()
            for ci, val in enumerate(patch):
                sq = Square(
                    side_length=SMALL, stroke_width=1,
                    stroke_color=GREY_D,
                    fill_color=color, fill_opacity=0.2
                )
                sq.move_to(
                    mat_origin + RIGHT * ci * SMALL + DOWN * pi * SMALL
                )
                txt = Text(str(int(val)), font_size=10, color=WHITE)
                txt.move_to(sq.get_center())
                row_group.add(sq, txt)
            all_row_groups.append(row_group)
            self.play(FadeIn(row_group), run_time=0.3)

        self.wait(0.3)

        # ===================================
        # PHASE 4: Kernel as column vector
        # ===================================
        new_step2 = Text(
            "Step 3: Flatten kernel into column vector",
            font_size=16, color="#a8e6cf"
        )
        self.play(Transform(step_label, new_step2), run_time=0.3)

        kern_flat = kernel.flatten()
        kern_col_origin = RIGHT * 0.8 + UP * 1.2
        kern_col_label = Text(
            "Kernel (9x1)", font_size=14, color=KERNEL_COLOR
        )
        kern_col_label.next_to(kern_col_origin + DOWN * 0.2, UP, buff=0.2)
        self.play(FadeIn(kern_col_label), run_time=0.2)

        kern_col_group = VGroup()
        for ri, val in enumerate(kern_flat):
            sq = Square(
                side_length=SMALL, stroke_width=1,
                stroke_color=GREY_D,
                fill_color=KERNEL_COLOR, fill_opacity=0.2
            )
            sq.move_to(kern_col_origin + DOWN * ri * SMALL)
            txt = Text(str(int(val)), font_size=10, color=WHITE)
            txt.move_to(sq.get_center())
            kern_col_group.add(sq, txt)

        self.play(FadeIn(kern_col_group), run_time=0.4)

        # Multiply symbol
        times_sym = Text("@", font_size=24, color=WHITE, weight=BOLD)
        times_sym.move_to(RIGHT * 0.15 + UP * 0.3)
        self.play(FadeIn(times_sym), run_time=0.2)

        # Equals symbol
        eq_sym = Text("=", font_size=24, color=WHITE, weight=BOLD)
        eq_sym.move_to(RIGHT * 2.2 + UP * 0.3)
        self.play(FadeIn(eq_sym), run_time=0.2)

        # ===================================
        # PHASE 5: Output vector
        # ===================================
        new_step3 = Text(
            "Step 4: Matrix multiply gives convolution output",
            font_size=16, color="#a8e6cf"
        )
        self.play(Transform(step_label, new_step3), run_time=0.3)

        # Compute actual output
        output_vals = []
        for pr, pc in positions_list:
            patch = image[pr:pr+kh, pc:pc+kw]
            output_vals.append(int(np.sum(patch * kernel)))

        out_origin = RIGHT * 3.0 + UP * 1.2
        out_label = Text("Output (4x1)", font_size=14, color="#a8e6cf")
        out_label.next_to(out_origin + DOWN * 0.2, UP, buff=0.2)
        self.play(FadeIn(out_label), run_time=0.2)

        out_group = VGroup()
        for ri, val in enumerate(output_vals):
            sq = Square(
                side_length=SMALL, stroke_width=1,
                stroke_color=GREY_D,
                fill_color="#a8e6cf", fill_opacity=0.2
            )
            sq.move_to(out_origin + DOWN * ri * SMALL)
            txt = Text(str(val), font_size=10, color=WHITE)
            txt.move_to(sq.get_center())
            out_group.add(sq, txt)

        # Animate each output row as dot product
        for ri in range(4):
            # Highlight the row in im2col matrix
            row_highlight_anims = []
            for si in range(0, len(all_row_groups[ri]), 2):  # squares are at even indices
                row_highlight_anims.append(
                    all_row_groups[ri][si].animate.set_fill(
                        PATCH_COLORS[ri], opacity=0.5
                    )
                )
            self.play(*row_highlight_anims, run_time=0.15)

            # Show the output cell appearing
            self.play(FadeIn(out_group[ri * 2], out_group[ri * 2 + 1]), run_time=0.2)

            # Reset row highlight
            reset_anims = []
            for si in range(0, len(all_row_groups[ri]), 2):
                reset_anims.append(
                    all_row_groups[ri][si].animate.set_fill(
                        PATCH_COLORS[ri], opacity=0.2
                    )
                )
            self.play(*reset_anims, run_time=0.1)

        self.wait(0.3)

        # Reshape note
        reshape_text = Text(
            "reshape (4,1) -> (2,2) output feature map",
            font_size=14, color=GREY_B
        )
        reshape_text.move_to(RIGHT * 3.0 + DOWN * 1.2)

        # Show the 2x2 output grid
        out_2d = np.array(output_vals).reshape(2, 2)
        out_grid_origin = RIGHT * 4.5 + DOWN * 1.8
        out_grid_sq, out_grid_tx = VGroup(), VGroup()
        for r in range(2):
            for c in range(2):
                sq = Square(
                    side_length=CELL, stroke_width=1.5,
                    stroke_color=GREY_B,
                    fill_color="#a8e6cf", fill_opacity=0.2
                )
                sq.move_to(out_grid_origin + RIGHT * c * CELL + DOWN * r * CELL)
                txt = Text(str(int(out_2d[r, c])), font_size=14, color=WHITE)
                txt.move_to(sq.get_center())
                out_grid_sq.add(sq)
                out_grid_tx.add(txt)

        out_grid_label = Text("Output (2x2)", font_size=14, color="#a8e6cf")
        out_grid_label.next_to(
            out_grid_origin + RIGHT * 0.5 * CELL, UP, buff=0.15
        )

        # Arrow from column vector to 2d grid
        reshape_arrow = Arrow(
            start=out_origin + DOWN * 3.5 * SMALL + RIGHT * 0.3,
            end=out_grid_origin + LEFT * 0.3,
            color=GREY_B, stroke_width=2, buff=0.1,
            max_tip_length_to_length_ratio=0.15
        )

        self.play(
            FadeIn(reshape_text),
            GrowArrow(reshape_arrow),
            run_time=0.3
        )
        self.play(
            FadeIn(out_grid_label),
            FadeIn(out_grid_sq), FadeIn(out_grid_tx),
            run_time=0.4
        )

        # Final insight
        new_step4 = Text(
            "Convolution = im2col + single GEMM  (GPU-friendly)",
            font_size=16, color="#a8e6cf"
        )
        self.play(Transform(step_label, new_step4), run_time=0.4)

        self.wait(2.0)
