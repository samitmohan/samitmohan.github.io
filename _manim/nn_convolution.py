from manimlib import *
import numpy as np


class Convolution(Scene):
    """Demonstrates 2D convolution: a 3x3 kernel slides across a 6x6 input
    image, computing dot products to produce a 4x4 output feature map."""

    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # ---- constants ----
        CELL = 0.42
        FONT = 16
        SMALL = 14
        ACCENT = "#e94560"
        KERNEL_COLOR = "#0f3460"
        OUTPUT_COLOR = "#16213e"
        HIGHLIGHT = YELLOW

        # ---- data ----
        np.random.seed(42)
        image = np.random.randint(0, 10, (6, 6))
        kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])

        # precompute full output
        out_h, out_w = 4, 4
        full_output = np.zeros((out_h, out_w), dtype=int)
        for r in range(out_h):
            for c in range(out_w):
                full_output[r, c] = int(
                    np.sum(image[r:r+3, c:c+3] * kernel))

        # ---- title ----
        title = Text("How Convolution Works", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=0.6)

        # ---- helper: build grid of squares + numbers ----
        def make_grid(data, rows, cols, cell_size, fill_color, origin,
                      font_size=FONT, text_color=WHITE):
            squares = VGroup()
            texts = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(side_length=cell_size, stroke_width=1.2,
                                stroke_color=WHITE,
                                fill_color=fill_color, fill_opacity=0.35)
                    sq.move_to(origin + RIGHT * c * cell_size
                               + DOWN * r * cell_size)
                    val = data[r, c]
                    txt = Text(str(int(val)), font_size=font_size,
                               color=text_color)
                    txt.move_to(sq.get_center())
                    squares.add(sq)
                    texts.add(txt)
            return squares, texts

        # ---- positions ----
        img_origin = LEFT * 4.8 + UP * 1.1
        kern_label_pos = LEFT * 0.3 + UP * 2.4
        out_origin = RIGHT * 2.8 + UP * 1.1

        # ---- input image grid ----
        img_label = Text("Input (6x6)", font_size=SMALL, color=BLUE_C)
        img_label.next_to(img_origin + RIGHT * 2.5 * CELL + DOWN * (-0.6) * CELL,
                          UP, buff=0.22)
        img_sq, img_tx = make_grid(image, 6, 6, CELL, "#16213e", img_origin)
        self.play(FadeIn(img_label), FadeIn(img_sq), FadeIn(img_tx),
                  run_time=0.6)

        # ---- kernel grid (shown separately in the middle-top area) ----
        kern_title = Text("Kernel (3x3)", font_size=SMALL, color=ACCENT)
        kern_title.move_to(kern_label_pos)
        kern_origin = kern_label_pos + DOWN * 0.45 + LEFT * CELL
        kern_sq, kern_tx = make_grid(kernel, 3, 3, CELL, KERNEL_COLOR,
                                     kern_origin, font_size=FONT,
                                     text_color="#e94560")
        self.play(FadeIn(kern_title), FadeIn(kern_sq), FadeIn(kern_tx),
                  run_time=0.5)

        # ---- output grid (starts empty) ----
        out_label = Text("Output (4x4)", font_size=SMALL, color=GREEN_C)
        out_label.next_to(out_origin + RIGHT * 1.5 * CELL + DOWN * (-0.6) * CELL,
                          UP, buff=0.22)
        empty_out = np.full((4, 4), 0)
        out_sq, out_tx = make_grid(empty_out, 4, 4, CELL, OUTPUT_COLOR,
                                   out_origin, font_size=FONT,
                                   text_color=GREEN_C)
        # make output texts invisible for now
        for t in out_tx:
            t.set_opacity(0)
        self.play(FadeIn(out_label), FadeIn(out_sq), FadeIn(out_tx),
                  run_time=0.5)

        # ---- sliding window highlight on input ----
        def get_highlight_rect(row, col):
            top_left = img_origin + RIGHT * col * CELL + DOWN * row * CELL
            rect = Rectangle(
                width=3 * CELL, height=3 * CELL,
                stroke_color=HIGHLIGHT, stroke_width=3,
                fill_color=HIGHLIGHT, fill_opacity=0.12,
            )
            rect.move_to(top_left + RIGHT * CELL + DOWN * CELL)
            return rect

        # ---- step through first 4 positions ----
        positions = [(0, 0), (0, 1), (0, 2), (0, 3)]
        highlight = get_highlight_rect(0, 0)
        self.play(ShowCreation(highlight), run_time=0.3)

        calc_text = None
        for step_i, (pr, pc) in enumerate(positions):
            # move highlight
            new_hl = get_highlight_rect(pr, pc)
            self.play(Transform(highlight, new_hl), run_time=0.35)

            # compute element-wise products
            patch = image[pr:pr+3, pc:pc+3]
            products = patch * kernel
            dot = int(np.sum(products))

            # build calculation string (compact)
            terms = []
            for kr in range(3):
                for kc in range(3):
                    if kernel[kr, kc] != 0:
                        terms.append(
                            f"{patch[kr, kc]}x{kernel[kr, kc]}")
            calc_str = " + ".join(terms) + f" = {dot}"
            new_calc = Text(calc_str, font_size=12, color=WHITE)
            new_calc.move_to(DOWN * 2.2)

            if calc_text is None:
                calc_text = new_calc
                self.play(FadeIn(calc_text), run_time=0.3)
            else:
                self.play(Transform(calc_text, new_calc), run_time=0.3)

            # fill output cell
            out_idx = pr * out_w + pc
            new_txt = Text(str(dot), font_size=FONT, color=GREEN_C)
            new_txt.move_to(out_sq[out_idx].get_center())

            # flash the output cell
            self.play(
                out_sq[out_idx].animate.set_fill(GREEN, opacity=0.5),
                FadeIn(new_txt),
                run_time=0.3,
            )
            self.play(
                out_sq[out_idx].animate.set_fill(OUTPUT_COLOR, opacity=0.35),
                run_time=0.2,
            )
            # replace the placeholder
            out_tx[out_idx].become(new_txt)

            self.wait(0.3)

        # ---- skip to full output ----
        self.play(FadeOut(highlight), FadeOut(calc_text), run_time=0.3)

        skip_text = Text("... kernel slides across all positions ...",
                         font_size=14, color=GREY_B)
        skip_text.move_to(DOWN * 2.2)
        self.play(FadeIn(skip_text), run_time=0.4)
        self.wait(0.5)

        # fill remaining cells
        fill_anims = []
        for r in range(out_h):
            for c in range(out_w):
                idx = r * out_w + c
                if (r, c) in positions:
                    continue  # already filled
                val = full_output[r, c]
                new_t = Text(str(val), font_size=FONT, color=GREEN_C)
                new_t.move_to(out_sq[idx].get_center())
                out_tx[idx].become(new_t)
                fill_anims.append(FadeIn(new_t))

        self.play(*fill_anims, run_time=0.6)
        self.play(FadeOut(skip_text), run_time=0.3)

        # ---- final explanation text ----
        footer = Text(
            "Kernel slides across image, computing dot products at each position",
            font_size=15, color=GREY_B,
        )
        footer.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(footer), run_time=0.5)
        self.wait(1.5)
