from manim import *
import numpy as np


class Broadcasting(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e2e"

        # Data
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        vector = np.array([1, 0, 1])
        result = matrix + vector

        cell_size = 0.55
        font_size = 22

        def make_grid(data, color, position, opacity=1.0):
            """Create a colored grid with numbers."""
            rows, cols = data.shape
            group = VGroup()
            for r in range(rows):
                for c in range(cols):
                    rect = Square(side_length=cell_size)
                    rect.set_fill(color, opacity=opacity)
                    rect.set_stroke(WHITE, width=1)
                    rect.move_to(
                        position
                        + RIGHT * c * cell_size
                        + DOWN * r * cell_size
                    )
                    txt = Text(str(data[r, c]), font_size=font_size, color=WHITE)
                    txt.move_to(rect.get_center())
                    group.add(VGroup(rect, txt))
            return group

        def make_row(data, color, position, opacity=1.0):
            """Create a single-row grid."""
            group = VGroup()
            for c in range(len(data)):
                rect = Square(side_length=cell_size)
                rect.set_fill(color, opacity=opacity)
                rect.set_stroke(WHITE, width=1)
                rect.move_to(position + RIGHT * c * cell_size)
                txt = Text(str(data[c]), font_size=font_size, color=WHITE)
                txt.move_to(rect.get_center())
                group.add(VGroup(rect, txt))
            return group

        # --- Part 1: Show matrix and vector ---
        mat_label = Text("A  (4x3)", font_size=20, color=YELLOW)
        mat_origin = LEFT * 4.5 + UP * 1.5
        mat_grid = make_grid(matrix, BLUE_E, mat_origin)
        mat_label.next_to(mat_grid, UP, buff=0.2)

        vec_label = Text("v  (1x3)", font_size=20, color=YELLOW)
        vec_origin = mat_origin + DOWN * 4 * cell_size + DOWN * 0.4
        vec_grid = make_row(vector, GREEN_E, vec_origin)
        vec_label.next_to(vec_grid, UP, buff=0.15)

        self.play(
            FadeIn(mat_grid, shift=DOWN * 0.3),
            FadeIn(mat_label),
            run_time=0.6,
        )
        self.play(
            FadeIn(vec_grid, shift=UP * 0.3),
            FadeIn(vec_label),
            run_time=0.6,
        )

        # --- Part 2: Show broadcasting (stretch) ---
        plus_sign = Text("+", font_size=36, color=WHITE)
        plus_sign.move_to(LEFT * 1.2 + UP * 0.3)

        broadcast_label = Text("broadcast v", font_size=18, color=GREEN_A)
        ghost_origin = LEFT * 0.3 + UP * 1.5
        ghost_grids = []
        for r in range(4):
            row_group = make_row(
                vector, GREEN_E, ghost_origin + DOWN * r * cell_size, opacity=0.5
            )
            ghost_grids.append(row_group)

        stretched_label = Text("v stretched (4x3)", font_size=18, color=GREEN_A)
        ghost_all = VGroup(*ghost_grids)
        stretched_label.next_to(ghost_all, UP, buff=0.2)

        self.play(FadeIn(plus_sign), run_time=0.3)

        # Animate rows fading in one by one
        self.play(FadeIn(stretched_label), run_time=0.3)
        for i, ghost_row in enumerate(ghost_grids):
            self.play(FadeIn(ghost_row, shift=DOWN * 0.15), run_time=0.3)

        self.wait(0.3)

        # --- Part 3: Element-wise addition result ---
        equals_sign = Text("=", font_size=36, color=WHITE)
        equals_sign.move_to(RIGHT * 2.2 + UP * 0.3)

        res_origin = RIGHT * 3.2 + UP * 1.5
        res_grid = make_grid(result, PURPLE_B, res_origin)
        res_label = Text("A + v  (4x3)", font_size=18, color=YELLOW)
        res_label.next_to(res_grid, UP, buff=0.2)

        self.play(FadeIn(equals_sign), run_time=0.3)

        # Animate result cells appearing row by row
        for r in range(4):
            row_cells = VGroup(*[res_grid[r * 3 + c] for c in range(3)])
            self.play(FadeIn(row_cells, shift=RIGHT * 0.15), run_time=0.25)

        self.play(FadeIn(res_label), run_time=0.3)
        self.wait(0.8)

        # --- Part 4: Failure case ---
        everything = VGroup(
            mat_grid, mat_label, vec_grid, vec_label,
            plus_sign, ghost_all, stretched_label,
            equals_sign, res_grid, res_label,
        )
        self.play(FadeOut(everything), run_time=0.5)

        fail_title = Text("What about incompatible shapes?", font_size=26, color=WHITE)
        fail_title.to_edge(UP, buff=0.5)
        self.play(FadeIn(fail_title), run_time=0.4)

        shape_a = Text("(4, 3)", font_size=30, color=BLUE_C)
        shape_plus = Text("+", font_size=30, color=WHITE)
        shape_b = Text("(2, 1)", font_size=30, color=GREEN_C)

        shapes_group = VGroup(shape_a, shape_plus, shape_b).arrange(RIGHT, buff=0.4)
        shapes_group.move_to(ORIGIN + UP * 0.3)

        self.play(FadeIn(shapes_group), run_time=0.4)
        self.wait(0.3)

        # Flash red
        self.play(
            shape_a.animate.set_color(RED),
            shape_b.animate.set_color(RED),
            run_time=0.3,
        )

        error_text = Text("Incompatible shapes!", font_size=32, color=RED)
        error_text.next_to(shapes_group, DOWN, buff=0.5)

        # Dim 0: 4 vs 2 (not 1, not equal) -> fail
        rule_text = Text(
            "dim 0:  4 vs 2  -- not equal, neither is 1",
            font_size=20,
            color=GREY_B,
        )
        rule_text.next_to(error_text, DOWN, buff=0.3)

        self.play(FadeIn(error_text, scale=1.2), run_time=0.4)
        self.play(FadeIn(rule_text), run_time=0.3)
        self.wait(1.2)

        self.play(
            FadeOut(VGroup(fail_title, shapes_group, error_text, rule_text)),
            run_time=0.5,
        )
