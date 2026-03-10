from manim import *
import numpy as np


class YOLOGrid(Scene):
    """YOLO grid detection visualization.

    Shows an image divided into SxS grid, highlights cells,
    and draws bounding box predictions with confidence scores.
    """

    def construct(self):
        self.camera.background_color = WHITE

        title = Text("YOLO: Grid-Based Detection", font_size=32, color=BLACK, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # -- Image area (represented as a square) --
        S = 7  # grid size
        img_size = 5.0
        cell_size = img_size / S

        # Image background
        img_rect = Square(side_length=img_size, color=GREY_B, fill_color=GREY_A,
                          fill_opacity=0.15, stroke_width=2)
        img_rect.move_to(LEFT * 1.5)
        img_corner = img_rect.get_corner(UL)

        self.play(Create(img_rect), run_time=0.3)

        # Draw the grid
        grid_lines = VGroup()
        for i in range(1, S):
            # Vertical lines
            x = img_corner[0] + i * cell_size
            vline = Line(
                start=[x, img_corner[1], 0],
                end=[x, img_corner[1] - img_size, 0],
                color=GREY_C, stroke_width=0.8,
            )
            grid_lines.add(vline)
            # Horizontal lines
            y = img_corner[1] - i * cell_size
            hline = Line(
                start=[img_corner[0], y, 0],
                end=[img_corner[0] + img_size, y, 0],
                color=GREY_C, stroke_width=0.8,
            )
            grid_lines.add(hline)

        grid_label = Text(f"{S}x{S} Grid", font_size=20, color=BLACK)
        grid_label.next_to(img_rect, DOWN, buff=0.2)
        self.play(Create(grid_lines), Write(grid_label), run_time=1)
        self.wait(0.3)

        # -- Helper to get cell center --
        def cell_center(row, col):
            x = img_corner[0] + (col + 0.5) * cell_size
            y = img_corner[1] - (row + 0.5) * cell_size
            return np.array([x, y, 0])

        # -- Highlight cells one row at a time (quick sweep) --
        highlight = Square(side_length=cell_size, color=YELLOW, fill_color=YELLOW,
                           fill_opacity=0.4, stroke_width=1)
        highlight.move_to(cell_center(0, 0))
        self.play(Create(highlight), run_time=0.2)

        # Quick sweep across a few cells
        sweep_cells = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 2), (3, 3), (4, 4), (3, 4)]
        for r, c in sweep_cells:
            self.play(highlight.animate.move_to(cell_center(r, c)), run_time=0.08)

        self.play(FadeOut(highlight), run_time=0.2)

        # -- Info panel on the right --
        info_title = Text("Each cell predicts:", font_size=20, color=BLACK, weight=BOLD)
        info_items = VGroup(
            Text("- B bounding boxes (B=2)", font_size=16, color=BLUE_D),
            Text("- Confidence per box", font_size=16, color=RED_D),
            Text("- C class probabilities", font_size=16, color=GREEN_D),
        )
        info_items.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        info_panel = VGroup(info_title, info_items)
        info_panel.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        info_panel.to_edge(RIGHT, buff=0.3)
        info_panel.shift(UP * 1.5)

        self.play(Write(info_title), run_time=0.3)
        for item in info_items:
            self.play(Write(item), run_time=0.2)

        # -- Show bounding box predictions for specific cells --

        # Object 1: "Dog" around cell (3, 2)
        # Two bounding box predictions
        cell_r1, cell_c1 = 3, 2
        cell_highlight1 = Square(side_length=cell_size, color=GREEN_D, fill_color=GREEN,
                                  fill_opacity=0.3, stroke_width=2)
        cell_highlight1.move_to(cell_center(cell_r1, cell_c1))
        cell_label1 = Text("Cell (3,2)", font_size=14, color=GREEN_D)
        cell_label1.next_to(cell_highlight1, DOWN, buff=0.05)

        self.play(Create(cell_highlight1), Write(cell_label1), run_time=0.3)

        cc1 = cell_center(cell_r1, cell_c1)
        # Box 1: larger, higher confidence
        box1_1 = Rectangle(
            width=cell_size * 2.8, height=cell_size * 2.2,
            color=GREEN_D, stroke_width=2.5,
        )
        box1_1.move_to(cc1 + DOWN * 0.1 * cell_size + LEFT * 0.1 * cell_size)
        conf1_1 = Text("0.92", font_size=14, color=GREEN_D, weight=BOLD)
        conf1_1.next_to(box1_1, UP, buff=0.05, aligned_edge=LEFT)
        class1_1 = Text("Dog", font_size=14, color=GREEN_D)
        class1_1.next_to(conf1_1, RIGHT, buff=0.1)

        # Box 2: slightly offset, lower confidence
        box1_2 = Rectangle(
            width=cell_size * 2.5, height=cell_size * 1.8,
            color=GREEN_C, stroke_width=1.5, stroke_opacity=0.6,
        )
        box1_2.move_to(cc1 + UP * 0.15 * cell_size + RIGHT * 0.2 * cell_size)
        conf1_2 = Text("0.31", font_size=12, color=GREEN_C)
        conf1_2.next_to(box1_2, UR, buff=0.05)

        self.play(Create(box1_1), Write(conf1_1), Write(class1_1), run_time=0.4)
        self.play(Create(box1_2), Write(conf1_2), run_time=0.3)
        self.wait(0.3)

        # Object 2: "Car" around cell (4, 5)
        cell_r2, cell_c2 = 4, 5
        cell_highlight2 = Square(side_length=cell_size, color=BLUE_D, fill_color=BLUE,
                                  fill_opacity=0.3, stroke_width=2)
        cell_highlight2.move_to(cell_center(cell_r2, cell_c2))
        cell_label2 = Text("Cell (4,5)", font_size=14, color=BLUE_D)
        cell_label2.next_to(cell_highlight2, DOWN, buff=0.05)

        self.play(Create(cell_highlight2), Write(cell_label2), run_time=0.3)

        cc2 = cell_center(cell_r2, cell_c2)
        box2_1 = Rectangle(
            width=cell_size * 2.2, height=cell_size * 1.5,
            color=BLUE_D, stroke_width=2.5,
        )
        box2_1.move_to(cc2 + RIGHT * 0.1 * cell_size)
        conf2_1 = Text("0.88", font_size=14, color=BLUE_D, weight=BOLD)
        conf2_1.next_to(box2_1, UP, buff=0.05, aligned_edge=LEFT)
        class2_1 = Text("Car", font_size=14, color=BLUE_D)
        class2_1.next_to(conf2_1, RIGHT, buff=0.1)

        box2_2 = Rectangle(
            width=cell_size * 1.8, height=cell_size * 1.3,
            color=BLUE_C, stroke_width=1.5, stroke_opacity=0.6,
        )
        box2_2.move_to(cc2 + DOWN * 0.1 * cell_size)
        conf2_2 = Text("0.25", font_size=12, color=BLUE_C)
        conf2_2.next_to(box2_2, UR, buff=0.05)

        self.play(Create(box2_1), Write(conf2_1), Write(class2_1), run_time=0.4)
        self.play(Create(box2_2), Write(conf2_2), run_time=0.3)

        # -- Output tensor shape --
        output_text = Text(
            f"Output tensor: {S} x {S} x (B*5 + C)",
            font_size=20, color=BLACK, weight=BOLD,
        )
        output_text.to_edge(DOWN, buff=0.3)
        output_box = SurroundingRectangle(output_text, color=BLACK, buff=0.15,
                                           corner_radius=0.1, stroke_width=1)
        self.play(Write(output_text), Create(output_box), run_time=0.5)
        self.wait(1)
