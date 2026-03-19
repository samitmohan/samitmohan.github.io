from manim import *
import numpy as np


class CudaParallel(Scene):
    """CPU vs GPU matrix multiply: sequential vs parallel element computation."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("CPU vs GPU: Matrix Multiply", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)

        # Labels for left (CPU) and right (GPU) sides
        cpu_label = Text("CPU (Sequential)", font_size=20, color="#ff6b6b")
        gpu_label = Text("GPU (Parallel)", font_size=20, color="#a8e6cf")
        cpu_label.move_to(LEFT * 3.5 + UP * 2.2)
        gpu_label.move_to(RIGHT * 3.5 + UP * 2.2)
        self.play(FadeIn(cpu_label), FadeIn(gpu_label), run_time=0.4)

        # Grid parameters
        grid_size = 4
        cell_size = 0.45
        gap = 0.05

        def make_grid(center):
            """Create a 4x4 grid of squares."""
            cells = VGroup()
            for r in range(grid_size):
                for c in range(grid_size):
                    sq = Square(
                        side_length=cell_size,
                        fill_color="#2a2a4e",
                        fill_opacity=0.4,
                        stroke_color=GREY,
                        stroke_width=1,
                    )
                    x = center[0] + (c - grid_size / 2 + 0.5) * (cell_size + gap)
                    y = center[1] + (grid_size / 2 - 0.5 - r) * (cell_size + gap)
                    sq.move_to(RIGHT * x + UP * y)
                    cells.add(sq)
            return cells

        cpu_center = np.array([-3.5, 0.4, 0])
        gpu_center = np.array([3.5, 0.4, 0])

        cpu_grid = make_grid(cpu_center)
        gpu_grid = make_grid(gpu_center)

        # "Result C" labels
        cpu_c_label = Text("Result C", font_size=14, color=GREY_B)
        cpu_c_label.next_to(cpu_grid, UP, buff=0.15)
        gpu_c_label = Text("Result C", font_size=14, color=GREY_B)
        gpu_c_label.next_to(gpu_grid, UP, buff=0.15)

        self.play(
            FadeIn(cpu_grid), FadeIn(gpu_grid),
            FadeIn(cpu_c_label), FadeIn(gpu_c_label),
            run_time=0.5,
        )

        # Divider line
        divider = DashedLine(
            start=UP * 2.5, end=DOWN * 3.5,
            color=GREY, stroke_width=1, dash_length=0.1
        )
        self.play(Create(divider), run_time=0.2)

        # --- CPU: Sequential highlighting ---
        cpu_timer = Text("t = 0", font_size=16, color="#ffd93d")
        cpu_timer.next_to(cpu_grid, DOWN, buff=0.3)
        self.play(FadeIn(cpu_timer), run_time=0.2)

        highlight_color = "#ff6b6b"
        done_color = "#4ecdc4"

        for idx in range(grid_size * grid_size):
            new_timer = Text(f"t = {idx + 1}", font_size=16, color="#ffd93d")
            new_timer.move_to(cpu_timer.get_center())

            self.play(
                cpu_grid[idx].animate.set_fill(highlight_color, opacity=0.8),
                Transform(cpu_timer, new_timer),
                run_time=0.12,
            )
            # Immediately mark as done
            cpu_grid[idx].set_fill(done_color, opacity=0.6)

        cpu_final_time = Text("t = 16 cycles", font_size=16, color="#ff6b6b")
        cpu_final_time.move_to(cpu_timer.get_center())
        self.play(Transform(cpu_timer, cpu_final_time), run_time=0.3)

        self.wait(0.3)

        # --- GPU: Parallel highlighting (all at once) ---
        gpu_timer = Text("t = 0", font_size=16, color="#ffd93d")
        gpu_timer.next_to(gpu_grid, DOWN, buff=0.3)
        self.play(FadeIn(gpu_timer), run_time=0.2)

        # Flash all cells simultaneously
        gpu_flash_anims = [
            gpu_grid[i].animate.set_fill("#a8e6cf", opacity=0.8)
            for i in range(grid_size * grid_size)
        ]
        gpu_new_timer = Text("t = 1", font_size=16, color="#ffd93d")
        gpu_new_timer.move_to(gpu_timer.get_center())

        self.play(*gpu_flash_anims, Transform(gpu_timer, gpu_new_timer), run_time=0.6)

        gpu_final_time = Text("t = 1 cycle", font_size=16, color="#a8e6cf")
        gpu_final_time.move_to(gpu_timer.get_center())
        self.play(Transform(gpu_timer, gpu_final_time), run_time=0.3)

        self.wait(0.5)

        # --- Speedup comparison at bottom ---
        comparison_box = RoundedRectangle(
            width=8, height=0.8,
            corner_radius=0.15,
            fill_color="#2a2a4e", fill_opacity=0.5,
            stroke_color="#ffd93d", stroke_width=2,
        )
        comparison_box.to_edge(DOWN, buff=0.3)

        comparison_text = Text(
            "16 elements: CPU = 16 steps, GPU = 1 step  (16x speedup)",
            font_size=18, color="#ffd93d",
        )
        comparison_text.move_to(comparison_box.get_center())

        self.play(FadeIn(comparison_box), Write(comparison_text), run_time=0.6)

        # Scale note
        scale_note = Text(
            "Real GPUs: thousands of cores, millions of elements",
            font_size=14, color=GREY_B,
        )
        scale_note.next_to(comparison_box, UP, buff=0.12)
        self.play(FadeIn(scale_note), run_time=0.4)

        self.wait(2.0)
