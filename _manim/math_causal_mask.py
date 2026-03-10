from manimlib import *
import numpy as np


class CausalMask(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Step 1: Title ---
        title = Text("Causal Masking in Transformers", font_size=34, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)
        self.wait(0.3)

        # --- Step 2: Build 5x5 grid with token labels ---
        tokens = ["The", "cat", "sat", "on", "mat"]
        n = len(tokens)
        cell_size = 0.72

        # Raw attention scores (lower triangle + diagonal) - random but deterministic
        np.random.seed(7)
        raw_scores = np.round(np.random.uniform(0.3, 2.5, size=(n, n)), 2)

        # Column labels (Keys) on top
        col_labels = VGroup(*[
            Text(t, font_size=18, color="#4ecdc4") for t in tokens
        ])
        # Row labels (Queries) on left
        row_labels = VGroup(*[
            Text(t, font_size=18, color="#ff6b6b") for t in tokens
        ])

        # Build grid of cells (initially empty)
        grid_cells = [[None for _ in range(n)] for _ in range(n)]
        grid_group = VGroup()

        for i in range(n):
            for j in range(n):
                rect = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=GREY_B, stroke_width=1.2,
                    fill_color="#1a1a2e", fill_opacity=1.0,
                )
                grid_cells[i][j] = rect
                grid_group.add(rect)

        grid_group.arrange_in_grid(n_rows=n, n_cols=n, buff=0.04)
        grid_group.move_to(ORIGIN + DOWN * 0.3)

        # Position row labels to the left of each row
        for i in range(n):
            cell = grid_cells[i][0]
            row_labels[i].next_to(cell, LEFT, buff=0.2)

        # Position column labels above each column
        for j in range(n):
            cell = grid_cells[0][j]
            col_labels[j].next_to(cell, UP, buff=0.15)

        # Axis labels
        key_label = Text("Keys", font_size=16, color="#4ecdc4")
        key_label.next_to(col_labels, UP, buff=0.15)
        query_label = Text("Queries", font_size=16, color="#ff6b6b")
        query_label.next_to(row_labels, UP, buff=0.15)

        self.play(
            FadeIn(grid_group),
            FadeIn(col_labels),
            FadeIn(row_labels),
            FadeIn(key_label),
            FadeIn(query_label),
            run_time=0.8,
        )
        self.wait(0.3)

        # --- Step 3: Fill lower triangle cell by cell with attention scores ---
        lower_label = Text("Raw attention scores (Q * K^T)", font_size=20, color="#ffd93d")
        lower_label.next_to(grid_group, DOWN, buff=0.4)
        self.play(Write(lower_label), run_time=0.5)

        # Color intensity based on score value
        score_min = 0.3
        score_max = 2.5
        score_texts = [[None for _ in range(n)] for _ in range(n)]

        for i in range(n):
            anims = []
            for j in range(i + 1):
                val = raw_scores[i][j]
                intensity = (val - score_min) / (score_max - score_min)
                fill_color = interpolate_color(
                    "#1e3a5f", "#4ecdc4", intensity
                )
                cell = grid_cells[i][j]
                txt = Text(f"{val:.2f}", font_size=13, color=WHITE)
                txt.move_to(cell.get_center())
                score_texts[i][j] = txt

                anims.append(cell.animate.set_fill(fill_color, opacity=0.85))
                anims.append(FadeIn(txt))

            self.play(*anims, run_time=0.4)

        self.wait(0.3)

        # --- Step 4: Fill upper triangle with -inf (masking) ---
        mask_label = Text("Mask future positions with -inf", font_size=20, color="#e74c3c")
        mask_label.next_to(lower_label, DOWN, buff=0.2)
        self.play(Write(mask_label), run_time=0.5)

        mask_texts = [[None for _ in range(n)] for _ in range(n)]

        for i in range(n):
            anims = []
            for j in range(i + 1, n):
                cell = grid_cells[i][j]
                txt = Text("-inf", font_size=13, color="#ff4444")
                txt.move_to(cell.get_center())
                mask_texts[i][j] = txt

                anims.append(cell.animate.set_fill("#3d0000", opacity=0.9))
                anims.append(FadeIn(txt))

            if anims:
                self.play(*anims, run_time=0.35)

        self.wait(0.5)

        # --- Step 5: Apply softmax - transform the grid ---
        # Compute softmax per row with masking
        def masked_softmax(scores, mask_size):
            """Apply softmax to lower triangle only."""
            result = np.zeros((mask_size, mask_size))
            for i in range(mask_size):
                row = scores[i, : i + 1]
                e_x = np.exp(row - np.max(row))
                result[i, : i + 1] = e_x / e_x.sum()
            return np.round(result, 3)

        softmax_vals = masked_softmax(raw_scores, n)

        softmax_label = Text("After softmax: probabilities (rows sum to 1)", font_size=20, color="#a8e6cf")
        # Fade out old labels, bring in new
        self.play(
            FadeOut(lower_label),
            FadeOut(mask_label),
            run_time=0.3,
        )
        softmax_label.next_to(grid_group, DOWN, buff=0.4)
        self.play(Write(softmax_label), run_time=0.5)

        # Transform lower triangle values to softmax probabilities
        for i in range(n):
            anims = []
            for j in range(i + 1):
                val = softmax_vals[i][j]
                intensity = val  # probabilities 0-1 map directly to intensity
                fill_color = interpolate_color(
                    "#1e3a5f", "#00cc88", intensity
                )
                cell = grid_cells[i][j]
                new_txt = Text(f"{val:.3f}", font_size=12, color=WHITE)
                new_txt.move_to(cell.get_center())

                anims.append(cell.animate.set_fill(fill_color, opacity=0.85))
                anims.append(FadeOut(score_texts[i][j]))
                anims.append(FadeIn(new_txt))

            # Upper triangle becomes 0
            for j in range(i + 1, n):
                cell = grid_cells[i][j]
                new_txt = Text("0", font_size=13, color="#666666")
                new_txt.move_to(cell.get_center())

                anims.append(cell.animate.set_fill("#1a1a2e", opacity=0.6))
                anims.append(FadeOut(mask_texts[i][j]))
                anims.append(FadeIn(new_txt))

            self.play(*anims, run_time=0.45)

        self.wait(0.5)

        # --- Step 6: Explanation text ---
        explanation = Text(
            "Each token can only attend to previous tokens and itself",
            font_size=22,
            color="#dda0dd",
        )
        explanation.next_to(softmax_label, DOWN, buff=0.25)
        self.play(Write(explanation), run_time=1.0)
        self.wait(1.5)
