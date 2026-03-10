from manim import *
import numpy as np

class SelfAttention(Scene):
    def construct(self):
        self.camera.background_color = "#1a1a2e"

        # --- Step 1: Show sentence tokens ---
        title = Text("Self-Attention", font_size=32, color=WHITE).to_edge(UP, buff=0.3)
        tokens = ["The", "crane", "flew"]
        token_texts = VGroup(*[
            Text(t, font_size=28, color=YELLOW) for t in tokens
        ]).arrange(RIGHT, buff=0.6).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=0.5)
        self.play(FadeIn(token_texts), run_time=0.5)
        self.wait(0.3)

        # --- Step 2: Q, K, V matrices ---
        np.random.seed(42)
        q_vals = np.round(np.random.randn(3, 4) * 0.5, 1)
        k_vals = np.round(np.random.randn(3, 4) * 0.5, 1)
        v_vals = np.round(np.random.randn(3, 4) * 0.5, 1)

        def make_matrix_grid(values, label_text, color, scale=0.35):
            rows, cols = values.shape
            cells = VGroup()
            for i in range(rows):
                for j in range(cols):
                    rect = Rectangle(
                        width=0.55, height=0.35,
                        stroke_color=color, stroke_width=1.5,
                        fill_color=color, fill_opacity=0.15
                    )
                    val_text = Text(f"{values[i,j]:.1f}", font_size=14, color=WHITE)
                    val_text.move_to(rect.get_center())
                    cells.add(VGroup(rect, val_text))
            grid = cells.arrange_in_grid(rows=rows, cols=cols, buff=0.05)
            label = Text(label_text, font_size=22, color=color, weight=BOLD)
            label.next_to(grid, UP, buff=0.15)
            return VGroup(label, grid).scale(scale)

        q_matrix = make_matrix_grid(q_vals, "Q (Query)", "#ff6b6b", scale=0.55)
        k_matrix = make_matrix_grid(k_vals, "K (Key)", "#4ecdc4", scale=0.55)
        v_matrix = make_matrix_grid(v_vals, "V (Value)", "#45b7d1", scale=0.55)

        matrices = VGroup(q_matrix, k_matrix, v_matrix).arrange(RIGHT, buff=0.4)
        matrices.next_to(token_texts, DOWN, buff=0.35)

        self.play(
            FadeIn(q_matrix, shift=UP * 0.3),
            FadeIn(k_matrix, shift=UP * 0.3),
            FadeIn(v_matrix, shift=UP * 0.3),
            run_time=0.8
        )
        self.wait(0.3)

        # --- Step 3: Compute Q * K^T -> attention scores ---
        # Shift everything up
        self.play(
            VGroup(title, token_texts, matrices).animate.shift(UP * 0.8),
            run_time=0.4
        )

        # Compute actual scores
        scores = q_vals @ k_vals.T  # 3x3
        score_vals = np.round(scores, 2)

        qkt_label = Text("Q * K^T  =  Attention Scores", font_size=22, color="#ffd93d")

        def make_heatmap(values, scale=0.55):
            rows, cols = values.shape
            min_v, max_v = values.min(), values.max()
            cells = VGroup()
            for i in range(rows):
                for j in range(cols):
                    # Normalize for color intensity
                    if max_v - min_v > 0:
                        intensity = (values[i, j] - min_v) / (max_v - min_v)
                    else:
                        intensity = 0.5
                    fill_color = interpolate_color(
                        ManimColor("#1a1a2e"), ManimColor("#ff6b6b"), intensity
                    )
                    rect = Rectangle(
                        width=0.7, height=0.4,
                        stroke_color=WHITE, stroke_width=1,
                        fill_color=fill_color, fill_opacity=0.8
                    )
                    val_text = Text(f"{values[i,j]:.2f}", font_size=13, color=WHITE)
                    val_text.move_to(rect.get_center())
                    cells.add(VGroup(rect, val_text))
            grid = cells.arrange_in_grid(rows=rows, cols=cols, buff=0.05)
            return grid.scale(scale)

        score_grid = make_heatmap(score_vals)
        qkt_label.next_to(matrices, DOWN, buff=0.35)
        score_grid.next_to(qkt_label, DOWN, buff=0.2)

        self.play(Write(qkt_label), run_time=0.5)
        self.play(FadeIn(score_grid), run_time=0.6)
        self.wait(0.3)

        # --- Step 4: Show scaling by sqrt(d_k) ---
        d_k = 4
        scaled_scores = scores / np.sqrt(d_k)
        scaled_vals = np.round(scaled_scores, 2)

        scale_text = Text("/ sqrt(d_k)  where d_k = 4", font_size=18, color="#ffd93d")
        scale_text.next_to(score_grid, RIGHT, buff=0.3)

        # Create new scaled heatmap
        scaled_grid = make_heatmap(scaled_vals)
        scaled_grid.move_to(score_grid.get_center())

        self.play(Write(scale_text), run_time=0.4)
        self.play(ReplacementTransform(score_grid, scaled_grid), run_time=0.6)
        self.wait(0.3)

        # --- Step 5: Softmax row by row ---
        def softmax_fn(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        attn_weights = np.round(softmax_fn(scaled_scores), 2)

        softmax_label = Text("softmax (rows sum to 1)", font_size=18, color="#a8e6cf")
        softmax_label.next_to(scale_text, DOWN, buff=0.2)

        attn_grid = make_heatmap(attn_weights)
        attn_grid.move_to(scaled_grid.get_center())

        self.play(Write(softmax_label), run_time=0.4)
        self.play(ReplacementTransform(scaled_grid, attn_grid), run_time=0.8)
        self.wait(0.3)

        # --- Step 6: Multiply by V ---
        output_vals = np.round(attn_weights @ v_vals, 2)

        # Clear lower area and show final step
        mult_label = Text("x V  =  Output", font_size=20, color="#45b7d1")
        mult_label.next_to(attn_grid, DOWN, buff=0.3)

        def make_output_grid(values, color="#45b7d1", scale=0.55):
            rows, cols = values.shape
            cells = VGroup()
            for i in range(rows):
                for j in range(cols):
                    rect = Rectangle(
                        width=0.55, height=0.35,
                        stroke_color=color, stroke_width=1.5,
                        fill_color=color, fill_opacity=0.2
                    )
                    val_text = Text(f"{values[i,j]:.2f}", font_size=13, color=WHITE)
                    val_text.move_to(rect.get_center())
                    cells.add(VGroup(rect, val_text))
            grid = cells.arrange_in_grid(rows=rows, cols=cols, buff=0.05)
            return grid.scale(scale)

        output_grid = make_output_grid(output_vals)
        output_grid.next_to(mult_label, DOWN, buff=0.2)

        self.play(Write(mult_label), run_time=0.4)
        self.play(FadeIn(output_grid, shift=UP * 0.2), run_time=0.6)
        self.wait(0.3)

        # --- Step 7: Final label ---
        # Token labels for output
        out_tokens = VGroup(*[
            Text(t, font_size=16, color=YELLOW) for t in tokens
        ])
        for i, tok in enumerate(out_tokens):
            # Position to the left of each row in output grid
            row_start = output_grid[i * 4]  # first cell of row i
            tok.next_to(row_start, LEFT, buff=0.15)

        final_label = Text(
            "Each output row = weighted combination of V",
            font_size=18, color="#dda0dd"
        )
        final_label.next_to(output_grid, DOWN, buff=0.25)

        self.play(FadeIn(out_tokens), Write(final_label), run_time=0.6)
        self.wait(1.0)
