from manimlib import *

DARK_BG = "#1a1a2e"

class VanishingGradient(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Title ---
        title = Text("The Vanishing Gradient Problem", font_size=40, color=WHITE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.2)
        self.wait(0.5)

        # --- Build 6 layers + Loss ---
        num_layers = 6
        layer_width = 0.8
        layer_height = 1.6
        gap = 0.7  # gap between layer edge and next layer edge

        layers = VGroup()
        labels = VGroup()
        arrows_fwd = VGroup()

        total_width = num_layers * layer_width + (num_layers - 1) * gap
        start_x = -total_width / 2 + layer_width / 2

        for i in range(num_layers):
            x = start_x + i * (layer_width + gap)
            rect = RoundedRectangle(
                width=layer_width, height=layer_height,
                corner_radius=0.1,
                color=BLUE_C, fill_color=BLUE_E, fill_opacity=0.6,
                stroke_width=2,
            )
            rect.move_to([x, 0, 0])
            layers.add(rect)

            lbl = Text(f"L{i+1}", font_size=20, color=WHITE)
            lbl.move_to(rect.get_center())
            labels.add(lbl)

        # Forward arrows between layers
        for i in range(num_layers - 1):
            arr = Arrow(
                layers[i].get_right(), layers[i + 1].get_left(),
                buff=0.08, color=GREY_B, stroke_width=2, max_tip_length_to_length_ratio=0.15,
            )
            arrows_fwd.add(arr)

        # Loss label
        loss_label = Text("Loss", font_size=28, color=RED_C)
        loss_label.next_to(layers[-1], RIGHT, buff=0.5)

        loss_arrow = Arrow(
            layers[-1].get_right(), loss_label.get_left(),
            buff=0.1, color=RED_C, stroke_width=3, max_tip_length_to_length_ratio=0.2,
        )

        # Show layers, labels, arrows, loss
        self.play(
            *[FadeIn(l, shift=UP * 0.2) for l in layers],
            *[FadeIn(l) for l in labels],
            run_time=1.0,
        )
        self.play(
            *[GrowArrow(a) for a in arrows_fwd],
            run_time=0.7,
        )
        self.play(GrowArrow(loss_arrow), FadeIn(loss_label), run_time=0.6)
        self.wait(0.5)

        # --- Vanishing gradient backward pass ---
        gradient_values = [1.0, 0.25, 0.063, 0.016, 0.004, 0.001]
        # Colors: bright green fading to dark/transparent
        grad_colors = [
            "#00ff88",  # 1.0
            "#00cc66",  # 0.25
            "#009944",  # 0.063
            "#006622",  # 0.016
            "#003311",  # 0.004
            "#001a08",  # 0.001
        ]
        grad_widths = [8, 6, 4.5, 3, 2, 1]

        # Subtitle for backward pass
        bwd_text = Text("Backpropagation with sigmoid", font_size=24, color=YELLOW)
        bwd_text.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(bwd_text), run_time=0.5)

        # Animate gradient flowing backward layer by layer
        grad_arrows = VGroup()
        val_labels = VGroup()

        for i in range(num_layers):
            # Index in layers: going backward from layer 5 to layer 0
            layer_idx = num_layers - 1 - i
            grad_val = gradient_values[i]
            col = grad_colors[i]
            sw = grad_widths[i]

            # Value label above layer
            val_text = Text(f"{grad_val}", font_size=18, color=col)
            val_text.next_to(layers[layer_idx], UP, buff=0.15)
            val_labels.add(val_text)

            if i == 0:
                # First arrow: from loss back to last layer
                arr = Arrow(
                    loss_label.get_left() + LEFT * 0.05,
                    layers[layer_idx].get_right() + RIGHT * 0.05,
                    buff=0.0, color=col, stroke_width=sw,
                    max_tip_length_to_length_ratio=0.15,
                )
            else:
                prev_layer_idx = num_layers - i  # the layer to the right
                arr = Arrow(
                    layers[prev_layer_idx].get_left() + LEFT * 0.05,
                    layers[layer_idx].get_right() + RIGHT * 0.05,
                    buff=0.0, color=col, stroke_width=sw,
                    max_tip_length_to_length_ratio=0.15,
                )
            grad_arrows.add(arr)

            # Highlight the current layer briefly
            self.play(
                GrowArrow(arr),
                FadeIn(val_text),
                layers[layer_idx].animate.set_stroke(color=col, width=3),
                run_time=0.5,
            )
            # Reset layer color
            self.play(
                layers[layer_idx].animate.set_stroke(color=BLUE_C, width=2),
                run_time=0.2,
            )

        self.wait(0.5)

        # Show the chain of values
        chain_text = Text(
            "1.0 -> 0.25 -> 0.063 -> 0.016 -> 0.004 -> 0.001",
            font_size=22, color=YELLOW,
        )
        chain_text.to_edge(DOWN, buff=1.2)
        self.play(Write(chain_text), run_time=1.0)
        self.wait(0.5)

        shrink_text = Text(
            "After 5 layers with sigmoid: gradient shrinks 1000x",
            font_size=24, color=RED_C,
        )
        shrink_text.next_to(chain_text, DOWN, buff=0.3)
        self.play(Write(shrink_text), run_time=1.0)
        self.wait(1.5)

        # --- Clean up for fix demonstration ---
        self.play(
            FadeOut(grad_arrows),
            FadeOut(val_labels),
            FadeOut(chain_text),
            FadeOut(shrink_text),
            FadeOut(bwd_text),
            run_time=0.8,
        )

        # --- Fix: ReLU / Residual Connections ---
        fix_title = Text(
            "With ReLU or Residual Connections: gradient stays strong",
            font_size=24, color=GREEN_C,
        )
        fix_title.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(fix_title), run_time=0.6)

        # Strong gradient: stays bright green and thick
        fix_values = [1.0, 0.95, 0.90, 0.86, 0.82, 0.78]
        fix_color = "#00ff88"
        fix_widths = [8, 7.5, 7, 6.5, 6, 5.5]

        fix_arrows = VGroup()
        fix_val_labels = VGroup()

        for i in range(num_layers):
            layer_idx = num_layers - 1 - i
            grad_val = fix_values[i]
            sw = fix_widths[i]

            val_text = Text(f"{grad_val:.2f}", font_size=18, color=fix_color)
            val_text.next_to(layers[layer_idx], UP, buff=0.15)
            fix_val_labels.add(val_text)

            if i == 0:
                arr = Arrow(
                    loss_label.get_left() + LEFT * 0.05,
                    layers[layer_idx].get_right() + RIGHT * 0.05,
                    buff=0.0, color=fix_color, stroke_width=sw,
                    max_tip_length_to_length_ratio=0.15,
                )
            else:
                prev_layer_idx = num_layers - i
                arr = Arrow(
                    layers[prev_layer_idx].get_left() + LEFT * 0.05,
                    layers[layer_idx].get_right() + RIGHT * 0.05,
                    buff=0.0, color=fix_color, stroke_width=sw,
                    max_tip_length_to_length_ratio=0.15,
                )
            fix_arrows.add(arr)

            self.play(
                GrowArrow(arr),
                FadeIn(val_text),
                layers[layer_idx].animate.set_stroke(color=GREEN_C, width=3),
                run_time=0.4,
            )
            self.play(
                layers[layer_idx].animate.set_stroke(color=BLUE_C, width=2),
                run_time=0.15,
            )

        # Show fix chain
        fix_chain = Text(
            "1.00 -> 0.95 -> 0.90 -> 0.86 -> 0.82 -> 0.78",
            font_size=22, color=GREEN_C,
        )
        fix_chain.to_edge(DOWN, buff=1.2)
        self.play(Write(fix_chain), run_time=0.8)

        fix_note = Text(
            "Gradient stays strong - early layers keep learning!",
            font_size=24, color=GREEN_A,
        )
        fix_note.next_to(fix_chain, DOWN, buff=0.3)
        self.play(Write(fix_note), run_time=0.8)
        self.wait(2)

        # Fade everything out
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.0,
        )
