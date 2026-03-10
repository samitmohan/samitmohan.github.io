from manimlib import *


class MLACompression(Scene):
    """Multi-Head Latent Attention: KV cache compression and reconstruction.

    Shows how MLA compresses K/V into a small latent vector, stores it
    in a tiny KV cache, then reconstructs K and V on demand - with a
    side-by-side memory comparison at the end.
    """

    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # ── 1. Title ──
        title = Text("Multi-Head Latent Attention", font_size=32, color=WHITE)
        subtitle = Text("KV Cache Compression", font_size=22, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.2)
        title_group = VGroup(title, subtitle).to_edge(UP, buff=0.3)
        self.play(FadeIn(title_group), run_time=0.6)

        # ── 2. K vector (tall bar) on the left ──
        k_bar = Rectangle(width=0.8, height=4.0, color=BLUE_C,
                          fill_color=BLUE_D, fill_opacity=0.7, stroke_width=2)
        k_bar.move_to(LEFT * 4.5 + DOWN * 0.3)
        k_label = Text("K (4096-dim)", font_size=18, color=WHITE)
        k_label.next_to(k_bar, DOWN, buff=0.15)

        self.play(FadeIn(k_bar), Write(k_label), run_time=0.6)
        self.wait(0.3)

        # ── 3. Arrow W_down + small latent bar ──
        latent_bar = Rectangle(width=0.8, height=0.8, color=GREEN_C,
                               fill_color=GREEN_D, fill_opacity=0.7, stroke_width=2)
        latent_bar.move_to(LEFT * 1.0 + DOWN * 0.3)
        latent_label = Text("Latent\n(512-dim)", font_size=16, color=WHITE)
        latent_label.next_to(latent_bar, DOWN, buff=0.15)

        arrow_down = Arrow(
            k_bar.get_right(), latent_bar.get_left(),
            buff=0.15, color=YELLOW_C, stroke_width=3
        )
        wdown_label = Text("W_down", font_size=16, color=YELLOW_C)
        wdown_label.next_to(arrow_down, UP, buff=0.1)

        self.play(
            GrowArrow(arrow_down), Write(wdown_label),
            run_time=0.5
        )

        # ── 4. Animate compression: tall bar shrinks into small bar ──
        k_bar_copy = k_bar.copy()
        self.play(
            Transform(k_bar_copy, latent_bar),
            FadeIn(latent_label),
            run_time=1.0
        )
        self.remove(k_bar_copy)
        self.add(latent_bar)
        self.wait(0.3)

        # ── 5. KV Cache box storing the latent ──
        cache_box = Rectangle(width=1.6, height=1.4, color=TEAL_C,
                              fill_color=TEAL_E, fill_opacity=0.4, stroke_width=2)
        cache_box.move_to(LEFT * 1.0 + UP * 1.8)
        cache_label = Text("KV Cache", font_size=16, color=TEAL_B)
        cache_label.next_to(cache_box, UP, buff=0.1)

        latent_in_cache = latent_bar.copy().scale(0.7)
        latent_in_cache.move_to(cache_box.get_center())

        arrow_to_cache = Arrow(
            latent_bar.get_top(), cache_box.get_bottom(),
            buff=0.1, color=TEAL_C, stroke_width=2
        )

        self.play(
            FadeIn(cache_box), Write(cache_label),
            GrowArrow(arrow_to_cache),
            TransformFromCopy(latent_bar, latent_in_cache),
            run_time=0.8
        )
        self.wait(0.3)

        # ── 6. Reconstruction: latent -> K' and V' ──
        recon_k = Rectangle(width=0.6, height=3.0, color=BLUE_C,
                            fill_color=BLUE_D, fill_opacity=0.6, stroke_width=2)
        recon_k.move_to(RIGHT * 2.5 + DOWN * 0.3)
        recon_k_label = Text("K'", font_size=18, color=BLUE_C)
        recon_k_label.next_to(recon_k, DOWN, buff=0.1)

        recon_v = Rectangle(width=0.6, height=3.0, color=PURPLE_B,
                            fill_color=PURPLE_C, fill_opacity=0.6, stroke_width=2)
        recon_v.move_to(RIGHT * 4.0 + DOWN * 0.3)
        recon_v_label = Text("V'", font_size=18, color=PURPLE_B)
        recon_v_label.next_to(recon_v, DOWN, buff=0.1)

        arrow_up_k = Arrow(
            latent_bar.get_right(), recon_k.get_left(),
            buff=0.15, color=BLUE_C, stroke_width=3
        )
        wupk_label = Text("W_up_k", font_size=14, color=BLUE_C)
        wupk_label.next_to(arrow_up_k, UP, buff=0.1)

        arrow_up_v = Arrow(
            latent_bar.get_right() + UP * 0.2,
            recon_v.get_left(),
            buff=0.15, color=PURPLE_B, stroke_width=3
        )
        wupv_label = Text("W_up_v", font_size=14, color=PURPLE_B)
        wupv_label.next_to(arrow_up_v, UP, buff=0.1)

        need_label = Text("When attention is needed:", font_size=16, color=GREY_B)
        need_label.move_to(RIGHT * 3.2 + UP * 2.2)

        self.play(FadeIn(need_label), run_time=0.3)
        self.play(
            GrowArrow(arrow_up_k), Write(wupk_label),
            GrowArrow(arrow_up_v), Write(wupv_label),
            run_time=0.6
        )
        self.play(
            FadeIn(recon_k), Write(recon_k_label),
            FadeIn(recon_v), Write(recon_v_label),
            run_time=0.6
        )
        self.wait(0.5)

        # ── 7. Transition to memory comparison ──
        all_prev = VGroup(
            title_group, k_bar, k_label, arrow_down, wdown_label,
            latent_bar, latent_label, cache_box, cache_label,
            latent_in_cache, arrow_to_cache, need_label,
            arrow_up_k, wupk_label, arrow_up_v, wupv_label,
            recon_k, recon_k_label, recon_v, recon_v_label,
        )
        self.play(FadeOut(all_prev), run_time=0.5)

        # ── 8. Side-by-side memory comparison ──
        comp_title = Text("KV Cache Memory Comparison", font_size=28, color=WHITE)
        comp_title.to_edge(UP, buff=0.5)
        self.play(Write(comp_title), run_time=0.4)

        # Standard MHA bar (large, red)
        mha_bar = Rectangle(width=1.2, height=4.5, color=RED_C,
                            fill_color=RED_D, fill_opacity=0.7, stroke_width=2)
        mha_bar.move_to(LEFT * 2.5 + DOWN * 0.2)
        mha_label = Text("Standard\nMHA", font_size=18, color=RED_C)
        mha_label.next_to(mha_bar, DOWN, buff=0.15)
        mha_size = Text("8192 per token", font_size=14, color=GREY_B)
        mha_size.next_to(mha_bar, UP, buff=0.1)

        # MLA bar (small, green)
        mla_bar = Rectangle(width=1.2, height=0.5, color=GREEN_C,
                             fill_color=GREEN_D, fill_opacity=0.7, stroke_width=2)
        mla_bar.move_to(RIGHT * 2.5 + DOWN * 0.2)
        # Align bottoms
        mla_bar.align_to(mha_bar, DOWN)
        mla_label = Text("MLA", font_size=18, color=GREEN_C)
        mla_label.next_to(mla_bar, DOWN, buff=0.15)
        mla_size = Text("512 per token", font_size=14, color=GREY_B)
        mla_size.next_to(mla_bar, UP, buff=0.1)

        self.play(
            FadeIn(mha_bar), Write(mha_label), Write(mha_size),
            FadeIn(mla_bar), Write(mla_label), Write(mla_size),
            run_time=0.8
        )

        # Reduction label
        reduction = Text("93% reduction", font_size=30, color=GREEN_A,
                         weight=BOLD)
        reduction.move_to(DOWN * 2.8)
        self.play(Write(reduction), run_time=0.6)
        self.wait(1.5)
