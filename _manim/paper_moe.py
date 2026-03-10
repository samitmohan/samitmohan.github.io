from manimlib import *


class MixtureOfExperts(Scene):
    """Mixture of Experts routing animation.

    Shows how a router selects top-2 experts per token, with two
    different tokens activating different expert pairs.
    """

    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # ── 1. Title ──
        title = Text("Mixture of Experts", font_size=36, color=WHITE, weight=BOLD)
        subtitle = Text("Sparse Expert Routing", font_size=22, color=GREY_B)
        subtitle.next_to(title, DOWN, buff=0.2)
        title_group = VGroup(title, subtitle).to_edge(UP, buff=0.3)
        self.play(FadeIn(title_group), run_time=0.6)

        # ── 2. Expert boxes (E1-E8) ──
        expert_colors_inactive = "#444466"
        expert_rects = []
        expert_labels = []
        for i in range(8):
            rect = Rectangle(
                width=0.8, height=0.6, color=WHITE,
                fill_color=expert_colors_inactive, fill_opacity=0.8,
                stroke_width=1.5,
            )
            label = Text(f"E{i+1}", font_size=14, color=WHITE)
            label.move_to(rect.get_center())
            expert_rects.append(rect)
            expert_labels.append(label)

        experts_group = VGroup(*expert_rects).arrange(RIGHT, buff=0.25)
        experts_group.move_to(DOWN * 1.5)

        for label, rect in zip(expert_labels, expert_rects):
            label.move_to(rect.get_center())

        all_experts = VGroup(*[VGroup(r, l) for r, l in zip(expert_rects, expert_labels)])

        self.play(
            *[FadeIn(e) for e in all_experts],
            run_time=0.6,
        )

        # ── 3. Router box ──
        router_box = Rectangle(
            width=2.0, height=0.7, color=YELLOW_C,
            fill_color="#3a3a1e", fill_opacity=0.8, stroke_width=2,
        )
        router_box.move_to(UP * 0.3)
        router_label = Text("Router", font_size=18, color=YELLOW_C)
        router_label.move_to(router_box.get_center())
        router_group = VGroup(router_box, router_label)

        self.play(FadeIn(router_group), run_time=0.5)
        self.wait(0.3)

        # ── 4. Token 1 arrives at router ──
        token1 = Rectangle(
            width=1.2, height=0.5, color=BLUE_C,
            fill_color=BLUE_D, fill_opacity=0.8, stroke_width=2,
        )
        token1_label = Text("Token 1", font_size=14, color=WHITE)
        token1.move_to(UP * 2.2 + LEFT * 4)
        token1_label.move_to(token1.get_center())
        token1_group = VGroup(token1, token1_label)

        self.play(FadeIn(token1_group), run_time=0.3)
        self.play(
            token1_group.animate.move_to(UP * 0.3 + LEFT * 2.5),
            run_time=0.6,
        )

        # Flash router to indicate computation
        self.play(
            router_box.animate.set_fill(YELLOW_E, opacity=0.6),
            run_time=0.2,
        )
        self.play(
            router_box.animate.set_fill("#3a3a1e", opacity=0.8),
            run_time=0.2,
        )

        # ── 5. Show score bars for each expert ──
        scores_t1 = [0.05, 0.08, 0.42, 0.03, 0.06, 0.04, 0.28, 0.04]
        max_bar_h = 0.9
        score_bars = []
        score_labels = []
        for i, s in enumerate(scores_t1):
            bar = Rectangle(
                width=0.35, height=max_bar_h * s / 0.42,
                color=GREY_B, fill_color=GREY_B, fill_opacity=0.5,
                stroke_width=1,
            )
            bar.next_to(expert_rects[i], UP, buff=0.15)
            bar.align_to(expert_rects[i], DOWN)
            bar.shift(UP * 0.75)
            sl = Text(f"{s:.2f}", font_size=10, color=GREY_B)
            sl.next_to(bar, UP, buff=0.05)
            score_bars.append(bar)
            score_labels.append(sl)

        self.play(
            *[GrowFromEdge(b, DOWN) for b in score_bars],
            *[FadeIn(sl) for sl in score_labels],
            run_time=0.8,
        )
        self.wait(0.3)

        # ── 6. Top-2 experts light up (E3=idx2, E7=idx6) ──
        active_indices_t1 = [2, 6]  # E3 and E7
        highlight_anims = []
        for idx in active_indices_t1:
            highlight_anims.append(
                expert_rects[idx].animate.set_fill(GREEN_C, opacity=0.9)
            )
            highlight_anims.append(
                score_bars[idx].animate.set_fill(GREEN_C, opacity=0.8).set_color(GREEN_C)
            )
            highlight_anims.append(
                score_labels[idx].animate.set_color(GREEN_A)
            )

        top2_label = Text("Top-2 selected", font_size=16, color=GREEN_A)
        top2_label.move_to(RIGHT * 4.5 + DOWN * 0.5)

        self.play(*highlight_anims, FadeIn(top2_label), run_time=0.6)
        self.wait(0.3)

        # ── 7. Animated arrows from router to active experts ──
        arrows_t1 = []
        for idx in active_indices_t1:
            arrow = Arrow(
                router_box.get_bottom(),
                expert_rects[idx].get_top(),
                buff=0.15, color=GREEN_C, stroke_width=3,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_t1.append(arrow)

        self.play(*[GrowArrow(a) for a in arrows_t1], run_time=0.6)

        # Move token copies to the active experts
        token1_copy_a = token1_group.copy().scale(0.5)
        token1_copy_b = token1_group.copy().scale(0.5)
        self.play(
            token1_copy_a.animate.move_to(expert_rects[2].get_center()),
            token1_copy_b.animate.move_to(expert_rects[6].get_center()),
            run_time=0.5,
        )
        self.wait(0.2)

        # ── 8. Results combined (weighted sum) ──
        combine_box = Rectangle(
            width=1.6, height=0.6, color=TEAL_C,
            fill_color=TEAL_E, fill_opacity=0.6, stroke_width=2,
        )
        combine_box.move_to(DOWN * 3.0)
        combine_label = Text("Weighted Sum", font_size=14, color=TEAL_B)
        combine_label.move_to(combine_box.get_center())
        combine_group = VGroup(combine_box, combine_label)

        arrow_e3_out = Arrow(
            expert_rects[2].get_bottom(), combine_box.get_top() + LEFT * 0.3,
            buff=0.15, color=GREEN_C, stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        arrow_e7_out = Arrow(
            expert_rects[6].get_bottom(), combine_box.get_top() + RIGHT * 0.3,
            buff=0.15, color=GREEN_C, stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )

        weight_a = Text("0.60", font_size=12, color=GREEN_A)
        weight_a.next_to(arrow_e3_out, LEFT, buff=0.05)
        weight_b = Text("0.40", font_size=12, color=GREEN_A)
        weight_b.next_to(arrow_e7_out, RIGHT, buff=0.05)

        self.play(
            GrowArrow(arrow_e3_out), GrowArrow(arrow_e7_out),
            FadeIn(combine_group),
            FadeIn(weight_a), FadeIn(weight_b),
            run_time=0.6,
        )
        self.wait(0.4)

        # ── 9. Efficiency text ──
        efficiency_text = Text(
            "Only 2 of 8 experts activate per token - 4x less compute",
            font_size=20, color=GREEN_A,
        )
        efficiency_text.to_edge(DOWN, buff=0.3)
        self.play(Write(efficiency_text), run_time=0.8)
        self.wait(1.0)

        # ── 10. Transition: clear for Token 2 ──
        fade_out_t1 = [
            *score_bars, *score_labels, *arrows_t1,
            token1_group, token1_copy_a, token1_copy_b,
            arrow_e3_out, arrow_e7_out, weight_a, weight_b,
            combine_group, top2_label, efficiency_text,
        ]
        # Reset expert colors
        reset_anims = []
        for idx in active_indices_t1:
            reset_anims.append(
                expert_rects[idx].animate.set_fill(expert_colors_inactive, opacity=0.8)
            )

        self.play(
            *[FadeOut(obj) for obj in fade_out_t1],
            *reset_anims,
            run_time=0.5,
        )

        # ── Token 2 arrives ──
        token2 = Rectangle(
            width=1.2, height=0.5, color=PURPLE_B,
            fill_color=PURPLE_C, fill_opacity=0.8, stroke_width=2,
        )
        token2_label = Text("Token 2", font_size=14, color=WHITE)
        token2.move_to(UP * 2.2 + LEFT * 4)
        token2_label.move_to(token2.get_center())
        token2_group = VGroup(token2, token2_label)

        self.play(FadeIn(token2_group), run_time=0.3)
        self.play(
            token2_group.animate.move_to(UP * 0.3 + LEFT * 2.5),
            run_time=0.6,
        )

        # Router flash
        self.play(
            router_box.animate.set_fill(YELLOW_E, opacity=0.6),
            run_time=0.2,
        )
        self.play(
            router_box.animate.set_fill("#3a3a1e", opacity=0.8),
            run_time=0.2,
        )

        # Token 2 scores - different distribution, E1 and E5 are top
        scores_t2 = [0.38, 0.06, 0.04, 0.07, 0.35, 0.03, 0.04, 0.03]
        score_bars2 = []
        score_labels2 = []
        for i, s in enumerate(scores_t2):
            bar = Rectangle(
                width=0.35, height=max_bar_h * s / 0.38,
                color=GREY_B, fill_color=GREY_B, fill_opacity=0.5,
                stroke_width=1,
            )
            bar.next_to(expert_rects[i], UP, buff=0.15)
            bar.align_to(expert_rects[i], DOWN)
            bar.shift(UP * 0.75)
            sl = Text(f"{s:.2f}", font_size=10, color=GREY_B)
            sl.next_to(bar, UP, buff=0.05)
            score_bars2.append(bar)
            score_labels2.append(sl)

        self.play(
            *[GrowFromEdge(b, DOWN) for b in score_bars2],
            *[FadeIn(sl) for sl in score_labels2],
            run_time=0.8,
        )
        self.wait(0.3)

        # Top-2: E1 (idx0) and E5 (idx4)
        active_indices_t2 = [0, 4]
        highlight_anims2 = []
        for idx in active_indices_t2:
            highlight_anims2.append(
                expert_rects[idx].animate.set_fill(PURPLE_B, opacity=0.9)
            )
            highlight_anims2.append(
                score_bars2[idx].animate.set_fill(PURPLE_B, opacity=0.8).set_color(PURPLE_B)
            )
            highlight_anims2.append(
                score_labels2[idx].animate.set_color(PURPLE_A)
            )

        diff_label = Text("Different experts activate!", font_size=16, color=PURPLE_A)
        diff_label.move_to(RIGHT * 4.5 + DOWN * 0.5)

        self.play(*highlight_anims2, FadeIn(diff_label), run_time=0.6)

        arrows_t2 = []
        for idx in active_indices_t2:
            arrow = Arrow(
                router_box.get_bottom(),
                expert_rects[idx].get_top(),
                buff=0.15, color=PURPLE_B, stroke_width=3,
                max_tip_length_to_length_ratio=0.15,
            )
            arrows_t2.append(arrow)

        self.play(*[GrowArrow(a) for a in arrows_t2], run_time=0.6)

        token2_copy_a = token2_group.copy().scale(0.5)
        token2_copy_b = token2_group.copy().scale(0.5)
        self.play(
            token2_copy_a.animate.move_to(expert_rects[0].get_center()),
            token2_copy_b.animate.move_to(expert_rects[4].get_center()),
            run_time=0.5,
        )

        # Combine for token 2
        combine_box2 = Rectangle(
            width=1.6, height=0.6, color=TEAL_C,
            fill_color=TEAL_E, fill_opacity=0.6, stroke_width=2,
        )
        combine_box2.move_to(DOWN * 3.0)
        combine_label2 = Text("Weighted Sum", font_size=14, color=TEAL_B)
        combine_label2.move_to(combine_box2.get_center())
        combine_group2 = VGroup(combine_box2, combine_label2)

        arrow_e1_out = Arrow(
            expert_rects[0].get_bottom(), combine_box2.get_top() + LEFT * 0.3,
            buff=0.15, color=PURPLE_B, stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        arrow_e5_out = Arrow(
            expert_rects[4].get_bottom(), combine_box2.get_top() + RIGHT * 0.3,
            buff=0.15, color=PURPLE_B, stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )

        weight_c = Text("0.52", font_size=12, color=PURPLE_A)
        weight_c.next_to(arrow_e1_out, LEFT, buff=0.05)
        weight_d = Text("0.48", font_size=12, color=PURPLE_A)
        weight_d.next_to(arrow_e5_out, RIGHT, buff=0.05)

        self.play(
            GrowArrow(arrow_e1_out), GrowArrow(arrow_e5_out),
            FadeIn(combine_group2),
            FadeIn(weight_c), FadeIn(weight_d),
            run_time=0.6,
        )
        self.wait(0.4)

        # Final summary
        summary = Text(
            "Each token activates different experts - only a fraction of parameters used",
            font_size=18, color=TEAL_A,
        )
        summary.to_edge(DOWN, buff=0.3)
        self.play(Write(summary), run_time=0.8)
        self.wait(2.0)
