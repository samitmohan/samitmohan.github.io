from manimlib import *
import numpy as np


class SpeculativeDecoding(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        DRAFT_COLOR = "#4ecdc4"
        VERIFIER_COLOR = "#ff6b6b"
        ACCEPT_COLOR = "#2ecc71"
        REJECT_COLOR = "#e74c3c"
        TOKEN_BG = "#2a2a4a"
        HIGHLIGHT = "#ffd93d"

        # --- Step 1: Title ---
        title = Text("Speculative Decoding", font_size=40, color=WHITE, weight=BOLD)
        subtitle = Text(
            "Draft fast, verify in parallel",
            font_size=22, color=HIGHLIGHT
        )
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.5)
        self.wait(0.5)
        self.play(
            FadeOut(title, shift=UP * 0.5),
            FadeOut(subtitle, shift=UP * 0.5),
            run_time=0.5
        )

        # --- Step 2: Prompt ---
        prompt_label = Text("Prompt:", font_size=20, color=GREY_B)
        prompt_text = Text('"The capital of France is"', font_size=24, color=WHITE)
        prompt_group = VGroup(prompt_label, prompt_text).arrange(RIGHT, buff=0.3)
        prompt_group.to_edge(UP, buff=0.5)

        self.play(FadeIn(prompt_group), run_time=0.5)
        self.wait(0.3)

        # --- Step 3: Draft model generating tokens ---
        draft_label = Text("Draft Model (small, fast)", font_size=22, color=DRAFT_COLOR, weight=BOLD)
        draft_label.next_to(prompt_group, DOWN, buff=0.6).to_edge(LEFT, buff=0.8)

        # Draft model box
        draft_box = RoundedRectangle(
            width=2.2, height=1.0, corner_radius=0.15,
            stroke_color=DRAFT_COLOR, stroke_width=2,
            fill_color=DRAFT_COLOR, fill_opacity=0.15
        )
        draft_icon = Text("7B", font_size=28, color=DRAFT_COLOR, weight=BOLD)
        draft_model = VGroup(draft_box, draft_icon)
        draft_model.next_to(draft_label, DOWN, buff=0.3)

        self.play(FadeIn(draft_label), FadeIn(draft_model), run_time=0.5)

        # Token predictions - appearing rapidly one by one
        tokens = ["Paris", ",", "known", "as"]
        token_boxes = VGroup()
        for i, tok in enumerate(tokens):
            box = RoundedRectangle(
                width=1.3, height=0.65, corner_radius=0.1,
                stroke_color=DRAFT_COLOR, stroke_width=1.5,
                fill_color=TOKEN_BG, fill_opacity=0.8
            )
            text = Text(tok, font_size=22, color=WHITE)
            text.move_to(box.get_center())
            token_boxes.add(VGroup(box, text))

        token_boxes.arrange(RIGHT, buff=0.2)
        token_boxes.next_to(draft_model, RIGHT, buff=0.6)

        # Arrow from draft model to tokens
        draft_arrow = Arrow(
            draft_model.get_right(), token_boxes.get_left(),
            buff=0.15, color=DRAFT_COLOR, stroke_width=2
        )

        self.play(GrowArrow(draft_arrow), run_time=0.3)

        # Tokens appear rapidly with a flash effect
        speed_label = Text("~4x faster", font_size=16, color=DRAFT_COLOR)
        speed_label.next_to(draft_arrow, UP, buff=0.1)

        for i, tb in enumerate(token_boxes):
            self.play(
                FadeIn(tb),
                run_time=0.15
            )
            # Quick flash
            self.play(
                tb[0].animate.set_stroke(color=HIGHLIGHT, width=3),
                run_time=0.08
            )
            self.play(
                tb[0].animate.set_stroke(color=DRAFT_COLOR, width=1.5),
                run_time=0.08
            )

        self.play(FadeIn(speed_label), run_time=0.3)
        self.wait(0.3)

        # --- Step 4: Verifier model checking all 4 in ONE pass ---
        verifier_label = Text(
            "Verifier Model (large, accurate)", font_size=22,
            color=VERIFIER_COLOR, weight=BOLD
        )
        verifier_label.next_to(draft_model, DOWN, buff=0.8).align_to(draft_label, LEFT)

        verifier_box = RoundedRectangle(
            width=2.2, height=1.0, corner_radius=0.15,
            stroke_color=VERIFIER_COLOR, stroke_width=2,
            fill_color=VERIFIER_COLOR, fill_opacity=0.15
        )
        verifier_icon = Text("70B", font_size=28, color=VERIFIER_COLOR, weight=BOLD)
        verifier_model = VGroup(verifier_box, verifier_icon)
        verifier_model.next_to(verifier_label, DOWN, buff=0.3)

        self.play(FadeIn(verifier_label), FadeIn(verifier_model), run_time=0.5)

        # Arrow from verifier to tokens area
        verify_arrow = Arrow(
            verifier_model.get_right(),
            token_boxes.get_left() + DOWN * 0.8,
            buff=0.15, color=VERIFIER_COLOR, stroke_width=2
        )

        one_pass_label = Text("1 forward pass", font_size=16, color=VERIFIER_COLOR)
        one_pass_label.next_to(verify_arrow, DOWN, buff=0.1)

        self.play(GrowArrow(verify_arrow), FadeIn(one_pass_label), run_time=0.4)

        # Verification bar scanning across all tokens
        scan_bar = Rectangle(
            width=0.08, height=0.85,
            stroke_color=VERIFIER_COLOR, stroke_width=0,
            fill_color=VERIFIER_COLOR, fill_opacity=0.8
        )
        scan_bar.move_to(token_boxes[0].get_left() + LEFT * 0.1)

        self.play(FadeIn(scan_bar), run_time=0.1)
        self.play(
            scan_bar.animate.move_to(token_boxes[-1].get_right() + RIGHT * 0.1),
            run_time=1.0,
            rate_func=linear
        )
        self.play(FadeOut(scan_bar), run_time=0.1)
        self.wait(0.2)

        # --- Step 5: Accept/Reject results ---
        # Tokens: "Paris" (accept), "," (accept), "known" (reject), "as" (accept)
        accept_reject = [True, True, False, True]
        check_marks = VGroup()

        for i, (tb, accepted) in enumerate(zip(token_boxes, accept_reject)):
            color = ACCEPT_COLOR if accepted else REJECT_COLOR
            symbol = "OK" if accepted else "X"
            mark = Text(symbol, font_size=16, color=color, weight=BOLD)
            mark.next_to(tb, DOWN, buff=0.15)
            check_marks.add(mark)

            # Flash the token box
            self.play(
                tb[0].animate.set_stroke(color=color, width=3),
                tb[0].animate.set_fill(color=color, opacity=0.25),
                FadeIn(mark),
                run_time=0.3
            )

        self.wait(0.5)

        # --- Step 6: Replace rejected token ---
        rejected_idx = 2  # "known"
        old_token = token_boxes[rejected_idx]

        # Create replacement token
        new_box = RoundedRectangle(
            width=1.3, height=0.65, corner_radius=0.1,
            stroke_color=VERIFIER_COLOR, stroke_width=2.5,
            fill_color=VERIFIER_COLOR, fill_opacity=0.2
        )
        new_text = Text("often", font_size=22, color=WHITE, weight=BOLD)
        new_box.move_to(old_token.get_center())
        new_text.move_to(new_box.get_center())
        new_token = VGroup(new_box, new_text)

        correction_label = Text(
            'Corrected: "known" -> "often"',
            font_size=18, color=HIGHLIGHT
        )
        correction_label.next_to(check_marks, DOWN, buff=0.4)

        # Animate the replacement
        self.play(
            old_token.animate.set_opacity(0.3),
            run_time=0.3
        )
        self.play(
            FadeIn(new_token),
            FadeOut(old_token),
            check_marks[rejected_idx].animate.set_opacity(0),
            Write(correction_label),
            run_time=0.6
        )

        # Update the accepted mark for the corrected token
        new_mark = Text("OK", font_size=16, color=ACCEPT_COLOR, weight=BOLD)
        new_mark.next_to(new_token, DOWN, buff=0.15)
        self.play(FadeIn(new_mark), run_time=0.3)

        self.wait(0.5)

        # --- Step 7: Final summary text ---
        # Fade out intermediate elements
        self.play(
            FadeOut(correction_label),
            FadeOut(speed_label),
            FadeOut(one_pass_label),
            run_time=0.4
        )

        # Final result line
        final_text = Text(
            "4 tokens verified in 1 pass instead of 4 separate passes",
            font_size=24, color=HIGHLIGHT, weight=BOLD
        )
        final_text.to_edge(DOWN, buff=0.5)

        # Highlight box around the final text
        final_box = SurroundingRectangle(
            final_text, buff=0.2,
            stroke_color=HIGHLIGHT, stroke_width=2,
            fill_color=HIGHLIGHT, fill_opacity=0.08,
        )

        self.play(
            Write(final_text),
            ShowCreation(final_box),
            run_time=1.0
        )
        self.wait(1.5)
