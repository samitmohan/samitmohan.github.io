from manimlib import *
import numpy as np


class KernelFusion(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        HBM_COLOR = "#4a4a6a"
        SRAM_COLOR = "#2d6a4f"
        OP_COLORS = ["#ff6b6b", "#4ecdc4", "#ffd93d"]
        ARROW_UP_COLOR = "#e07cff"
        ARROW_DOWN_COLOR = "#45b7d1"
        HIGHLIGHT = "#ffd93d"

        # --- Title ---
        title = Text(
            "Kernel Fusion: Reducing HBM Traffic",
            font_size=30, color=WHITE, weight=BOLD
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # --- Divider ---
        divider = DashedLine(
            start=UP * 2.5, end=DOWN * 3.5,
            color=GREY, stroke_width=1, dash_length=0.1
        )
        self.play(ShowCreation(divider), run_time=0.2)

        # --- Column headers ---
        unfused_header = Text("Unfused Kernels", font_size=22, color="#ff6b6b", weight=BOLD)
        fused_header = Text("Fused Kernel", font_size=22, color="#2ecc71", weight=BOLD)
        unfused_header.move_to(LEFT * 3.5 + UP * 2.2)
        fused_header.move_to(RIGHT * 3.5 + UP * 2.2)
        self.play(FadeIn(unfused_header), FadeIn(fused_header), run_time=0.4)

        # =====================================================================
        # LEFT SIDE: UNFUSED - 3 separate kernels with HBM round trips
        # =====================================================================
        left_center = LEFT * 3.5

        # HBM cloud at top
        hbm_left = RoundedRectangle(
            width=4.5, height=1.0, corner_radius=0.2,
            stroke_color=HBM_COLOR, stroke_width=2,
            fill_color=HBM_COLOR, fill_opacity=0.2,
        )
        hbm_left.move_to(left_center + UP * 1.2)
        hbm_left_label = Text("HBM (Global Memory)", font_size=14, color="#8888aa")
        hbm_left_label.move_to(hbm_left.get_center())

        self.play(FadeIn(hbm_left), FadeIn(hbm_left_label), run_time=0.4)

        # Three operation boxes
        op_names = ["LayerNorm", "Linear", "GeLU"]
        op_boxes_left = VGroup()
        op_labels_left = VGroup()

        for i, (name, color) in enumerate(zip(op_names, OP_COLORS)):
            box = RoundedRectangle(
                width=1.3, height=0.7, corner_radius=0.1,
                stroke_color=color, stroke_width=2,
                fill_color=color, fill_opacity=0.15,
            )
            label = Text(name, font_size=14, color=color, weight=BOLD)
            label.move_to(box.get_center())
            op_boxes_left.add(box)
            op_labels_left.add(label)

        op_boxes_left.arrange(DOWN, buff=0.5)
        op_boxes_left.move_to(left_center + DOWN * 1.2)

        for label, box in zip(op_labels_left, op_boxes_left):
            label.move_to(box.get_center())

        self.play(
            *[FadeIn(b) for b in op_boxes_left],
            *[FadeIn(l) for l in op_labels_left],
            run_time=0.5,
        )
        self.wait(0.3)

        # Animate the 6 HBM round trips - data going up and down
        trip_counter_label = Text("HBM trips: ", font_size=16, color=GREY_B)
        trip_counter_label.move_to(left_center + DOWN * 3.2 + LEFT * 0.6)

        trip_count = 0
        trip_count_text = Text("0", font_size=18, color=HIGHLIGHT, weight=BOLD)
        trip_count_text.next_to(trip_counter_label, RIGHT, buff=0.1)

        self.play(FadeIn(trip_counter_label), FadeIn(trip_count_text), run_time=0.3)

        # For each op: read from HBM (arrow down), write to HBM (arrow up)
        for i, (box, color) in enumerate(zip(op_boxes_left, OP_COLORS)):
            # Arrow: HBM -> kernel (read)
            read_start = hbm_left.get_bottom()
            read_end = box.get_top()

            # Data dot traveling down
            data_dot_down = Dot(read_start, color=ARROW_DOWN_COLOR, radius=0.06)
            read_label = Text("read", font_size=10, color=ARROW_DOWN_COLOR)
            read_label.next_to(data_dot_down, RIGHT, buff=0.08)

            self.play(FadeIn(data_dot_down), FadeIn(read_label), run_time=0.1)
            self.play(
                data_dot_down.animate.move_to(read_end),
                read_label.animate.next_to(read_end, RIGHT, buff=0.08),
                run_time=0.35,
            )
            trip_count += 1
            new_count = Text(str(trip_count), font_size=18, color=HIGHLIGHT, weight=BOLD)
            new_count.next_to(trip_counter_label, RIGHT, buff=0.1)
            self.play(
                FadeOut(data_dot_down), FadeOut(read_label),
                box.animate.set_fill(color, opacity=0.4),
                FadeTransform(trip_count_text, new_count),
                run_time=0.2,
            )
            trip_count_text = new_count

            # Arrow: kernel -> HBM (write)
            write_start = box.get_top()
            data_dot_up = Dot(write_start, color=ARROW_UP_COLOR, radius=0.06)
            write_label = Text("write", font_size=10, color=ARROW_UP_COLOR)
            write_label.next_to(data_dot_up, LEFT, buff=0.08)

            self.play(FadeIn(data_dot_up), FadeIn(write_label), run_time=0.1)
            self.play(
                data_dot_up.animate.move_to(hbm_left.get_bottom()),
                write_label.animate.next_to(hbm_left.get_bottom(), LEFT, buff=0.08),
                run_time=0.35,
            )
            trip_count += 1
            new_count = Text(str(trip_count), font_size=18, color=HIGHLIGHT, weight=BOLD)
            new_count.next_to(trip_counter_label, RIGHT, buff=0.1)
            self.play(
                FadeOut(data_dot_up), FadeOut(write_label),
                box.animate.set_fill(color, opacity=0.15),
                FadeTransform(trip_count_text, new_count),
                run_time=0.2,
            )
            trip_count_text = new_count

        # Final count highlight
        final_left_count = Text("6 HBM round trips", font_size=16, color="#ff6b6b", weight=BOLD)
        final_left_count.move_to(trip_counter_label.get_center() + RIGHT * 0.5)
        self.play(
            FadeOut(trip_counter_label), FadeOut(trip_count_text),
            FadeIn(final_left_count),
            run_time=0.4,
        )
        self.wait(0.3)

        # =====================================================================
        # RIGHT SIDE: FUSED - single kernel with SRAM
        # =====================================================================
        right_center = RIGHT * 3.5

        # HBM cloud at top
        hbm_right = RoundedRectangle(
            width=4.5, height=1.0, corner_radius=0.2,
            stroke_color=HBM_COLOR, stroke_width=2,
            fill_color=HBM_COLOR, fill_opacity=0.2,
        )
        hbm_right.move_to(right_center + UP * 1.2)
        hbm_right_label = Text("HBM (Global Memory)", font_size=14, color="#8888aa")
        hbm_right_label.move_to(hbm_right.get_center())

        self.play(FadeIn(hbm_right), FadeIn(hbm_right_label), run_time=0.3)

        # Single fused kernel box containing all three ops
        fused_box = RoundedRectangle(
            width=3.8, height=2.5, corner_radius=0.15,
            stroke_color="#2ecc71", stroke_width=2.5,
            fill_color="#2ecc71", fill_opacity=0.08,
        )
        fused_box.move_to(right_center + DOWN * 1.0)

        fused_title = Text("Fused Kernel", font_size=14, color="#2ecc71", weight=BOLD)
        fused_title.next_to(fused_box, UP, buff=0.08)

        # SRAM label inside the fused box
        sram_label = Text("SRAM (On-Chip)", font_size=12, color=SRAM_COLOR)
        sram_label.move_to(fused_box.get_corner(UR) + DL * 0.35)

        # Mini op boxes inside fused kernel
        mini_ops = VGroup()
        for name, color in zip(op_names, OP_COLORS):
            mini_box = RoundedRectangle(
                width=2.8, height=0.45, corner_radius=0.08,
                stroke_color=color, stroke_width=1.5,
                fill_color=color, fill_opacity=0.15,
            )
            mini_label = Text(name, font_size=12, color=color)
            mini_label.move_to(mini_box.get_center())
            mini_ops.add(VGroup(mini_box, mini_label))

        mini_ops.arrange(DOWN, buff=0.15)
        mini_ops.move_to(fused_box.get_center())

        # Arrows between mini ops (data stays in SRAM)
        sram_arrows = VGroup()
        for i in range(len(mini_ops) - 1):
            arrow = Arrow(
                mini_ops[i].get_bottom(), mini_ops[i + 1].get_top(),
                buff=0.05, color=SRAM_COLOR, stroke_width=2,
                max_tip_length_to_length_ratio=0.3,
            )
            sram_arrows.add(arrow)

        self.play(
            FadeIn(fused_box), FadeIn(fused_title), FadeIn(sram_label),
            *[FadeIn(op) for op in mini_ops],
            *[GrowArrow(a) for a in sram_arrows],
            run_time=0.6,
        )
        self.wait(0.3)

        # Animate: only 2 HBM trips (1 read in, 1 write out)
        trip_counter_right = Text("HBM trips: ", font_size=16, color=GREY_B)
        trip_counter_right.move_to(right_center + DOWN * 3.2 + LEFT * 0.6)
        trip_count_r = 0
        trip_count_r_text = Text("0", font_size=18, color=HIGHLIGHT, weight=BOLD)
        trip_count_r_text.next_to(trip_counter_right, RIGHT, buff=0.1)

        self.play(FadeIn(trip_counter_right), FadeIn(trip_count_r_text), run_time=0.3)

        # Read from HBM into fused kernel
        data_dot_in = Dot(hbm_right.get_bottom(), color=ARROW_DOWN_COLOR, radius=0.06)
        read_lbl = Text("read", font_size=10, color=ARROW_DOWN_COLOR)
        read_lbl.next_to(data_dot_in, RIGHT, buff=0.08)

        self.play(FadeIn(data_dot_in), FadeIn(read_lbl), run_time=0.1)
        self.play(
            data_dot_in.animate.move_to(fused_box.get_top()),
            read_lbl.animate.next_to(fused_box.get_top(), RIGHT, buff=0.08),
            run_time=0.4,
        )
        trip_count_r += 1
        new_r_count = Text(str(trip_count_r), font_size=18, color=HIGHLIGHT, weight=BOLD)
        new_r_count.next_to(trip_counter_right, RIGHT, buff=0.1)
        self.play(
            FadeOut(data_dot_in), FadeOut(read_lbl),
            FadeTransform(trip_count_r_text, new_r_count),
            run_time=0.2,
        )
        trip_count_r_text = new_r_count

        # Data processes through all three ops in SRAM (flash through mini ops)
        for op in mini_ops:
            self.play(
                op[0].animate.set_fill(opacity=0.4),
                run_time=0.2,
            )
            self.play(
                op[0].animate.set_fill(opacity=0.15),
                run_time=0.15,
            )

        # Write result back to HBM
        data_dot_out = Dot(fused_box.get_top(), color=ARROW_UP_COLOR, radius=0.06)
        write_lbl = Text("write", font_size=10, color=ARROW_UP_COLOR)
        write_lbl.next_to(data_dot_out, LEFT, buff=0.08)

        self.play(FadeIn(data_dot_out), FadeIn(write_lbl), run_time=0.1)
        self.play(
            data_dot_out.animate.move_to(hbm_right.get_bottom()),
            write_lbl.animate.next_to(hbm_right.get_bottom(), LEFT, buff=0.08),
            run_time=0.4,
        )
        trip_count_r += 1
        new_r_count = Text(str(trip_count_r), font_size=18, color=HIGHLIGHT, weight=BOLD)
        new_r_count.next_to(trip_counter_right, RIGHT, buff=0.1)
        self.play(
            FadeOut(data_dot_out), FadeOut(write_lbl),
            FadeTransform(trip_count_r_text, new_r_count),
            run_time=0.2,
        )
        trip_count_r_text = new_r_count

        # Final count
        final_right_count = Text("2 HBM round trips", font_size=16, color="#2ecc71", weight=BOLD)
        final_right_count.move_to(trip_counter_right.get_center() + RIGHT * 0.5)
        self.play(
            FadeOut(trip_counter_right), FadeOut(trip_count_r_text),
            FadeIn(final_right_count),
            run_time=0.4,
        )
        self.wait(0.5)

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        all_objects = VGroup(
            hbm_left, hbm_left_label, *op_boxes_left, *op_labels_left,
            final_left_count,
            hbm_right, hbm_right_label, fused_box, fused_title, sram_label,
            *mini_ops, sram_arrows, final_right_count,
            divider, unfused_header, fused_header, title,
        )
        self.play(FadeOut(all_objects), run_time=0.5)

        final_text = Text(
            "3x less memory traffic. Same computation.",
            font_size=28, color=HIGHLIGHT, weight=BOLD
        )
        final_text.move_to(ORIGIN)

        final_box = SurroundingRectangle(
            final_text, buff=0.2,
            stroke_color=HIGHLIGHT, stroke_width=2,
            fill_color=HIGHLIGHT, fill_opacity=0.08,
        )

        self.play(Write(final_text), ShowCreation(final_box), run_time=0.8)
        self.wait(1.5)
