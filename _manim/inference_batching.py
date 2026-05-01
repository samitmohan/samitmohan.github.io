from manimlib import *
import numpy as np


class BatchingComparison(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        REQ_COLORS = ["#ff6b6b", "#4ecdc4", "#ffd93d", "#e07cff"]
        NEW_COLORS = ["#45b7d1", "#f39c12", "#2ecc71"]
        PAD_COLOR = "#333355"
        HIGHLIGHT = "#ffd93d"
        WASTE_COLOR = "#555555"

        # --- Title ---
        title = Text(
            "Batching Strategies for LLM Inference",
            font_size=30, color=WHITE, weight=BOLD
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # Request lengths (in time units)
        req_lengths = [3, 5, 2, 4]
        max_len = max(req_lengths)

        cell_w = 0.45
        cell_h = 0.35
        bar_gap = 0.12

        # =====================================================================
        # PANEL 1: STATIC BATCHING
        # =====================================================================
        panel1_title = Text("Static Batching", font_size=20, color="#ff6b6b", weight=BOLD)
        panel1_title.move_to(LEFT * 4.0 + UP * 1.5)

        panel1_group = VGroup()
        pad_cells = VGroup()

        for i, length in enumerate(req_lengths):
            row = VGroup()
            for j in range(max_len):
                if j < length:
                    cell = Rectangle(
                        width=cell_w, height=cell_h,
                        stroke_color=REQ_COLORS[i], stroke_width=1.5,
                        fill_color=REQ_COLORS[i], fill_opacity=0.5,
                    )
                else:
                    cell = Rectangle(
                        width=cell_w, height=cell_h,
                        stroke_color=PAD_COLOR, stroke_width=1,
                        fill_color=PAD_COLOR, fill_opacity=0.3,
                    )
                    pad_cells.add(cell)
                row.add(cell)
            row.arrange(RIGHT, buff=0.04)
            panel1_group.add(row)

        panel1_group.arrange(DOWN, buff=bar_gap)
        panel1_group.next_to(panel1_title, DOWN, buff=0.2)
        panel1_group.align_to(panel1_title, LEFT)

        # Request labels
        req_labels_1 = VGroup()
        for i in range(4):
            lbl = Text(f"R{i+1}", font_size=11, color=REQ_COLORS[i])
            lbl.next_to(panel1_group[i], LEFT, buff=0.1)
            req_labels_1.add(lbl)

        self.play(FadeIn(panel1_title), run_time=0.3)
        self.play(
            *[FadeIn(row) for row in panel1_group],
            *[FadeIn(lbl) for lbl in req_labels_1],
            run_time=0.5,
        )

        # Highlight padding as waste
        waste_label_1 = Text("GPU idle (padding)", font_size=12, color=WASTE_COLOR)
        waste_label_1.next_to(panel1_group, DOWN, buff=0.15)

        # Flash padding cells
        pad_anims = [cell.animate.set_fill(WASTE_COLOR, opacity=0.5) for cell in pad_cells]
        self.play(*pad_anims, FadeIn(waste_label_1), run_time=0.4)

        # Start/end bracket
        start_line = Line(
            panel1_group.get_corner(UL) + UP * 0.1,
            panel1_group.get_corner(DL) + DOWN * 0.1,
            color=GREY, stroke_width=1.5,
        )
        end_line = Line(
            panel1_group.get_corner(UR) + UP * 0.1,
            panel1_group.get_corner(DR) + DOWN * 0.1,
            color=GREY, stroke_width=1.5,
        )
        start_lbl = Text("start", font_size=10, color=GREY)
        start_lbl.next_to(start_line, UP, buff=0.05)
        end_lbl = Text("end", font_size=10, color=GREY)
        end_lbl.next_to(end_line, UP, buff=0.05)

        self.play(
            ShowCreation(start_line), ShowCreation(end_line),
            FadeIn(start_lbl), FadeIn(end_lbl),
            run_time=0.3,
        )
        self.wait(0.5)

        # Fade panel 1 to dim
        panel1_all = VGroup(
            panel1_title, panel1_group, req_labels_1, waste_label_1,
            start_line, end_line, start_lbl, end_lbl,
        )
        self.play(panel1_all.animate.set_opacity(0.3), run_time=0.3)

        # =====================================================================
        # PANEL 2: CONTINUOUS BATCHING
        # =====================================================================
        panel2_title = Text("Continuous Batching", font_size=20, color="#2ecc71", weight=BOLD)
        panel2_title.move_to(LEFT * 4.0 + UP * 1.5)

        # Build a timeline: 6 time steps total
        # R1: 3 steps, R2: 5 steps, R3: 2 steps, R4: 4 steps
        # When R3 finishes at t=2, new request R5 slots in
        # When R1 finishes at t=3, new request R6 slots in
        total_steps = 6
        n_slots = 4

        # Timeline grid: rows = slots, cols = time steps
        # Slot assignments at each time step
        # slot 0: R1(0-2), R6(3-5)
        # slot 1: R2(0-4), idle(5)
        # slot 2: R3(0-1), R5(2-5)
        # slot 3: R4(0-3), idle(4-5)
        slot_schedule = [
            [(REQ_COLORS[0], "R1")] * 3 + [(NEW_COLORS[2], "R6")] * 3,
            [(REQ_COLORS[1], "R2")] * 5 + [(None, "")] * 1,
            [(REQ_COLORS[2], "R3")] * 2 + [(NEW_COLORS[0], "R5")] * 4,
            [(REQ_COLORS[3], "R4")] * 4 + [(NEW_COLORS[1], "R7")] * 2,
        ]

        panel2_group = VGroup()
        for slot_idx in range(n_slots):
            row = VGroup()
            for t in range(total_steps):
                color, label = slot_schedule[slot_idx][t]
                if color is not None:
                    cell = Rectangle(
                        width=cell_w, height=cell_h,
                        stroke_color=color, stroke_width=1.5,
                        fill_color=color, fill_opacity=0.5,
                    )
                else:
                    cell = Rectangle(
                        width=cell_w, height=cell_h,
                        stroke_color=PAD_COLOR, stroke_width=1,
                        fill_color=PAD_COLOR, fill_opacity=0.15,
                    )
                row.add(cell)
            row.arrange(RIGHT, buff=0.04)
            panel2_group.add(row)

        panel2_group.arrange(DOWN, buff=bar_gap)
        panel2_group.next_to(panel2_title, DOWN, buff=0.2)
        panel2_group.align_to(panel2_title, LEFT)

        # Slot labels
        slot_labels = VGroup()
        for i in range(n_slots):
            lbl = Text(f"Slot {i+1}", font_size=11, color=GREY_B)
            lbl.next_to(panel2_group[i], LEFT, buff=0.1)
            slot_labels.add(lbl)

        # Time step markers
        time_markers = VGroup()
        for t in range(total_steps):
            marker = Text(f"t{t}", font_size=10, color=GREY)
            marker.next_to(panel2_group[0][t], UP, buff=0.08)
            time_markers.add(marker)

        self.play(
            panel1_all.animate.set_opacity(0),
            FadeIn(panel2_title),
            run_time=0.3,
        )
        self.play(FadeOut(panel1_all), run_time=0.1)

        # Animate column by column (time step by time step)
        for t in range(total_steps):
            col_anims = []
            for slot_idx in range(n_slots):
                col_anims.append(FadeIn(panel2_group[slot_idx][t]))
            col_anims.append(FadeIn(time_markers[t]))
            if t == 0:
                col_anims.extend([FadeIn(lbl) for lbl in slot_labels])
            self.play(*col_anims, run_time=0.3)

        no_pad_label = Text("No padding waste - new requests fill gaps", font_size=12, color="#2ecc71")
        no_pad_label.next_to(panel2_group, DOWN, buff=0.15)
        self.play(FadeIn(no_pad_label), run_time=0.3)
        self.wait(0.5)

        # Fade panel 2
        panel2_all = VGroup(
            panel2_title, panel2_group, slot_labels, time_markers, no_pad_label,
        )
        self.play(panel2_all.animate.set_opacity(0.3), run_time=0.3)

        # =====================================================================
        # PANEL 3: CHUNKED PREFILL
        # =====================================================================
        panel3_title = Text("Chunked Prefill", font_size=20, color="#45b7d1", weight=BOLD)
        panel3_title.move_to(LEFT * 4.0 + UP * 1.5)

        # Show a long prefill broken into chunks with decode steps interleaved
        # Row 0: Long prefill request broken into 3 chunks
        # Rows 1-3: Other requests getting decode time between chunks

        chunk_color = "#45b7d1"
        decode_colors = ["#ff6b6b", "#4ecdc4", "#ffd93d"]
        total_cols = 9

        # Schedule:
        # Step:  chunk1  decode  chunk2  decode  chunk3  decode  chunk1  decode  decode
        # Row 0: [P]     [ ]     [P]     [ ]     [P]     [ ]     [d]     [d]     [d]
        # Row 1: [ ]     [d]     [ ]     [d]     [ ]     [d]     [ ]     [d]     [d]
        # Row 2: [ ]     [d]     [ ]     [d]     [ ]     [d]     [ ]     [d]     [ ]
        # Row 3: [ ]     [d]     [ ]     [d]     [ ]     [d]     [ ]     [ ]     [d]

        panel3_rows = 4
        schedule = []

        # Row 0: long prefill chunks interleaved with idle, then decode
        schedule.append([
            (chunk_color, "P", 0.6), (None, "", 0), (chunk_color, "P", 0.6),
            (None, "", 0), (chunk_color, "P", 0.6), (None, "", 0),
            (chunk_color, "d", 0.35), (chunk_color, "d", 0.35), (chunk_color, "d", 0.35),
        ])
        # Row 1: decode during chunk gaps
        schedule.append([
            (None, "", 0), (decode_colors[0], "d", 0.5), (None, "", 0),
            (decode_colors[0], "d", 0.5), (None, "", 0), (decode_colors[0], "d", 0.5),
            (None, "", 0), (decode_colors[0], "d", 0.5), (decode_colors[0], "d", 0.5),
        ])
        # Row 2
        schedule.append([
            (None, "", 0), (decode_colors[1], "d", 0.5), (None, "", 0),
            (decode_colors[1], "d", 0.5), (None, "", 0), (decode_colors[1], "d", 0.5),
            (None, "", 0), (decode_colors[1], "d", 0.5), (None, "", 0),
        ])
        # Row 3
        schedule.append([
            (None, "", 0), (decode_colors[2], "d", 0.5), (None, "", 0),
            (decode_colors[2], "d", 0.5), (None, "", 0), (decode_colors[2], "d", 0.5),
            (None, "", 0), (None, "", 0), (decode_colors[2], "d", 0.5),
        ])

        panel3_group = VGroup()
        for row_idx in range(panel3_rows):
            row = VGroup()
            for col_idx in range(total_cols):
                color, text, opacity = schedule[row_idx][col_idx]
                if color is not None:
                    cell = Rectangle(
                        width=cell_w * 0.85, height=cell_h,
                        stroke_color=color, stroke_width=1.5,
                        fill_color=color, fill_opacity=opacity,
                    )
                    cell_label = Text(text, font_size=9, color=WHITE)
                    cell_label.move_to(cell.get_center())
                    row.add(VGroup(cell, cell_label))
                else:
                    cell = Rectangle(
                        width=cell_w * 0.85, height=cell_h,
                        stroke_color=PAD_COLOR, stroke_width=0.5,
                        fill_color=PAD_COLOR, fill_opacity=0.05,
                    )
                    row.add(cell)
            row.arrange(RIGHT, buff=0.03)
            panel3_group.add(row)

        panel3_group.arrange(DOWN, buff=bar_gap)
        panel3_group.next_to(panel3_title, DOWN, buff=0.2)
        panel3_group.align_to(panel3_title, LEFT)

        # Row labels
        row_labels_3 = VGroup()
        row_names = ["Long\nPrefill", "Req B\nDecode", "Req C\nDecode", "Req D\nDecode"]
        row_label_colors = [chunk_color] + decode_colors
        for i, (name, color) in enumerate(zip(row_names, row_label_colors)):
            lbl = Text(name, font_size=9, color=color)
            lbl.next_to(panel3_group[i], LEFT, buff=0.12)
            row_labels_3.add(lbl)

        # Column group labels
        chunk_bracket_label = Text("Prefill chunks interleaved with decode", font_size=12, color="#45b7d1")
        chunk_bracket_label.next_to(panel3_group, DOWN, buff=0.15)

        self.play(
            panel2_all.animate.set_opacity(0),
            FadeIn(panel3_title),
            run_time=0.3,
        )
        self.play(FadeOut(panel2_all), run_time=0.1)

        # Animate column by column
        for t in range(total_cols):
            col_anims = []
            for row_idx in range(panel3_rows):
                col_anims.append(FadeIn(panel3_group[row_idx][t]))
            if t == 0:
                col_anims.extend([FadeIn(lbl) for lbl in row_labels_3])
            self.play(*col_anims, run_time=0.25)

        self.play(FadeIn(chunk_bracket_label), run_time=0.3)
        self.wait(0.5)

        # =====================================================================
        # FINAL: THROUGHPUT COMPARISON
        # =====================================================================
        panel3_all = VGroup(
            panel3_title, panel3_group, row_labels_3, chunk_bracket_label,
        )
        self.play(FadeOut(panel3_all), run_time=0.4)

        # Throughput bars
        compare_title = Text("Throughput Comparison", font_size=26, color=WHITE, weight=BOLD)
        compare_title.to_edge(UP, buff=0.8)
        self.play(Write(compare_title), run_time=0.4)

        strategies = [
            ("Static", "#ff6b6b", 1.0),
            ("Continuous", "#2ecc71", 3.0),
            ("Chunked Prefill", "#45b7d1", 4.0),
        ]

        bar_max_width = 6.0
        bar_height = 0.6
        bar_start_x = -3.0

        bars_group = VGroup()
        for i, (name, color, multiplier) in enumerate(strategies):
            bar_w = (multiplier / 4.0) * bar_max_width
            bar = Rectangle(
                width=bar_w, height=bar_height,
                stroke_color=color, stroke_width=2,
                fill_color=color, fill_opacity=0.4,
            )
            bar.move_to(
                RIGHT * (bar_start_x + bar_w / 2) + DOWN * (i * 1.0 - 0.5)
            )

            label = Text(name, font_size=16, color=color, weight=BOLD)
            label.next_to(bar, LEFT, buff=0.2)

            mult_text = Text(f"{multiplier:.0f}x", font_size=20, color=WHITE, weight=BOLD)
            mult_text.next_to(bar, RIGHT, buff=0.15)

            self.play(
                FadeIn(label),
                FadeIn(bar, shift=RIGHT * 0.3),
                FadeIn(mult_text),
                run_time=0.5,
            )
            bars_group.add(VGroup(bar, label, mult_text))

        self.wait(1.5)
