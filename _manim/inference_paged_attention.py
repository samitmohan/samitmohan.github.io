from manimlib import *
import numpy as np


class PagedAttention(Scene):
    """Virtual memory analogy for KV cache management.

    Shows traditional contiguous allocation waste, paged allocation
    with block tables, copy-on-write for shared prefixes, and the
    memory utilization comparison.
    """

    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        REQ_COLORS = ["#ff6b6b", "#4ecdc4", "#a78bfa"]
        EMPTY_COLOR = "#333355"
        WASTE_COLOR = "#ff4444"
        PAGE_COLOR = "#52b788"
        SHARED_COLOR = "#2ecc71"
        HIGHLIGHT = "#ffd93d"
        LABEL_DIM = "#8888aa"

        # =====================================================================
        # STEP 1: Title
        # =====================================================================
        title = Text("PagedAttention: Virtual Memory for KV Cache", font_size=28, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # =====================================================================
        # STEP 2: Traditional contiguous allocation
        # =====================================================================
        self.play(FadeOut(title), run_time=0.3)

        trad_title = Text("Traditional Allocation", font_size=26, color=WASTE_COLOR, weight=BOLD)
        trad_title.to_edge(UP, buff=0.3)
        trad_sub = Text("Each request pre-allocates max_seq_len contiguous memory", font_size=16, color=LABEL_DIM)
        trad_sub.next_to(trad_title, DOWN, buff=0.2)
        self.play(FadeIn(trad_title), FadeIn(trad_sub), run_time=0.5)

        # VRAM label
        vram_label = Text("GPU VRAM", font_size=18, color=LABEL_DIM)
        vram_label.move_to(LEFT * 5.5 + UP * 0.5)
        self.play(FadeIn(vram_label), run_time=0.3)

        # 3 requests, each allocates a large block but only uses part of it
        block_w = 0.35
        block_h = 0.5
        req_names = ["Req A", "Req B", "Req C"]
        # Each request has 10 slots, but uses only [4, 3, 5] of them
        total_slots = 10
        used_slots = [4, 3, 5]

        all_req_groups = VGroup()
        req_labels_group = VGroup()

        for ri in range(3):
            req_label = Text(req_names[ri], font_size=14, color=REQ_COLORS[ri], weight=BOLD)

            blocks = VGroup()
            for si in range(total_slots):
                is_used = si < used_slots[ri]
                fill_col = REQ_COLORS[ri] if is_used else EMPTY_COLOR
                fill_op = 0.5 if is_used else 0.15
                stroke_col = REQ_COLORS[ri] if is_used else "#555577"

                rect = Rectangle(
                    width=block_w, height=block_h,
                    stroke_color=stroke_col, stroke_width=1.2,
                    fill_color=fill_col, fill_opacity=fill_op,
                )
                blocks.add(rect)

            blocks.arrange(RIGHT, buff=0.03)
            req_label.next_to(blocks, LEFT, buff=0.2)

            req_group = VGroup(blocks)
            all_req_groups.add(req_group)
            req_labels_group.add(req_label)

        all_req_groups.arrange(DOWN, buff=0.25)
        all_req_groups.move_to(RIGHT * 0.5 + UP * 0.3)

        for ri in range(3):
            req_labels_group[ri].next_to(all_req_groups[ri], LEFT, buff=0.2)

        self.play(
            *[FadeIn(rg) for rg in all_req_groups],
            *[FadeIn(rl) for rl in req_labels_group],
            run_time=0.7,
        )
        self.wait(0.3)

        # Waste indicators
        waste_labels = VGroup()
        for ri in range(3):
            wasted = total_slots - used_slots[ri]
            pct = int(100 * wasted / total_slots)
            wl = Text(f"{pct}% wasted", font_size=12, color=WASTE_COLOR)
            wl.next_to(all_req_groups[ri], RIGHT, buff=0.3)
            waste_labels.add(wl)

        total_waste = sum(total_slots - u for u in used_slots)
        total_pct = int(100 * total_waste / (total_slots * 3))
        total_waste_text = Text(
            f"Total: {total_pct}% memory wasted (fragmentation + over-allocation)",
            font_size=18, color=WASTE_COLOR, weight=BOLD,
        )
        total_waste_text.next_to(all_req_groups, DOWN, buff=0.5)

        self.play(
            *[FadeIn(wl) for wl in waste_labels],
            Write(total_waste_text),
            run_time=0.6,
        )
        self.wait(0.5)

        # Clear step 2
        step2 = VGroup(
            trad_title, trad_sub, vram_label,
            all_req_groups, req_labels_group,
            waste_labels, total_waste_text,
        )
        self.play(FadeOut(step2), run_time=0.4)

        # =====================================================================
        # STEP 3: PagedAttention - scattered blocks via page table
        # =====================================================================
        paged_title = Text("PagedAttention", font_size=26, color=PAGE_COLOR, weight=BOLD)
        paged_title.to_edge(UP, buff=0.3)
        paged_sub = Text("Non-contiguous blocks mapped through a block table", font_size=16, color=LABEL_DIM)
        paged_sub.next_to(paged_title, DOWN, buff=0.2)
        self.play(FadeIn(paged_title), FadeIn(paged_sub), run_time=0.5)

        # Block table on the left
        bt_title = Text("Block Table", font_size=16, color=WHITE, weight=BOLD)
        bt_title.move_to(LEFT * 5.0 + UP * 1.0)

        # 3 requests, each with a few logical blocks mapped to physical blocks
        logical_maps = [
            [(0, 2), (1, 7), (2, 11)],   # Req A: logical 0->phys 2, 1->7, 2->11
            [(0, 0), (1, 5), (2, 9)],     # Req B
            [(0, 3), (1, 6), (2, 8), (3, 10)],  # Req C
        ]

        bt_entries = VGroup()
        bt_labels = VGroup()
        for ri in range(3):
            rl = Text(req_names[ri], font_size=12, color=REQ_COLORS[ri], weight=BOLD)
            entries = VGroup()
            for logical, physical in logical_maps[ri]:
                entry = Text(f"L{logical}->P{physical}", font_size=10, color=WHITE)
                entries.add(entry)
            entries.arrange(RIGHT, buff=0.15)
            row = VGroup(rl, entries).arrange(RIGHT, buff=0.2)
            bt_entries.add(row)

        bt_entries.arrange(DOWN, buff=0.15)
        bt_group = VGroup(bt_title, bt_entries).arrange(DOWN, buff=0.2)
        bt_group.move_to(LEFT * 4.5 + DOWN * 0.2)

        self.play(FadeIn(bt_group), run_time=0.6)

        # Physical VRAM blocks on the right - 12 blocks in a 3x4 grid
        phys_label = Text("Physical VRAM Blocks", font_size=16, color=WHITE, weight=BOLD)
        phys_label.move_to(RIGHT * 2.0 + UP * 1.0)

        phys_blocks = VGroup()
        # Map physical block -> which request owns it
        phys_owners = {}
        for ri in range(3):
            for _, phys in logical_maps[ri]:
                phys_owners[phys] = ri

        for pi in range(12):
            owner = phys_owners.get(pi, -1)
            fill_col = REQ_COLORS[owner] if owner >= 0 else EMPTY_COLOR
            fill_op = 0.5 if owner >= 0 else 0.1
            stroke_col = REQ_COLORS[owner] if owner >= 0 else "#555577"

            rect = Rectangle(
                width=0.7, height=0.5,
                stroke_color=stroke_col, stroke_width=1.5,
                fill_color=fill_col, fill_opacity=fill_op,
            )
            plabel = Text(f"P{pi}", font_size=10, color=WHITE)
            plabel.move_to(rect.get_center())
            phys_blocks.add(VGroup(rect, plabel))

        phys_grid = phys_blocks.arrange_in_grid(n_rows=3, n_cols=4, buff=0.08)
        phys_grid.move_to(RIGHT * 2.0 + DOWN * 0.3)

        self.play(
            FadeIn(phys_label),
            *[FadeIn(pb) for pb in phys_blocks],
            run_time=0.7,
        )

        scatter_note = Text(
            "Blocks scatter across VRAM - no contiguous requirement",
            font_size=16, color=PAGE_COLOR,
        )
        scatter_note.next_to(phys_grid, DOWN, buff=0.35)
        self.play(FadeIn(scatter_note), run_time=0.4)
        self.wait(0.5)

        # Clear step 3
        step3 = VGroup(
            paged_title, paged_sub, bt_group,
            phys_label, phys_grid, scatter_note,
        )
        self.play(FadeOut(step3), run_time=0.4)

        # =====================================================================
        # STEP 4: Copy-on-write for shared prefixes
        # =====================================================================
        cow_title = Text("Copy-on-Write: Shared Prefixes", font_size=26, color=SHARED_COLOR, weight=BOLD)
        cow_title.to_edge(UP, buff=0.3)
        cow_sub = Text("Two requests share a system prompt", font_size=16, color=LABEL_DIM)
        cow_sub.next_to(cow_title, DOWN, buff=0.2)
        self.play(FadeIn(cow_title), FadeIn(cow_sub), run_time=0.5)

        # Shared prefix blocks
        shared_label = Text("System Prompt (shared)", font_size=16, color=SHARED_COLOR)
        shared_label.move_to(UP * 0.8)

        shared_blocks = VGroup()
        for i in range(4):
            rect = Rectangle(
                width=0.9, height=0.55,
                stroke_color=SHARED_COLOR, stroke_width=2,
                fill_color=SHARED_COLOR, fill_opacity=0.4,
            )
            txt = Text(f"S{i}", font_size=12, color=WHITE)
            txt.move_to(rect.get_center())
            shared_blocks.add(VGroup(rect, txt))

        shared_blocks.arrange(RIGHT, buff=0.08)
        shared_blocks.next_to(shared_label, DOWN, buff=0.2)

        self.play(FadeIn(shared_label), FadeIn(shared_blocks), run_time=0.5)

        # Two requests diverge from the shared prefix
        req_a_label = Text("Req A (diverges)", font_size=14, color=REQ_COLORS[0])
        req_a_blocks = VGroup()
        for i in range(2):
            rect = Rectangle(
                width=0.9, height=0.55,
                stroke_color=REQ_COLORS[0], stroke_width=2,
                fill_color=REQ_COLORS[0], fill_opacity=0.4,
            )
            txt = Text(f"A{i}", font_size=12, color=WHITE)
            txt.move_to(rect.get_center())
            req_a_blocks.add(VGroup(rect, txt))

        req_a_blocks.arrange(RIGHT, buff=0.08)

        req_b_label = Text("Req B (diverges)", font_size=14, color=REQ_COLORS[1])
        req_b_blocks = VGroup()
        for i in range(3):
            rect = Rectangle(
                width=0.9, height=0.55,
                stroke_color=REQ_COLORS[1], stroke_width=2,
                fill_color=REQ_COLORS[1], fill_opacity=0.4,
            )
            txt = Text(f"B{i}", font_size=12, color=WHITE)
            txt.move_to(rect.get_center())
            req_b_blocks.add(VGroup(rect, txt))

        req_b_blocks.arrange(RIGHT, buff=0.08)

        # Position divergent blocks below shared, branching left and right
        req_a_label.move_to(DOWN * 0.6 + LEFT * 3.0)
        req_a_blocks.next_to(req_a_label, DOWN, buff=0.15)

        req_b_label.move_to(DOWN * 0.6 + RIGHT * 3.0)
        req_b_blocks.next_to(req_b_label, DOWN, buff=0.15)

        # Arrows from shared to each request
        arrow_a = Arrow(
            shared_blocks.get_bottom() + LEFT * 0.5,
            req_a_label.get_top(),
            buff=0.1, color=REQ_COLORS[0], stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        arrow_b = Arrow(
            shared_blocks.get_bottom() + RIGHT * 0.5,
            req_b_label.get_top(),
            buff=0.1, color=REQ_COLORS[1], stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )

        self.play(
            GrowArrow(arrow_a), GrowArrow(arrow_b),
            FadeIn(req_a_label), FadeIn(req_b_label),
            FadeIn(req_a_blocks), FadeIn(req_b_blocks),
            run_time=0.7,
        )

        # Copy-on-write explanation
        cow_note = Text(
            "Shared blocks: ref count = 2. Only allocate new blocks when content diverges.",
            font_size=14, color=SHARED_COLOR,
        )
        cow_note.next_to(VGroup(req_a_blocks, req_b_blocks), DOWN, buff=0.4)

        savings = Text(
            "4 shared blocks saved - no duplication of system prompt",
            font_size=16, color=HIGHLIGHT,
        )
        savings.next_to(cow_note, DOWN, buff=0.2)

        self.play(Write(cow_note), run_time=0.5)
        self.play(FadeIn(savings), run_time=0.4)
        self.wait(0.5)

        # Clear step 4
        step4 = VGroup(
            cow_title, cow_sub, shared_label, shared_blocks,
            req_a_label, req_a_blocks, req_b_label, req_b_blocks,
            arrow_a, arrow_b, cow_note, savings,
        )
        self.play(FadeOut(step4), run_time=0.4)

        # =====================================================================
        # STEP 5: Memory utilization comparison
        # =====================================================================
        comp_title = Text("Memory Utilization", font_size=26, color=WHITE, weight=BOLD)
        comp_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(comp_title), run_time=0.3)

        # Traditional bar
        trad_bar_bg = Rectangle(
            width=8.0, height=0.8,
            stroke_color=WASTE_COLOR, stroke_width=2,
            fill_color=WASTE_COLOR, fill_opacity=0.1,
        )
        trad_bar_bg.move_to(UP * 0.8 + RIGHT * 0.5)

        trad_bar_used = Rectangle(
            width=8.0 * 0.4, height=0.8,
            stroke_color=WASTE_COLOR, stroke_width=0,
            fill_color=WASTE_COLOR, fill_opacity=0.5,
        )
        trad_bar_used.align_to(trad_bar_bg, LEFT)

        trad_label = Text("Traditional", font_size=18, color=WASTE_COLOR, weight=BOLD)
        trad_label.next_to(trad_bar_bg, LEFT, buff=0.3)

        trad_pct = Text("~60% wasted", font_size=16, color=WASTE_COLOR)
        trad_pct.next_to(trad_bar_bg, RIGHT, buff=0.2)

        # Shade the wasted portion
        trad_waste_region = Rectangle(
            width=8.0 * 0.6, height=0.8,
            stroke_color=WASTE_COLOR, stroke_width=0,
            fill_color=EMPTY_COLOR, fill_opacity=0.3,
        )
        trad_waste_region.align_to(trad_bar_bg, RIGHT)

        # Paged bar
        paged_bar_bg = Rectangle(
            width=8.0, height=0.8,
            stroke_color=PAGE_COLOR, stroke_width=2,
            fill_color=PAGE_COLOR, fill_opacity=0.1,
        )
        paged_bar_bg.move_to(DOWN * 0.5 + RIGHT * 0.5)

        paged_bar_used = Rectangle(
            width=8.0 * 0.96, height=0.8,
            stroke_color=PAGE_COLOR, stroke_width=0,
            fill_color=PAGE_COLOR, fill_opacity=0.5,
        )
        paged_bar_used.align_to(paged_bar_bg, LEFT)

        paged_label = Text("Paged", font_size=18, color=PAGE_COLOR, weight=BOLD)
        paged_label.next_to(paged_bar_bg, LEFT, buff=0.3)

        paged_pct = Text("<4% wasted", font_size=16, color=PAGE_COLOR)
        paged_pct.next_to(paged_bar_bg, RIGHT, buff=0.2)

        self.play(
            FadeIn(trad_bar_bg), FadeIn(trad_bar_used),
            FadeIn(trad_waste_region),
            FadeIn(trad_label), FadeIn(trad_pct),
            run_time=0.6,
        )
        self.wait(0.3)

        self.play(
            FadeIn(paged_bar_bg), FadeIn(paged_bar_used),
            FadeIn(paged_label), FadeIn(paged_pct),
            run_time=0.6,
        )

        # Highlight paged bar
        paged_glow = SurroundingRectangle(
            VGroup(paged_bar_bg, paged_pct), color=PAGE_COLOR, buff=0.1, stroke_width=3
        )
        self.play(ShowCreation(paged_glow), run_time=0.4)

        # Bottom summary
        summary = Text(
            "PagedAttention: near-zero waste, dynamic allocation, shared prefixes",
            font_size=18, color=HIGHLIGHT,
        )
        summary.next_to(paged_bar_bg, DOWN, buff=0.7)
        self.play(Write(summary), run_time=0.8)

        # vLLM credit
        credit = Text(
            "Introduced by vLLM (Kwon et al., 2023)",
            font_size=14, color=LABEL_DIM,
        )
        credit.next_to(summary, DOWN, buff=0.25)
        self.play(FadeIn(credit), run_time=0.4)
        self.wait(1.5)
