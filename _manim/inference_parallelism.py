from manimlib import *
import numpy as np


class ParallelismStrategies(Scene):
    """Three parallelism strategies for LLM inference.

    Tensor Parallelism, Pipeline Parallelism, and Expert Parallelism
    shown side-by-side with their tradeoffs.
    """

    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Colors ---
        GPU_COLORS = ["#ff6b6b", "#4ecdc4", "#ffd93d", "#a78bfa"]
        TP_COLOR = "#ff6b6b"
        PP_COLOR = "#4ecdc4"
        EP_COLOR = "#a78bfa"
        BUBBLE_COLOR = "#555577"
        HIGHLIGHT = "#ffd93d"
        LABEL_DIM = "#8888aa"

        # =====================================================================
        # STEP 1: Title
        # =====================================================================
        title = Text("Parallelism Strategies for LLM Inference", font_size=30, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=0.6)
        self.wait(0.3)

        # =====================================================================
        # PANEL 1: Tensor Parallelism
        # =====================================================================
        self.play(FadeOut(title), run_time=0.3)

        tp_title = Text("Tensor Parallelism", font_size=28, color=TP_COLOR, weight=BOLD)
        tp_title.to_edge(UP, buff=0.3)
        tp_sub = Text("Split weight matrix columns across GPUs", font_size=18, color=LABEL_DIM)
        tp_sub.next_to(tp_title, DOWN, buff=0.2)
        self.play(FadeIn(tp_title), FadeIn(tp_sub), run_time=0.5)

        # 4x4 weight matrix
        cell_size = 0.55
        matrix_cells = VGroup()
        for r in range(4):
            for c in range(4):
                gpu_idx = c  # each column belongs to a different GPU
                rect = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=GPU_COLORS[gpu_idx], stroke_width=1.5,
                    fill_color=GPU_COLORS[gpu_idx], fill_opacity=0.3,
                )
                matrix_cells.add(rect)

        matrix_grid = matrix_cells.arrange_in_grid(n_rows=4, n_cols=4, buff=0.04)
        matrix_grid.move_to(UP * 0.5 + LEFT * 2.0)

        matrix_label = Text("Weight Matrix W", font_size=16, color=WHITE)
        matrix_label.next_to(matrix_grid, UP, buff=0.15)

        # Column brackets and GPU labels
        gpu_labels = VGroup()
        for i in range(4):
            col_cells = VGroup(*[matrix_cells[r * 4 + i] for r in range(4)])
            gl = Text(f"GPU {i}", font_size=12, color=GPU_COLORS[i])
            gl.next_to(col_cells, DOWN, buff=0.12)
            gpu_labels.add(gl)

        self.play(
            FadeIn(matrix_label),
            FadeIn(matrix_grid),
            *[FadeIn(gl) for gl in gpu_labels],
            run_time=0.7,
        )
        self.wait(0.3)

        # Each GPU computes its slice - show 4 GPU boxes on the right
        gpu_boxes = VGroup()
        for i in range(4):
            box = Rectangle(
                width=1.4, height=0.7,
                stroke_color=GPU_COLORS[i], stroke_width=2,
                fill_color=GPU_COLORS[i], fill_opacity=0.15,
            )
            label = Text(f"GPU {i}", font_size=14, color=GPU_COLORS[i])
            label.move_to(box.get_center())
            gpu_boxes.add(VGroup(box, label))

        gpu_boxes.arrange(DOWN, buff=0.15)
        gpu_boxes.move_to(RIGHT * 2.0 + UP * 0.3)

        # Arrows from matrix columns to GPU boxes
        col_arrows = VGroup()
        for i in range(4):
            col_cells = VGroup(*[matrix_cells[r * 4 + i] for r in range(4)])
            arrow = Arrow(
                col_cells.get_right(), gpu_boxes[i].get_left(),
                buff=0.1, color=GPU_COLORS[i], stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            col_arrows.add(arrow)

        self.play(
            *[FadeIn(gb) for gb in gpu_boxes],
            *[GrowArrow(a) for a in col_arrows],
            run_time=0.6,
        )

        # AllReduce step
        allreduce_box = Rectangle(
            width=2.0, height=0.6,
            stroke_color=HIGHLIGHT, stroke_width=2,
            fill_color=HIGHLIGHT, fill_opacity=0.15,
        )
        allreduce_box.next_to(gpu_boxes, RIGHT, buff=0.6)
        allreduce_label = Text("AllReduce", font_size=16, color=HIGHLIGHT, weight=BOLD)
        allreduce_label.move_to(allreduce_box.get_center())

        reduce_arrows = VGroup()
        for gb in gpu_boxes:
            arrow = Arrow(
                gb.get_right(), allreduce_box.get_left(),
                buff=0.1, color=HIGHLIGHT, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            reduce_arrows.add(arrow)

        self.play(
            FadeIn(allreduce_box), FadeIn(allreduce_label),
            *[GrowArrow(a) for a in reduce_arrows],
            run_time=0.5,
        )

        nvlink_label = Text("NVLink required (high bandwidth)", font_size=16, color=HIGHLIGHT)
        nvlink_label.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(nvlink_label), run_time=0.4)
        self.wait(0.5)

        # Clear panel 1
        panel1 = VGroup(
            tp_title, tp_sub, matrix_label, matrix_grid, gpu_labels,
            gpu_boxes, col_arrows, allreduce_box, allreduce_label,
            reduce_arrows, nvlink_label,
        )
        self.play(FadeOut(panel1), run_time=0.4)

        # =====================================================================
        # PANEL 2: Pipeline Parallelism
        # =====================================================================
        pp_title = Text("Pipeline Parallelism", font_size=28, color=PP_COLOR, weight=BOLD)
        pp_title.to_edge(UP, buff=0.3)
        pp_sub = Text("Split layers across GPUs sequentially", font_size=18, color=LABEL_DIM)
        pp_sub.next_to(pp_title, DOWN, buff=0.2)
        self.play(FadeIn(pp_title), FadeIn(pp_sub), run_time=0.5)

        # 3 GPU stages with layer ranges
        stage_data = [
            ("GPU 0", "Layers 1-10", GPU_COLORS[0]),
            ("GPU 1", "Layers 11-20", GPU_COLORS[1]),
            ("GPU 2", "Layers 21-30", GPU_COLORS[2]),
        ]

        stage_boxes = VGroup()
        for gpu_name, layer_range, color in stage_data:
            box = Rectangle(
                width=2.8, height=1.2,
                stroke_color=color, stroke_width=2,
                fill_color=color, fill_opacity=0.15,
            )
            name = Text(gpu_name, font_size=16, color=color, weight=BOLD)
            layers = Text(layer_range, font_size=14, color=WHITE)
            content = VGroup(name, layers).arrange(DOWN, buff=0.1)
            content.move_to(box.get_center())
            stage_boxes.add(VGroup(box, content))

        stage_boxes.arrange(RIGHT, buff=0.5)
        stage_boxes.move_to(UP * 0.3)

        # Data flow arrows
        flow_arrows = VGroup()
        for i in range(2):
            arrow = Arrow(
                stage_boxes[i].get_right(), stage_boxes[i + 1].get_left(),
                buff=0.1, color=WHITE, stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            flow_arrows.add(arrow)

        self.play(
            *[FadeIn(sb) for sb in stage_boxes],
            *[GrowArrow(a) for a in flow_arrows],
            run_time=0.7,
        )
        self.wait(0.3)

        # Pipeline timeline showing bubbles
        timeline_label = Text("Timeline (single batch):", font_size=16, color=LABEL_DIM)
        timeline_label.move_to(DOWN * 1.0 + LEFT * 4.0)
        self.play(FadeIn(timeline_label), run_time=0.3)

        # Timeline grid: 3 rows (GPUs), 6 time slots
        tl_cell_w = 1.0
        tl_cell_h = 0.4
        tl_anchor = DOWN * 1.8 + LEFT * 2.5

        timeline_cells = VGroup()
        gpu_tl_labels = VGroup()

        for gpu_i in range(3):
            gl = Text(f"GPU {gpu_i}", font_size=12, color=GPU_COLORS[gpu_i])
            gl.move_to(tl_anchor + DOWN * gpu_i * (tl_cell_h + 0.06) + LEFT * 1.8)
            gpu_tl_labels.add(gl)

            for t in range(6):
                # GPU i is active during time slots [i, i+1] for a single batch
                is_active = (t == gpu_i)
                is_bubble = not is_active
                fill_col = GPU_COLORS[gpu_i] if is_active else BUBBLE_COLOR
                fill_op = 0.6 if is_active else 0.2

                cell = Rectangle(
                    width=tl_cell_w, height=tl_cell_h,
                    stroke_color=fill_col, stroke_width=1,
                    fill_color=fill_col, fill_opacity=fill_op,
                )
                cell.move_to(
                    tl_anchor + RIGHT * t * (tl_cell_w + 0.04) +
                    DOWN * gpu_i * (tl_cell_h + 0.06)
                )
                timeline_cells.add(cell)

        bubble_label = Text("Gray = idle (pipeline bubble)", font_size=14, color=BUBBLE_COLOR)
        bubble_label.next_to(timeline_cells, DOWN, buff=0.2)

        self.play(
            *[FadeIn(c) for c in timeline_cells],
            *[FadeIn(gl) for gl in gpu_tl_labels],
            FadeIn(bubble_label),
            run_time=0.6,
        )
        self.wait(0.4)

        # Micro-batching reduces bubbles
        micro_label = Text("With micro-batching: bubbles shrink", font_size=18, color=PP_COLOR)
        micro_label.next_to(bubble_label, DOWN, buff=0.25)

        # Show improved timeline: GPUs overlap more
        micro_cells = VGroup()
        micro_anchor = micro_label.get_bottom() + DOWN * 0.4 + LEFT * 2.5

        # 3 micro-batches, each GPU processes them in sequence
        # GPU 0: [mb1, mb2, mb3, -, -, -]
        # GPU 1: [-, mb1, mb2, mb3, -, -]
        # GPU 2: [-, -, mb1, mb2, mb3, -]
        mb_colors = ["#ff6b6b", "#4ecdc4", "#ffd93d"]
        micro_cell_w = 0.7

        for gpu_i in range(3):
            for t in range(6):
                mb_idx = t - gpu_i
                is_active = (0 <= mb_idx < 3)
                fill_col = mb_colors[mb_idx] if is_active else BUBBLE_COLOR
                fill_op = 0.5 if is_active else 0.15

                cell = Rectangle(
                    width=micro_cell_w, height=tl_cell_h,
                    stroke_color=fill_col if is_active else BUBBLE_COLOR,
                    stroke_width=1,
                    fill_color=fill_col, fill_opacity=fill_op,
                )
                cell.move_to(
                    micro_anchor + RIGHT * t * (micro_cell_w + 0.04) +
                    DOWN * gpu_i * (tl_cell_h + 0.06)
                )
                if is_active:
                    mb_text = Text(f"mb{mb_idx + 1}", font_size=9, color=WHITE)
                    mb_text.move_to(cell.get_center())
                    micro_cells.add(VGroup(cell, mb_text))
                else:
                    micro_cells.add(cell)

        self.play(
            Write(micro_label),
            *[FadeIn(c) for c in micro_cells],
            run_time=0.7,
        )
        self.wait(0.5)

        # Clear panel 2
        panel2 = VGroup(
            pp_title, pp_sub, stage_boxes, flow_arrows,
            timeline_label, timeline_cells, gpu_tl_labels,
            bubble_label, micro_label, micro_cells,
        )
        self.play(FadeOut(panel2), run_time=0.4)

        # =====================================================================
        # PANEL 3: Expert Parallelism
        # =====================================================================
        ep_title = Text("Expert Parallelism", font_size=28, color=EP_COLOR, weight=BOLD)
        ep_title.to_edge(UP, buff=0.3)
        ep_sub = Text("Distribute MoE experts across GPUs", font_size=18, color=LABEL_DIM)
        ep_sub.next_to(ep_title, DOWN, buff=0.2)
        self.play(FadeIn(ep_title), FadeIn(ep_sub), run_time=0.5)

        # 4 GPU boxes, 2 experts each
        ep_gpu_groups = VGroup()
        expert_rects = {}

        for gpu_i in range(4):
            gpu_box = Rectangle(
                width=2.5, height=1.6,
                stroke_color=GPU_COLORS[gpu_i], stroke_width=2,
                fill_color=GPU_COLORS[gpu_i], fill_opacity=0.08,
            )
            gpu_label = Text(f"GPU {gpu_i}", font_size=14, color=GPU_COLORS[gpu_i], weight=BOLD)
            gpu_label.next_to(gpu_box, UP, buff=0.08)

            # 2 expert boxes inside
            experts = VGroup()
            for e in range(2):
                eidx = gpu_i * 2 + e
                ebox = Rectangle(
                    width=0.9, height=0.5,
                    stroke_color=WHITE, stroke_width=1.5,
                    fill_color="#444466", fill_opacity=0.7,
                )
                elabel = Text(f"E{eidx + 1}", font_size=12, color=WHITE)
                elabel.move_to(ebox.get_center())
                expert_rects[eidx] = VGroup(ebox, elabel)
                experts.add(expert_rects[eidx])

            experts.arrange(RIGHT, buff=0.2)
            experts.move_to(gpu_box.get_center())
            ep_gpu_groups.add(VGroup(gpu_box, gpu_label, experts))

        ep_gpu_groups.arrange_in_grid(n_rows=2, n_cols=2, buff=0.4)
        ep_gpu_groups.move_to(DOWN * 0.2)

        self.play(
            *[FadeIn(g) for g in ep_gpu_groups],
            run_time=0.7,
        )
        self.wait(0.3)

        # Token enters and router picks 2 experts on different GPUs
        token_box = Rectangle(
            width=1.0, height=0.45,
            stroke_color=HIGHLIGHT, stroke_width=2,
            fill_color=HIGHLIGHT, fill_opacity=0.2,
        )
        token_text = Text("Token", font_size=14, color=HIGHLIGHT)
        token_text.move_to(token_box.get_center())
        token_group = VGroup(token_box, token_text)
        token_group.move_to(UP * 2.5 + LEFT * 4.0)

        router_box = Rectangle(
            width=1.4, height=0.5,
            stroke_color=HIGHLIGHT, stroke_width=2,
            fill_color="#3a3a1e", fill_opacity=0.7,
        )
        router_label = Text("Router", font_size=14, color=HIGHLIGHT)
        router_label.move_to(router_box.get_center())
        router_group = VGroup(router_box, router_label)
        router_group.move_to(UP * 2.5)

        self.play(FadeIn(token_group), FadeIn(router_group), run_time=0.4)
        self.play(
            token_group.animate.move_to(router_group.get_left() + LEFT * 0.8),
            run_time=0.4,
        )

        # Router selects E2 (GPU 0) and E5 (GPU 2)
        selected = [1, 4]  # E2 and E5
        select_label = Text("Selects E2 + E5", font_size=14, color=HIGHLIGHT)
        select_label.next_to(router_group, RIGHT, buff=0.3)

        # Highlight selected experts
        highlight_anims = []
        for idx in selected:
            highlight_anims.append(
                expert_rects[idx][0].animate.set_fill("#52b788", opacity=0.7)
            )

        self.play(
            *highlight_anims,
            FadeIn(select_label),
            run_time=0.5,
        )

        # All-to-All communication label
        a2a_label = Text("All-to-All send", font_size=16, color=EP_COLOR)
        a2a_label.to_edge(DOWN, buff=0.5)

        # Arrows from router to selected experts
        route_arrows = VGroup()
        for idx in selected:
            arrow = Arrow(
                router_group.get_bottom(),
                expert_rects[idx].get_top(),
                buff=0.1, color="#52b788", stroke_width=2,
                max_tip_length_to_length_ratio=0.15,
            )
            route_arrows.add(arrow)

        self.play(
            *[GrowArrow(a) for a in route_arrows],
            FadeIn(a2a_label),
            run_time=0.5,
        )
        self.wait(0.5)

        # Clear panel 3
        panel3 = VGroup(
            ep_title, ep_sub, ep_gpu_groups,
            token_group, router_group, select_label,
            route_arrows, a2a_label,
        )
        self.play(FadeOut(panel3), run_time=0.4)

        # =====================================================================
        # FINAL: Comparison summary
        # =====================================================================
        comp_title = Text("When to Use Each Strategy", font_size=26, color=WHITE, weight=BOLD)
        comp_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(comp_title), run_time=0.3)

        strategies = [
            ("Tensor Parallelism", "Low latency within a node\nNeeds NVLink", TP_COLOR),
            ("Pipeline Parallelism", "Scales across nodes\nHigher latency", PP_COLOR),
            ("Expert Parallelism", "For MoE architectures\nAll-to-All comms", EP_COLOR),
        ]

        strat_panels = VGroup()
        for name, desc, color in strategies:
            box = Rectangle(
                width=3.8, height=2.2,
                stroke_color=color, stroke_width=2,
                fill_color=color, fill_opacity=0.08,
            )
            name_text = Text(name, font_size=18, color=color, weight=BOLD)
            desc_text = Text(desc, font_size=14, color=WHITE)
            content = VGroup(name_text, desc_text).arrange(DOWN, buff=0.2)
            content.move_to(box.get_center())
            strat_panels.add(VGroup(box, content))

        strat_panels.arrange(RIGHT, buff=0.4)
        strat_panels.next_to(comp_title, DOWN, buff=0.5)

        self.play(
            *[FadeIn(sp, shift=UP * 0.2) for sp in strat_panels],
            run_time=0.7,
        )

        # Bottom note
        note = Text(
            "Production systems combine all three: TP within node, PP across nodes, EP for MoE",
            font_size=16, color=HIGHLIGHT,
        )
        note.next_to(strat_panels, DOWN, buff=0.5)
        self.play(Write(note), run_time=0.8)
        self.wait(1.5)
