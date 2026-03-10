from manimlib import *
import numpy as np


class NonMaxSuppression(Scene):
    """Non-Maximum Suppression (NMS) visualization.

    Shows overlapping bounding boxes, sorting by confidence,
    IoU computation, and suppression of redundant boxes.
    """

    def construct(self):
        self.camera.background_rgba = [1, 1, 1, 1]

        title = Text("Non-Maximum Suppression (NMS)", font_size=32, color=BLACK, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.5)

        # -- Define bounding boxes: (x_center, y_center, width, height, confidence, class, color) --
        # All coords in scene units
        boxes_data = [
            # Class "Dog" (green family) - overlapping cluster
            {"cx": -2.0, "cy": -0.3, "w": 2.5, "h": 1.8, "conf": 0.95, "cls": "Dog", "color": GREEN_D},
            {"cx": -1.7, "cy": -0.1, "w": 2.3, "h": 1.6, "conf": 0.82, "cls": "Dog", "color": GREEN_C},
            {"cx": -2.2, "cy": -0.5, "w": 2.6, "h": 1.9, "conf": 0.71, "cls": "Dog", "color": GREEN_B},
            {"cx": -1.5, "cy":  0.0, "w": 2.0, "h": 1.5, "conf": 0.55, "cls": "Dog", "color": GREEN_A},
            # Class "Car" (blue family) - overlapping cluster
            {"cx":  2.5, "cy": -0.5, "w": 2.2, "h": 1.4, "conf": 0.91, "cls": "Car", "color": BLUE_D},
            {"cx":  2.3, "cy": -0.3, "w": 2.0, "h": 1.3, "conf": 0.78, "cls": "Car", "color": BLUE_C},
            {"cx":  2.7, "cy": -0.6, "w": 2.4, "h": 1.5, "conf": 0.60, "cls": "Car", "color": BLUE_B},
        ]

        # Sort by confidence (descending)
        boxes_data.sort(key=lambda b: b["conf"], reverse=True)

        # -- Draw all boxes --
        box_mobs = []
        conf_labels = []
        for bd in boxes_data:
            rect = Rectangle(
                width=bd["w"], height=bd["h"],
                color=bd["color"], stroke_width=2.5,
                fill_color=bd["color"], fill_opacity=0.1,
            )
            rect.move_to([bd["cx"], bd["cy"], 0])

            label = Text(
                f'{bd["cls"]} {bd["conf"]:.2f}',
                font_size=13, color=bd["color"], weight=BOLD,
            )
            label.next_to(rect, UP, buff=0.03, aligned_edge=LEFT)

            box_mobs.append(rect)
            conf_labels.append(label)

        all_boxes = VGroup(*box_mobs)
        all_labels = VGroup(*conf_labels)

        self.play(
            *[ShowCreation(b) for b in box_mobs],
            *[Write(l) for l in conf_labels],
            run_time=1,
        )
        self.wait(0.3)

        # -- Step 1: Sort by confidence --
        step1 = Text("Step 1: Sort boxes by confidence (descending)", font_size=20, color=BLACK)
        step1.to_edge(DOWN, buff=0.5)
        self.play(Write(step1), run_time=0.4)

        # Show sorted order on the side
        sorted_list = VGroup()
        for i, bd in enumerate(boxes_data):
            entry = Text(
                f'{i+1}. {bd["cls"]} = {bd["conf"]:.2f}',
                font_size=14, color=bd["color"],
            )
            sorted_list.add(entry)
        sorted_list.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        sorted_list.to_edge(RIGHT, buff=0.2)
        sorted_list.shift(UP * 0.5)

        sorted_title = Text("Sorted:", font_size=16, color=BLACK, weight=BOLD)
        sorted_title.next_to(sorted_list, UP, buff=0.15)
        self.play(Write(sorted_title), run_time=0.2)
        for entry in sorted_list:
            self.play(Write(entry), run_time=0.12)
        self.wait(0.3)

        # -- IoU helper --
        def compute_iou(b1, b2):
            x1_min = b1["cx"] - b1["w"] / 2
            x1_max = b1["cx"] + b1["w"] / 2
            y1_min = b1["cy"] - b1["h"] / 2
            y1_max = b1["cy"] + b1["h"] / 2
            x2_min = b2["cx"] - b2["w"] / 2
            x2_max = b2["cx"] + b2["w"] / 2
            y2_min = b2["cy"] - b2["h"] / 2
            y2_max = b2["cy"] + b2["h"] / 2

            inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            inter_area = inter_x * inter_y
            area1 = b1["w"] * b1["h"]
            area2 = b2["w"] * b2["h"]
            union = area1 + area2 - inter_area
            return inter_area / union if union > 0 else 0

        # -- NMS process --
        iou_threshold = 0.3
        kept_indices = []
        suppressed_indices = set()

        self.play(FadeOut(step1), run_time=0.2)

        step2 = Text(f"Step 2: NMS with IoU threshold = {iou_threshold}", font_size=20, color=BLACK)
        step2.to_edge(DOWN, buff=0.5)
        self.play(Write(step2), run_time=0.4)
        self.wait(0.2)

        for i, bd in enumerate(boxes_data):
            if i in suppressed_indices:
                continue

            # Pick this box (highest remaining confidence)
            kept_indices.append(i)

            # Highlight the kept box
            self.play(
                box_mobs[i].animate.set_stroke(width=5),
                run_time=0.2,
            )

            # Check against remaining boxes of same class
            status_text = Text(
                f"Keep {bd['cls']} ({bd['conf']:.2f}) - checking overlaps...",
                font_size=16, color=bd["color"],
            )
            status_text.to_edge(DOWN, buff=0.2)
            self.play(Write(status_text), run_time=0.3)

            for j in range(i + 1, len(boxes_data)):
                if j in suppressed_indices:
                    continue
                if boxes_data[j]["cls"] != bd["cls"]:
                    continue

                iou = compute_iou(bd, boxes_data[j])
                iou_text = Text(
                    f"IoU with box {j+1} = {iou:.2f}",
                    font_size=14, color=GREY_D,
                )
                iou_text.next_to(status_text, DOWN, buff=0.1)
                self.play(Write(iou_text), run_time=0.2)

                if iou > iou_threshold:
                    # Suppress this box
                    suppressed_indices.add(j)
                    suppress_note = Text(
                        f"  > {iou_threshold} -> SUPPRESS",
                        font_size=14, color=RED_D, weight=BOLD,
                    )
                    suppress_note.next_to(iou_text, RIGHT, buff=0.1)
                    self.play(
                        Write(suppress_note),
                        box_mobs[j].animate.set_fill(opacity=0).set_stroke(opacity=0.15),
                        conf_labels[j].animate.set_opacity(0.15),
                        sorted_list[j].animate.set_opacity(0.2),
                        run_time=0.3,
                    )
                    self.play(FadeOut(suppress_note), FadeOut(iou_text), run_time=0.15)
                else:
                    keep_note = Text(
                        f"  < {iou_threshold} -> keep",
                        font_size=14, color=GREEN_D,
                    )
                    keep_note.next_to(iou_text, RIGHT, buff=0.1)
                    self.play(Write(keep_note), run_time=0.15)
                    self.play(FadeOut(keep_note), FadeOut(iou_text), run_time=0.15)

            self.play(FadeOut(status_text), run_time=0.15)

        # -- Final result --
        self.play(FadeOut(step2), run_time=0.2)
        result = Text(
            f"NMS complete: {len(kept_indices)} boxes kept, {len(suppressed_indices)} suppressed",
            font_size=22, color=BLACK, weight=BOLD,
        )
        result.to_edge(DOWN, buff=0.4)
        result_box = SurroundingRectangle(result, color=GREEN_D, buff=0.15,
                                           stroke_width=2)

        # Mark kept boxes with a checkmark
        for idx in kept_indices:
            check = Text("Kept", font_size=12, color=GREEN_D, weight=BOLD)
            check.next_to(box_mobs[idx], DOWN, buff=0.05)
            self.play(Write(check), run_time=0.15)

        self.play(Write(result), ShowCreation(result_box), run_time=0.5)
        self.wait(1)
