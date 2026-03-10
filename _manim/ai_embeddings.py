from manimlib import *
import numpy as np


class WordEmbeddings(Scene):
    def construct(self):
        self.camera.background_rgba = [0x1a/255, 0x1a/255, 0x2e/255, 1]

        # --- Scene 1: Title ---
        title = Text("Word Embeddings in Vector Space", font_size=36, color=WHITE)
        self.play(Write(title), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(title), run_time=0.5)

        # --- Scene 2: 2D coordinate plane with word dots ---
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            width=6,
            height=6,
            axis_config={"color": GREY_B, "include_ticks": True, "tick_size": 0.05},
        ).shift(DOWN * 0.3)

        axis_label = Text("2D Vector Space", font_size=24, color=GREY_A).to_edge(UP, buff=0.3)
        self.play(ShowCreation(axes), Write(axis_label), run_time=1)

        # Word positions
        words = {
            "king":  (3, 4),
            "queen": (1, 4),
            "man":   (3, 1),
            "woman": (1, 1),
        }
        word_colors = {
            "king":  "#FFD700",
            "queen": "#FF69B4",
            "man":   "#4FC3F7",
            "woman": "#AB47BC",
        }

        dots = {}
        labels = {}
        for word, (x, y) in words.items():
            point = axes.c2p(x, y)
            dot = Dot(point, radius=0.1, color=word_colors[word])
            label = Text(word, font_size=20, color=word_colors[word]).next_to(dot, UR, buff=0.1)
            dots[word] = dot
            labels[word] = label

        # Animate dots appearing
        for word in ["king", "queen", "man", "woman"]:
            self.play(FadeIn(dots[word]), Write(labels[word]), run_time=0.4)
        self.wait(0.5)

        # --- Scene 3: Arrow from king to man labeled "- male" ---
        king_pt = axes.c2p(3, 4)
        man_pt = axes.c2p(3, 1)
        queen_pt = axes.c2p(1, 4)
        woman_pt = axes.c2p(1, 1)

        arrow_king_man = Arrow(
            king_pt, man_pt, buff=0.15, color="#FF6B6B", stroke_width=3, max_tip_length_to_length_ratio=0.1
        )
        label_km = Text("- male", font_size=16, color="#FF6B6B").next_to(arrow_king_man, RIGHT, buff=0.15)

        self.play(GrowArrow(arrow_king_man), Write(label_km), run_time=0.8)
        self.wait(0.4)

        # --- Scene 4: Parallel arrow from queen to woman labeled "- male" ---
        arrow_queen_woman = Arrow(
            queen_pt, woman_pt, buff=0.15, color="#FF6B6B", stroke_width=3, max_tip_length_to_length_ratio=0.1
        )
        label_qw = Text("- male", font_size=16, color="#FF6B6B").next_to(arrow_queen_woman, LEFT, buff=0.15)

        self.play(GrowArrow(arrow_queen_woman), Write(label_qw), run_time=0.8)
        self.wait(0.4)

        # --- Scene 5: Arrow from man to woman labeled "- male + female" ---
        arrow_man_woman = Arrow(
            man_pt, woman_pt, buff=0.15, color="#69FF6B", stroke_width=3, max_tip_length_to_length_ratio=0.1
        )
        label_mw = Text("- male + female", font_size=16, color="#69FF6B").next_to(arrow_man_woman, DOWN, buff=0.15)

        self.play(GrowArrow(arrow_man_woman), Write(label_mw), run_time=0.8)
        self.wait(0.5)

        # --- Scene 6: Show the equation ---
        equation = Text(
            "king - man + woman = queen",
            font_size=30,
            color=WHITE,
        ).to_edge(UP, buff=0.3)

        self.play(FadeOut(axis_label), run_time=0.3)
        self.play(Write(equation), run_time=1)

        # Highlight path: king -> man -> woman -> queen with a moving dot
        tracer = Dot(king_pt, radius=0.12, color=WHITE)
        self.play(FadeIn(tracer), run_time=0.3)
        self.play(tracer.animate.move_to(man_pt), run_time=0.6)
        self.play(tracer.animate.move_to(woman_pt), run_time=0.6)
        self.play(tracer.animate.move_to(queen_pt), run_time=0.6)

        # Flash the queen dot
        self.play(Flash(queen_pt, color="#FF69B4", flash_radius=0.4), run_time=0.5)
        self.play(FadeOut(tracer), run_time=0.3)
        self.wait(0.8)

        # --- Scene 7: Transition to clustering ---
        # Fade out everything from the analogy scene
        all_analogy = VGroup(
            axes, equation,
            arrow_king_man, label_km,
            arrow_queen_woman, label_qw,
            arrow_man_woman, label_mw,
            *dots.values(), *labels.values(),
        )
        self.play(FadeOut(all_analogy), run_time=0.8)

        # New axes for clustering
        axes2 = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            width=8,
            height=6,
            axis_config={"color": GREY_B, "include_ticks": True, "tick_size": 0.05},
        ).shift(DOWN * 0.3)

        cluster_title = Text("Semantic Clustering", font_size=28, color=WHITE).to_edge(UP, buff=0.3)
        self.play(ShowCreation(axes2), Write(cluster_title), run_time=0.8)

        # Cluster 1: animals/pets - bottom left region
        animal_words = {
            "cat": (2, 3),
            "dog": (2.8, 3.5),
            "pet": (2.4, 2.5),
        }
        animal_color = "#4FC3F7"

        # Cluster 2: vehicles - top right region
        vehicle_words = {
            "car":   (7, 7),
            "truck": (7.8, 7.5),
            "bus":   (7.4, 6.5),
        }
        vehicle_color = "#FFA726"

        cluster_dots = {}
        cluster_labels = {}

        # Animate animal cluster
        for word, (x, y) in animal_words.items():
            pt = axes2.c2p(x, y)
            dot = Dot(pt, radius=0.1, color=animal_color)
            label = Text(word, font_size=18, color=animal_color).next_to(dot, UR, buff=0.1)
            cluster_dots[word] = dot
            cluster_labels[word] = label
            self.play(FadeIn(dot), Write(label), run_time=0.3)

        # Draw a dashed ellipse around animals
        animal_center = axes2.c2p(2.4, 3)
        animal_ellipse = Ellipse(width=2.2, height=2.2, color=animal_color, stroke_width=1.5).move_to(animal_center)
        animal_ellipse.set_stroke(opacity=0.5)
        animal_group_label = Text("animals", font_size=16, color=animal_color).next_to(animal_ellipse, DOWN, buff=0.15)
        self.play(ShowCreation(animal_ellipse), Write(animal_group_label), run_time=0.5)

        # Animate vehicle cluster
        for word, (x, y) in vehicle_words.items():
            pt = axes2.c2p(x, y)
            dot = Dot(pt, radius=0.1, color=vehicle_color)
            label = Text(word, font_size=18, color=vehicle_color).next_to(dot, UR, buff=0.1)
            cluster_dots[word] = dot
            cluster_labels[word] = label
            self.play(FadeIn(dot), Write(label), run_time=0.3)

        # Draw a dashed ellipse around vehicles
        vehicle_center = axes2.c2p(7.4, 7)
        vehicle_ellipse = Ellipse(width=2.2, height=2.2, color=vehicle_color, stroke_width=1.5).move_to(vehicle_center)
        vehicle_ellipse.set_stroke(opacity=0.5)
        vehicle_group_label = Text("vehicles", font_size=16, color=vehicle_color).next_to(vehicle_ellipse, DOWN, buff=0.15)
        self.play(ShowCreation(vehicle_ellipse), Write(vehicle_group_label), run_time=0.5)

        self.wait(0.5)

        # --- Scene 8: Final message ---
        final_text = Text(
            "Similar meanings = nearby vectors",
            font_size=32,
            color="#E0E0E0",
        )
        # Place it in the center-right area
        final_text.move_to(axes2.c2p(5, 1.5))

        self.play(FadeOut(cluster_title), run_time=0.3)
        self.play(Write(final_text), run_time=1)
        self.wait(1.5)

        # Fade everything out
        all_cluster = VGroup(
            axes2,
            *cluster_dots.values(), *cluster_labels.values(),
            animal_ellipse, animal_group_label,
            vehicle_ellipse, vehicle_group_label,
            final_text,
        )
        self.play(FadeOut(all_cluster), run_time=0.8)
