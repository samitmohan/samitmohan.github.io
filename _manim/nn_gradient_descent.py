from manimlib import *
import numpy as np


class GradientDescent(Scene):
    """3D gradient descent visualization showing a ball rolling down a loss surface."""

    def construct(self):
        frame = self.camera.frame
        frame.set_euler_angles(theta=-45 * DEGREES, phi=60 * DEGREES)

        # Loss surface: f(x,y) = x^2 + y^2 (bowl shape)
        surface = ParametricSurface(
            lambda u, v: [u, v, u**2 + v**2],
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
        )
        surface.set_opacity(0.6)
        surface.set_color_by_gradient(BLUE_E, GREEN_E, YELLOW_E)

        # Axes without numbers (avoids LaTeX)
        axes = ThreeDAxes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            z_range=[0, 5, 1],
            width=5,
            height=5,
            depth=4,
            axis_config={"include_numbers": False, "include_ticks": False},
        )

        # Labels
        title = Text("Gradient Descent on Loss Surface", font_size=28, color=WHITE)
        title.to_corner(UL)
        title.fix_in_frame()

        self.play(ShowCreation(axes), ShowCreation(surface), Write(title), run_time=1.5)
        self.wait(0.3)

        # Gradient descent trajectory
        lr = 0.3
        pos = np.array([1.8, 1.5])
        path_points = [pos.copy()]

        for _ in range(12):
            grad = 2 * pos
            pos = pos - lr * grad
            path_points.append(pos.copy())

        # Ball
        ball = Sphere(radius=0.08, color=RED)
        start_3d = np.array([path_points[0][0], path_points[0][1],
                             path_points[0][0]**2 + path_points[0][1]**2])
        ball.move_to(axes.c2p(*start_3d))
        self.play(ShowCreation(ball), run_time=0.5)

        # Step label
        step_label = Text("Step 0", font_size=22, color=YELLOW)
        step_label.to_corner(UR)
        step_label.fix_in_frame()
        self.play(Write(step_label), run_time=0.3)

        for i in range(1, len(path_points)):
            x, y = path_points[i]
            z = x**2 + y**2
            new_pos = axes.c2p(x, y, z)

            px, py = path_points[i - 1]
            pz = px**2 + py**2
            gx, gy = -2 * px, -2 * py
            scale = 0.3
            arrow = Arrow(
                start=axes.c2p(px, py, pz),
                end=axes.c2p(px + scale * gx, py + scale * gy, pz),
                color=YELLOW,
            )

            new_step = Text(f"Step {i}  lr={lr}", font_size=22, color=YELLOW)
            new_step.to_corner(UR)

            seg = Line(
                start=axes.c2p(px, py, pz),
                end=axes.c2p(x, y, z),
                color=RED_A,
            )

            self.remove(step_label)
            new_step.fix_in_frame()
            step_label = new_step

            anim_time = max(0.15, 0.5 - i * 0.03)
            self.play(
                ball.animate.move_to(new_pos),
                ShowCreation(seg),
                ShowCreation(arrow),
                run_time=anim_time,
            )
            self.play(FadeOut(arrow), run_time=0.1)

        final = Text("Converged to minimum!", font_size=24, color=GREEN)
        final.to_corner(DL)
        final.fix_in_frame()
        self.play(Write(final), run_time=0.5)
        self.wait(0.5)


class GradientDescent2D(Scene):
    """2D gradient descent - simpler fallback."""

    def construct(self):
        self.camera.background_rgba = [1, 1, 1, 1]

        # Axes without number labels (no LaTeX needed)
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 9, 2],
            width=8,
            height=5,
            axis_config={"color": BLACK, "include_numbers": False, "include_ticks": True},
        )
        axes.set_color(BLACK)

        # Manual axis labels using Text
        x_label = Text("x", font_size=20, color=BLACK)
        x_label.next_to(axes.x_axis, RIGHT, buff=0.1)
        y_label = Text("Loss", font_size=20, color=BLACK)
        y_label.next_to(axes.y_axis, UP, buff=0.1)

        # Loss function: f(x) = x^2
        curve = axes.get_graph(lambda x: x**2, color=BLUE_D, x_range=[-3, 3])
        curve_label = Text("Loss = x^2", font_size=22, color=BLUE_D)
        curve_label.next_to(axes, UP)

        title = Text("Gradient Descent", font_size=30, color=BLACK)
        title.to_corner(UL)

        self.play(ShowCreation(axes), Write(x_label), Write(y_label),
                  ShowCreation(curve), Write(curve_label), Write(title), run_time=1)

        # Gradient descent
        lr = 0.3
        x_val = 2.5
        dot = Dot(axes.c2p(x_val, x_val**2), color=RED, radius=0.1)
        self.play(ShowCreation(dot), run_time=0.3)

        info = Text(f"x={x_val:.2f}, loss={x_val**2:.2f}", font_size=20, color=BLACK)
        info.to_corner(UR)
        self.play(Write(info), run_time=0.3)

        for step in range(10):
            grad = 2 * x_val
            new_x = x_val - lr * grad

            # Gradient arrow
            arrow_start = axes.c2p(x_val, x_val**2)
            arrow_end = axes.c2p(x_val - 0.5 * np.sign(grad), x_val**2)
            grad_arrow = Arrow(arrow_start, arrow_end, color=ORANGE, buff=0,
                               stroke_width=3, max_tip_length_to_length_ratio=0.3)

            # Trace line
            trace = Line(
                axes.c2p(x_val, x_val**2),
                axes.c2p(new_x, new_x**2),
                color=RED_A, stroke_width=2,
            )

            x_val = new_x
            new_dot_pos = axes.c2p(x_val, x_val**2)
            new_info = Text(f"x={x_val:.2f}, loss={x_val**2:.2f}", font_size=20, color=BLACK)
            new_info.to_corner(UR)

            anim_time = max(0.15, 0.4 - step * 0.025)
            self.play(ShowCreation(grad_arrow), run_time=0.15)
            self.play(
                dot.animate.move_to(new_dot_pos),
                ShowCreation(trace),
                Transform(info, new_info),
                FadeOut(grad_arrow),
                run_time=anim_time,
            )

        final = Text("Minimum reached!", font_size=24, color=GREEN_E)
        final.next_to(axes, DOWN)
        self.play(Write(final), run_time=0.5)
        self.wait(0.5)
