from manim import *


class NetworkStack(Scene):
    """DNS/TCP/TLS handshake flow between client and server."""

    def construct(self):
        self.camera.background_color = "#1a1a2e"

        title = Text("What Happens When You Type a URL", font_size=26, color=WHITE)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title), run_time=0.6)

        # Two columns: Client and Server
        client_x = -3.5
        server_x = 3.5
        col_top = 2.7

        client_box = RoundedRectangle(
            width=1.8, height=0.5,
            corner_radius=0.1,
            fill_color="#45b7d1", fill_opacity=0.2,
            stroke_color="#45b7d1", stroke_width=2,
        )
        client_box.move_to(RIGHT * client_x + UP * col_top)
        client_text = Text("Client", font_size=16, color="#45b7d1")
        client_text.move_to(client_box.get_center())

        server_box = RoundedRectangle(
            width=1.8, height=0.5,
            corner_radius=0.1,
            fill_color="#96ceb4", fill_opacity=0.2,
            stroke_color="#96ceb4", stroke_width=2,
        )
        server_box.move_to(RIGHT * server_x + UP * col_top)
        server_text = Text("Server", font_size=16, color="#96ceb4")
        server_text.move_to(server_box.get_center())

        self.play(
            FadeIn(client_box), FadeIn(client_text),
            FadeIn(server_box), FadeIn(server_text),
            run_time=0.4,
        )

        # Vertical timeline lines
        client_line = DashedLine(
            start=RIGHT * client_x + UP * (col_top - 0.3),
            end=RIGHT * client_x + DOWN * 3.6,
            color=GREY, stroke_width=1, dash_length=0.08,
        )
        server_line = DashedLine(
            start=RIGHT * server_x + UP * (col_top - 0.3),
            end=RIGHT * server_x + DOWN * 3.6,
            color=GREY, stroke_width=1, dash_length=0.08,
        )
        self.play(Create(client_line), Create(server_line), run_time=0.3)

        # Define message sequence - tighter spacing to fit in frame
        y_start = col_top - 0.6
        step = 0.48

        messages = [
            (y_start - 0 * step, "DNS query", "right", "#ffd93d", "DNS Resolution"),
            (y_start - 1 * step, "IP address", "left", "#ffd93d", None),
            (y_start - 2 * step, "SYN", "right", "#4ecdc4", "TCP Handshake"),
            (y_start - 3 * step, "SYN-ACK", "left", "#4ecdc4", None),
            (y_start - 4 * step, "ACK", "right", "#4ecdc4", None),
            (y_start - 5 * step, "ClientHello", "right", "#a8e6cf", "TLS Handshake"),
            (y_start - 6 * step, "ServerHello + Cert", "left", "#a8e6cf", None),
            (y_start - 7 * step, "Key Exchange", "right", "#a8e6cf", None),
            (y_start - 8 * step, "GET /index.html", "right", "#ff6b6b", "HTTP Request"),
            (y_start - 9 * step, "200 OK + HTML", "left", "#ff6b6b", None),
        ]

        for y, label_text, direction, color, phase in messages:
            # Phase label on the left margin
            if phase is not None:
                phase_label = Text(phase, font_size=11, color=color)
                phase_label.move_to(LEFT * 6.2 + UP * y)

                bracket = Line(
                    start=LEFT * 5.2 + UP * (y + 0.12),
                    end=LEFT * 5.2 + UP * (y - 0.12),
                    color=color, stroke_width=1.5,
                )
                self.play(FadeIn(phase_label), Create(bracket), run_time=0.12)

            if direction == "right":
                arrow = Arrow(
                    start=RIGHT * (client_x + 0.3) + UP * y,
                    end=RIGHT * (server_x - 0.3) + UP * y,
                    color=color, stroke_width=2, buff=0,
                    max_tip_length_to_length_ratio=0.05,
                )
            else:
                arrow = Arrow(
                    start=RIGHT * (server_x - 0.3) + UP * y,
                    end=RIGHT * (client_x + 0.3) + UP * y,
                    color=color, stroke_width=2, buff=0,
                    max_tip_length_to_length_ratio=0.05,
                )

            msg_label = Text(label_text, font_size=12, color=WHITE)
            msg_label.next_to(arrow, UP, buff=0.03)

            self.play(GrowArrow(arrow), FadeIn(msg_label), run_time=0.2)

        self.wait(0.3)

        # Render step - positioned to stay within frame
        render_y = y_start - 10 * step
        render_box = RoundedRectangle(
            width=2.2, height=0.4,
            corner_radius=0.1,
            fill_color="#ffd93d", fill_opacity=0.15,
            stroke_color="#ffd93d", stroke_width=2,
        )
        render_box.move_to(RIGHT * client_x + UP * render_y)
        render_text = Text("Browser Renders", font_size=13, color="#ffd93d")
        render_text.move_to(render_box.get_center())

        self.play(FadeIn(render_box), FadeIn(render_text), run_time=0.4)

        self.wait(2.0)
