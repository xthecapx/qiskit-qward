"""
DSR (Differential Success Rate) — 3Blue1Brown-Style Visualization
=================================================================
ManimCE animations explaining the DSR metric.

Scenes:
    1. DSRIntuition       — title + circuit + ideal vs noisy histograms
    2. DSRHistogramSetup   — concrete example with labeled components
    3. DSREdgeCases        — DSR=1, DSR=0 (uniform), DSR=0 (wrong peak)
    4. DSRDegradation      — animated noise sweep with synced histogram + curve

Render individual scenes:
    uv run manim -pql qward/examples/papers/dsr_visualization.py DSRIntuition
Render all scenes:
    uv run manim -pql -a qward/examples/papers/dsr_visualization.py
High quality:
    uv run manim -pqh qward/examples/papers/dsr_visualization.py DSRDegradation
"""

from manim import *
import numpy as np

# ── Shared constants ──────────────────────────────────────────────
BG_COLOR = "#1C1C2E"
EXPECTED_COLOR = GREEN
COMPETITOR_COLOR = RED_C
OTHER_COLOR = GREY_B
FORMULA_COLOR = YELLOW
DSR_COLOR = GOLD
HF_COLOR = TEAL_B
TVDF_COLOR = MAROON_B
BAR_NAMES = ["00", "01", "10", "11"]


# ══════════════════════════════════════════════════════════════════
# Scene 1: Intuition — pose the question visually
# ══════════════════════════════════════════════════════════════════
class DSRIntuition(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Title card ────────────────────────────────────────────
        title = Text("Differential Success Rate", font_size=52, color=DSR_COLOR)
        subtitle = Text(
            "Measuring Quantum Algorithm Quality",
            font_size=28,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Simplified quantum circuit ────────────────────────────
        wire1 = Line(LEFT * 3, RIGHT * 3, color=WHITE).shift(UP * 0.5)
        wire2 = Line(LEFT * 3, RIGHT * 3, color=WHITE).shift(DOWN * 0.5)
        gate1 = Square(side_length=0.6, color=BLUE_C, fill_opacity=0.3).move_to(
            LEFT * 1.5 + UP * 0.5
        )
        gate1_label = MathTex("H", font_size=28).move_to(gate1)
        gate2 = Square(side_length=0.6, color=BLUE_C, fill_opacity=0.3).move_to(
            LEFT * 1.5 + DOWN * 0.5
        )
        gate2_label = MathTex("H", font_size=28).move_to(gate2)
        # CNOT-like connection
        cnot_ctrl = Dot(RIGHT * 0 + UP * 0.5, radius=0.08, color=WHITE)
        cnot_line = Line(RIGHT * 0 + UP * 0.5, RIGHT * 0 + DOWN * 0.5, color=WHITE)
        cnot_targ = Circle(radius=0.15, color=WHITE).move_to(RIGHT * 0 + DOWN * 0.5)
        # Measurement boxes
        meas1 = VGroup(
            Square(side_length=0.5, color=YELLOW, fill_opacity=0.2),
            Arc(radius=0.15, angle=PI, color=YELLOW).shift(DOWN * 0.05),
            Line(ORIGIN, UP * 0.15 + RIGHT * 0.1, color=YELLOW).shift(DOWN * 0.05),
        ).move_to(RIGHT * 2 + UP * 0.5)
        meas2 = meas1.copy().move_to(RIGHT * 2 + DOWN * 0.5)

        circuit = VGroup(
            wire1,
            wire2,
            gate1,
            gate1_label,
            gate2,
            gate2_label,
            cnot_ctrl,
            cnot_line,
            cnot_targ,
            meas1,
            meas2,
        ).shift(UP * 2)
        circuit_label = Text("Quantum Circuit", font_size=24, color=GREY_B).next_to(
            circuit, UP, buff=0.3
        )
        self.play(
            LaggedStart(*[Create(m) for m in circuit], lag_ratio=0.08),
            FadeIn(circuit_label),
            run_time=2,
        )
        self.wait(0.5)

        # ── Two contrasting histograms ────────────────────────────
        ideal_chart = BarChart(
            values=[0.03, 0.90, 0.04, 0.03],
            bar_names=BAR_NAMES,
            y_range=[0, 1, 0.2],
            x_length=4,
            y_length=2.5,
            bar_colors=[OTHER_COLOR, EXPECTED_COLOR, OTHER_COLOR, OTHER_COLOR],
        ).shift(LEFT * 3 + DOWN * 1.5)
        ideal_label = Text("Ideal", font_size=24, color=EXPECTED_COLOR).next_to(
            ideal_chart, UP, buff=0.2
        )

        noisy_chart = BarChart(
            values=[0.20, 0.40, 0.20, 0.20],
            bar_names=BAR_NAMES,
            y_range=[0, 1, 0.2],
            x_length=4,
            y_length=2.5,
            bar_colors=[OTHER_COLOR, EXPECTED_COLOR, OTHER_COLOR, OTHER_COLOR],
        ).shift(RIGHT * 3 + DOWN * 1.5)
        noisy_label = Text("Noisy QPU", font_size=24, color=COMPETITOR_COLOR).next_to(
            noisy_chart, UP, buff=0.2
        )

        arrow_between = Arrow(
            ideal_chart.get_right() + RIGHT * 0.1,
            noisy_chart.get_left() + LEFT * 0.1,
            color=GREY_B,
            stroke_width=2,
        )
        noise_text = Text("noise", font_size=18, color=GREY_B).next_to(arrow_between, UP, buff=0.1)

        self.play(
            FadeIn(ideal_chart, shift=UP * 0.3),
            Write(ideal_label),
            run_time=1,
        )
        self.play(
            GrowArrow(arrow_between),
            FadeIn(noise_text),
            FadeIn(noisy_chart, shift=UP * 0.3),
            Write(noisy_label),
            run_time=1,
        )
        self.wait(1)

        # ── Question ──────────────────────────────────────────────
        question = Text(
            "How much does the expected outcome stand out?",
            font_size=30,
            color=FORMULA_COLOR,
            slant=ITALIC,
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(question), run_time=1.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════
# Scene 2: Histogram Setup — label all DSR components
# ══════════════════════════════════════════════════════════════════
class DSRHistogramSetup(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        section_title = Text("Identifying the Components", font_size=36, color=DSR_COLOR).to_edge(
            UP, buff=0.4
        )
        self.play(Write(section_title))

        # ── Bar chart (narrower, shifted left to leave room for labels) ─
        chart = BarChart(
            values=[0.2, 0.4, 0.2, 0.2],
            bar_names=BAR_NAMES,
            y_range=[0, 0.5, 0.1],
            x_length=6,
            y_length=4,
            bar_colors=[BLUE_D] * 4,
        ).shift(DOWN * 0.3 + LEFT * 1.5)
        self.play(Create(chart), run_time=1.5)
        self.wait(0.5)

        # ── Count labels above bars ───────────────────────────────
        counts = [20, 40, 20, 20]
        count_labels = VGroup()
        for i, c in enumerate(counts):
            label = Text(str(c), font_size=22, color=WHITE).next_to(chart.bars[i], UP, buff=0.1)
            count_labels.add(label)
        self.play(LaggedStart(*[FadeIn(l, shift=DOWN * 0.1) for l in count_labels], lag_ratio=0.15))
        self.wait(0.5)

        # ── Highlight expected outcome (bar index 1 = "01") ──────
        expected_rect = SurroundingRectangle(chart.bars[1], color=EXPECTED_COLOR, buff=0.05)
        self.play(Create(expected_rect))
        self.play(chart.bars[1].animate.set_fill(EXPECTED_COLOR, opacity=0.8))

        exp_label = MathTex(
            r"\text{Expected: } |01\rangle", font_size=30, color=EXPECTED_COLOR
        ).next_to(chart.bars[1], UP, buff=0.6)
        exp_arrow = Arrow(
            exp_label.get_bottom(),
            chart.bars[1].get_top() + UP * 0.1,
            color=EXPECTED_COLOR,
            stroke_width=2,
            buff=0.05,
        )
        self.play(Write(exp_label), GrowArrow(exp_arrow))
        self.wait(0.5)

        # ── Highlight competitor (bar index 0 = "00", tied) ──────
        self.play(
            chart.bars[0].animate.set_fill(COMPETITOR_COLOR, opacity=0.7),
            chart.bars[2].animate.set_fill(OTHER_COLOR, opacity=0.4),
            chart.bars[3].animate.set_fill(OTHER_COLOR, opacity=0.4),
        )
        comp_label = MathTex(
            r"p_{\text{comp}} = 0.2", font_size=28, color=COMPETITOR_COLOR
        ).next_to(chart.bars[0], LEFT, buff=0.8)
        comp_arrow = Arrow(
            comp_label.get_right(),
            chart.bars[0].get_left() + LEFT * 0.05,
            color=COMPETITOR_COLOR,
            stroke_width=2,
            buff=0.05,
        )
        self.play(Write(comp_label), GrowArrow(comp_arrow))
        self.wait(0.5)
        self.play(FadeOut(expected_rect))

        # ── Dashed reference lines (stop at chart edge) ──────────
        y_exp = chart.bars[1].get_top()[1]
        y_comp = chart.bars[0].get_top()[1]
        x_left = chart.get_left()[0] - 0.3
        x_right = chart.get_right()[0] + 0.2

        dash_exp = DashedLine(
            [x_left, y_exp, 0],
            [x_right, y_exp, 0],
            color=EXPECTED_COLOR,
            stroke_width=2,
            dash_length=0.1,
        )
        dash_comp = DashedLine(
            [x_left, y_comp, 0],
            [x_right, y_comp, 0],
            color=COMPETITOR_COLOR,
            stroke_width=2,
            dash_length=0.1,
        )

        self.play(Create(dash_exp), Create(dash_comp))

        # ── Brace between the two dashed lines ───────────────────
        brace_x = x_right + 0.3
        brace = Brace(
            Line([brace_x, y_comp, 0], [brace_x, y_exp, 0]),
            direction=RIGHT,
            color=DSR_COLOR,
        )

        # ── Labels aligned to the right of the brace ─────────────
        label_x = brace.get_right()[0] + 0.25
        pexp_label = MathTex(
            r"\bar{p}_{\text{exp}} = 0.4", font_size=22, color=EXPECTED_COLOR
        ).move_to([label_x, y_exp, 0], aligned_edge=LEFT)
        pcomp_label = MathTex(
            r"p_{\text{comp}} = 0.2", font_size=22, color=COMPETITOR_COLOR
        ).move_to([label_x, y_comp, 0], aligned_edge=LEFT)
        gap_label = (
            MathTex(r"\text{Gap} = 0.2", font_size=24, color=DSR_COLOR)
            .next_to(brace, RIGHT, buff=0.15)
            .move_to([label_x, (y_exp + y_comp) / 2, 0], aligned_edge=LEFT)
        )

        self.play(GrowFromCenter(brace))
        self.play(FadeIn(pexp_label), FadeIn(pcomp_label), Write(gap_label))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════
# Scene 3: Edge Cases — DSR = 1, 0 (uniform), 0 (wrong peak)
# ══════════════════════════════════════════════════════════════════
class DSREdgeCases(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        section_title = Text("Edge Cases", font_size=36, color=DSR_COLOR).to_edge(UP, buff=0.4)
        self.play(Write(section_title))

        chart_kwargs = dict(
            bar_names=BAR_NAMES,
            y_range=[0, 1.1, 0.25],
            x_length=3.2,
            y_length=2.8,
        )

        # ── Perfect: DSR = 1 ─────────────────────────────────────
        c1 = BarChart(
            values=[0, 1.0, 0, 0],
            bar_colors=[OTHER_COLOR, EXPECTED_COLOR, OTHER_COLOR, OTHER_COLOR],
            **chart_kwargs,
        )
        l1 = MathTex(r"\text{DSR} = 1.0", font_size=28, color=EXPECTED_COLOR).next_to(
            c1, UP, buff=0.2
        )
        t1 = Text("Perfect", font_size=20, color=EXPECTED_COLOR).next_to(l1, UP, buff=0.1)
        g1 = VGroup(t1, l1, c1)

        # ── Uniform: DSR = 0 ─────────────────────────────────────
        c2 = BarChart(
            values=[0.25, 0.25, 0.25, 0.25],
            bar_colors=[OTHER_COLOR, EXPECTED_COLOR, OTHER_COLOR, OTHER_COLOR],
            **chart_kwargs,
        )
        l2 = MathTex(r"\text{DSR} = 0.0", font_size=28, color=COMPETITOR_COLOR).next_to(
            c2, UP, buff=0.2
        )
        t2 = Text("Uniform noise", font_size=20, color=COMPETITOR_COLOR).next_to(l2, UP, buff=0.1)
        g2 = VGroup(t2, l2, c2)

        # ── Wrong peak: DSR = 0 ──────────────────────────────────
        c3 = BarChart(
            values=[0.6, 0.1, 0.2, 0.1],
            bar_colors=[COMPETITOR_COLOR, EXPECTED_COLOR, OTHER_COLOR, OTHER_COLOR],
            **chart_kwargs,
        )
        l3 = MathTex(r"\text{DSR} = 0.0", font_size=28, color=COMPETITOR_COLOR).next_to(
            c3, UP, buff=0.2
        )
        t3 = Text("Wrong peak", font_size=20, color=COMPETITOR_COLOR).next_to(l3, UP, buff=0.1)
        g3 = VGroup(t3, l3, c3)

        # ── Arrange side by side ──────────────────────────────────
        charts_group = VGroup(g1, g2, g3).arrange(RIGHT, buff=0.8).shift(UP * 0.3)
        self.play(
            LaggedStart(
                FadeIn(g1, shift=UP * 0.3),
                FadeIn(g2, shift=UP * 0.3),
                FadeIn(g3, shift=UP * 0.3),
                lag_ratio=0.3,
            ),
            run_time=2,
        )
        self.wait(1.5)

        # ── Number line showing DSR scale ─────────────────────────
        nline = NumberLine(
            x_range=[0, 1, 0.25],
            length=10,
            include_numbers=True,
            color=WHITE,
            font_size=24,
        ).to_edge(DOWN, buff=1.0)
        nline_label_left = Text("no contrast", font_size=18, color=COMPETITOR_COLOR).next_to(
            nline.n2p(0), DOWN, buff=0.25
        )
        nline_label_right = Text("perfect", font_size=18, color=EXPECTED_COLOR).next_to(
            nline.n2p(1), DOWN, buff=0.25
        )

        self.play(Create(nline), FadeIn(nline_label_left), FadeIn(nline_label_right))

        # Dots
        dot0 = Dot(nline.n2p(0), radius=0.1, color=COMPETITOR_COLOR)
        dot033 = Dot(nline.n2p(0.333), radius=0.1, color=DSR_COLOR)
        dot1 = Dot(nline.n2p(1), radius=0.1, color=EXPECTED_COLOR)
        label033 = MathTex("0.333", font_size=22, color=DSR_COLOR).next_to(dot033, UP, buff=0.15)
        label_our = Text("Our example", font_size=16, color=DSR_COLOR).next_to(
            label033, UP, buff=0.1
        )

        self.play(
            GrowFromCenter(dot0),
            GrowFromCenter(dot1),
            GrowFromCenter(dot033),
            FadeIn(label033),
            FadeIn(label_our),
        )
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════
# Scene 4: Degradation — animated noise sweep
# ══════════════════════════════════════════════════════════════════
class DSRDegradation(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        section_title = Text("DSR Degradation Under Noise", font_size=36, color=DSR_COLOR).to_edge(
            UP, buff=0.4
        )
        self.play(Write(section_title))

        tracker = ValueTracker(0)

        # ── Helper: probability as function of noise t ────────────
        def p_expected(t):
            return 1.0 - 0.75 * t

        def p_other(t):
            return 0.25 * t

        def dsr_func(t):
            pe = p_expected(t)
            pc = p_other(t)
            if pe + pc == 0:
                return 0
            return max(0, min(1, (pe - pc) / (pe + pc)))

        # ── LEFT: animated histogram bars ─────────────────────────
        hist_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.1, 0.25],
            x_length=4,
            y_length=3.5,
            axis_config={"include_ticks": True, "font_size": 20},
        ).shift(LEFT * 3.2 + DOWN * 0.3)

        # Bar name labels
        bar_labels = VGroup()
        for i, name in enumerate(BAR_NAMES):
            label = Text(name, font_size=18, color=WHITE).move_to(
                hist_axes.c2p(i + 0.5, 0) + DOWN * 0.3
            )
            bar_labels.add(label)

        bar_width = 0.7

        def make_bar(index, prob_func, color):
            return always_redraw(
                lambda idx=index, pf=prob_func, c=color: Rectangle(
                    width=bar_width,
                    height=max(0.01, hist_axes.y_axis.unit_size * pf(tracker.get_value())),
                    fill_color=c,
                    fill_opacity=0.8,
                    stroke_color=c,
                    stroke_width=1,
                ).move_to(hist_axes.c2p(idx + 0.5, 0), aligned_edge=DOWN)
            )

        bar0 = make_bar(0, p_other, OTHER_COLOR)
        bar1 = make_bar(1, p_expected, EXPECTED_COLOR)
        bar2 = make_bar(2, p_other, OTHER_COLOR)
        bar3 = make_bar(3, p_other, OTHER_COLOR)

        hist_label = Text("Histogram", font_size=22, color=GREY_B).next_to(hist_axes, UP, buff=0.1)

        # ── RIGHT: DSR curve ──────────────────────────────────────
        curve_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1.1, 0.25],
            x_length=4,
            y_length=3.5,
            axis_config={"include_ticks": True, "include_numbers": True, "font_size": 18},
        ).shift(RIGHT * 3.2 + DOWN * 0.3)

        x_label = Text("Noise level", font_size=20, color=GREY_B).next_to(
            curve_axes.x_axis, DOWN, buff=0.35
        )

        # Plot DSR(t) curve
        dsr_curve = curve_axes.plot(
            lambda t: dsr_func(t),
            x_range=[0, 1, 0.01],
            color=DSR_COLOR,
            stroke_width=3,
        )

        # Moving dot on curve
        moving_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), dsr_func(tracker.get_value())),
                radius=0.08,
                color=WHITE,
            )
        )

        # Vertical dashed line from dot to x-axis
        vert_dash = always_redraw(
            lambda: DashedLine(
                curve_axes.c2p(tracker.get_value(), 0),
                curve_axes.c2p(tracker.get_value(), dsr_func(tracker.get_value())),
                color=GREY_B,
                stroke_width=1,
                dash_length=0.08,
            )
        )

        # DSR value label
        dsr_value_label = always_redraw(
            lambda: MathTex(
                rf"\text{{DSR}} = {dsr_func(tracker.get_value()):.2f}",
                font_size=28,
                color=DSR_COLOR,
            ).next_to(curve_axes, UP, buff=0.15)
        )

        # ── Assemble and animate ──────────────────────────────────
        self.play(
            Create(hist_axes),
            Create(curve_axes),
            FadeIn(bar_labels),
            FadeIn(hist_label),
            FadeIn(x_label),
            run_time=1.5,
        )
        self.add(bar0, bar1, bar2, bar3)
        self.play(Create(dsr_curve), run_time=1.5)
        self.add(moving_dot, vert_dash, dsr_value_label)
        self.wait(0.5)

        # Key point labels on curve
        perfect_label = Text("Perfect", font_size=20, color=EXPECTED_COLOR).next_to(
            curve_axes.c2p(0, 1), UL, buff=0.15
        )
        random_label = Text("Random", font_size=20, color=COMPETITOR_COLOR).next_to(
            curve_axes.c2p(1, 0), DR, buff=0.15
        )
        self.play(FadeIn(perfect_label), FadeIn(random_label))

        # ── Main sweep animation ──────────────────────────────────
        self.play(
            tracker.animate.set_value(1),
            run_time=6,
            rate_func=linear,
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════
# Scene 5: Combined Degradation — DSR vs HF vs TVDF
# ══════════════════════════════════════════════════════════════════
class CombinedDegradation(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        section_title = Text("Metric Comparison Under Noise", font_size=36, color=WHITE).to_edge(
            UP, buff=0.4
        )
        self.play(Write(section_title))

        tracker = ValueTracker(0)

        # ── Probability functions (same noise model) ──────────────
        def p_expected(t):
            return 1.0 - 0.75 * t

        def p_other(t):
            return 0.25 * t

        def dsr_func(t):
            pe, pc = p_expected(t), p_other(t)
            if pe + pc == 0:
                return 0
            return max(0, min(1, (pe - pc) / (pe + pc)))

        def hf_func(t):
            return max(0, p_expected(t))

        def tvdf_func(t):
            return max(0, 1 - 0.75 * t)

        # ── LEFT: animated histogram bars ─────────────────────────
        hist_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.1, 0.25],
            x_length=4,
            y_length=3.2,
            axis_config={"include_ticks": True, "font_size": 20},
        ).shift(LEFT * 3.2 + DOWN * 0.5)

        bar_labels = VGroup()
        for i, name in enumerate(BAR_NAMES):
            label = Text(name, font_size=18, color=WHITE).move_to(
                hist_axes.c2p(i + 0.5, 0) + DOWN * 0.3
            )
            bar_labels.add(label)

        bar_width = 0.7

        def make_bar(index, prob_func, color):
            return always_redraw(
                lambda idx=index, pf=prob_func, c=color: Rectangle(
                    width=bar_width,
                    height=max(0.01, hist_axes.y_axis.unit_size * pf(tracker.get_value())),
                    fill_color=c,
                    fill_opacity=0.8,
                    stroke_color=c,
                    stroke_width=1,
                ).move_to(hist_axes.c2p(idx + 0.5, 0), aligned_edge=DOWN)
            )

        bar0 = make_bar(0, p_other, OTHER_COLOR)
        bar1 = make_bar(1, p_expected, EXPECTED_COLOR)
        bar2 = make_bar(2, p_other, OTHER_COLOR)
        bar3 = make_bar(3, p_other, OTHER_COLOR)

        hist_label = Text("Measured", font_size=22, color=GREY_B).next_to(hist_axes, UP, buff=0.1)

        # ── RIGHT: combined curve plot ────────────────────────────
        curve_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1.1, 0.25],
            x_length=5,
            y_length=3.2,
            axis_config={"include_ticks": True, "include_numbers": True, "font_size": 18},
        ).shift(RIGHT * 2.8 + DOWN * 0.5)

        x_label = Text("Noise level", font_size=20, color=GREY_B).next_to(
            curve_axes.x_axis, DOWN, buff=0.35
        )

        # Plot all three curves
        dsr_curve = curve_axes.plot(
            dsr_func,
            x_range=[0, 1, 0.01],
            color=DSR_COLOR,
            stroke_width=3,
        )
        hf_curve = curve_axes.plot(
            hf_func,
            x_range=[0, 1, 0.01],
            color=HF_COLOR,
            stroke_width=3,
        )
        tvdf_curve = curve_axes.plot(
            tvdf_func,
            x_range=[0, 1, 0.01],
            color=TVDF_COLOR,
            stroke_width=3,
        )

        # Moving dots on each curve
        dsr_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), dsr_func(tracker.get_value())),
                radius=0.07,
                color=DSR_COLOR,
            )
        )
        hf_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), hf_func(tracker.get_value())),
                radius=0.07,
                color=HF_COLOR,
            )
        )
        tvdf_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), tvdf_func(tracker.get_value())),
                radius=0.07,
                color=TVDF_COLOR,
            )
        )

        # ── Legend ────────────────────────────────────────────────
        legend = (
            VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=DSR_COLOR, stroke_width=3),
                    Text("DSR", font_size=18, color=DSR_COLOR),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=HF_COLOR, stroke_width=3),
                    Text("HF", font_size=18, color=HF_COLOR),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=TVDF_COLOR, stroke_width=3),
                    Text("TVDF", font_size=18, color=TVDF_COLOR),
                ).arrange(RIGHT, buff=0.15),
            )
            .arrange(RIGHT, buff=0.6)
            .next_to(curve_axes, UP, buff=0.15)
        )

        # Key point labels
        perfect_label = Text("Perfect", font_size=20, color=EXPECTED_COLOR).next_to(
            curve_axes.c2p(0, 1), UL, buff=0.15
        )
        random_label = Text("Random", font_size=20, color=COMPETITOR_COLOR).next_to(
            curve_axes.c2p(1, 0), DR, buff=0.15
        )

        # ── Assemble ──────────────────────────────────────────────
        self.play(
            Create(hist_axes),
            Create(curve_axes),
            FadeIn(bar_labels),
            FadeIn(hist_label),
            FadeIn(x_label),
            run_time=1.5,
        )
        self.add(bar0, bar1, bar2, bar3)

        # Draw curves one by one
        self.play(Create(dsr_curve), run_time=1)
        self.play(Create(hf_curve), run_time=1)
        self.play(Create(tvdf_curve), run_time=1)
        self.play(FadeIn(legend), FadeIn(perfect_label), FadeIn(random_label))

        self.add(dsr_dot, hf_dot, tvdf_dot)
        self.wait(0.5)

        # ── Main sweep animation ──────────────────────────────────
        self.play(
            tracker.animate.set_value(1),
            run_time=8,
            rate_func=linear,
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════
# Scene 6: Combined Degradation — Multi-State (3 qubits, 3 marked)
# ══════════════════════════════════════════════════════════════════
class CombinedDegradationMultiState(Scene):
    """3 qubits (8 states), 3 marked: {001, 010, 100}.
    Ideal: 1/3 on each marked, 0 elsewhere.
    Noise: uniform depolarization toward 1/8 per state.

    DSR(t)  = 4(1-t)/(4-t)          non-linear, → 0
    HF(t)   = 1 - 5t/8              linear, → 0.375
    TVDF(t) = 1 - 5t/8              linear, → 0.375  (same as HF for
                                     uniform-over-marked ideal)
    """

    def construct(self):
        self.camera.background_color = BG_COLOR

        section_title = Text(
            "Metric Comparison — 3 Qubits, 3 Marked States",
            font_size=32,
            color=WHITE,
        ).to_edge(UP, buff=0.4)
        self.play(Write(section_title))

        tracker = ValueTracker(0)

        # ── Constants ─────────────────────────────────────────────
        n_qubits = 3
        n_states = 2**n_qubits  # 8
        n_marked = 3
        p_ideal_marked = 1.0 / n_marked  # 1/3
        p_uniform = 1.0 / n_states  # 1/8
        bar_names_3q = [f"{i:0{n_qubits}b}" for i in range(n_states)]
        marked_indices = {1, 2, 4}  # 001, 010, 100

        # ── Probability functions ─────────────────────────────────
        def p_marked(t):
            return p_ideal_marked * (1 - t) + p_uniform * t

        def p_other(t):
            return p_uniform * t

        def dsr_func(t):
            # p̄_exp = p_marked(t), p_comp = p_other(t)
            pe = p_marked(t)
            pc = p_other(t)
            if pe + pc == 0:
                return 0
            return max(0, min(1, (pe - pc) / (pe + pc)))

        def hf_func(t):
            # HF = (3·√(1/3 · p_marked(t)))² = 3·p_marked(t) = 1 - 5t/8
            return max(0, n_marked * p_marked(t))

        def tvdf_func(t):
            # TVD = 0.5·(3·|1/3-pm| + 5·|0-po|) = 5t/8
            tvd = 0.5 * (
                n_marked * abs(p_ideal_marked - p_marked(t)) + (n_states - n_marked) * p_other(t)
            )
            return max(0, 1 - tvd)

        # ── LEFT: animated histogram (8 bars) ────────────────────
        hist_axes = Axes(
            x_range=[0, n_states, 1],
            y_range=[0, 0.4, 0.1],
            x_length=5,
            y_length=3,
            axis_config={"include_ticks": True, "font_size": 16},
        ).shift(LEFT * 3 + DOWN * 0.5)

        bar_labels = VGroup()
        for i, name in enumerate(bar_names_3q):
            label = (
                Text(name, font_size=12, color=WHITE)
                .move_to(hist_axes.c2p(i + 0.5, 0) + DOWN * 0.25)
                .rotate(45 * DEGREES)
            )
            bar_labels.add(label)

        bar_width = 0.45

        def make_bar(index):
            is_marked = index in marked_indices
            prob_func = p_marked if is_marked else p_other
            color = EXPECTED_COLOR if is_marked else OTHER_COLOR
            return always_redraw(
                lambda idx=index, pf=prob_func, c=color: Rectangle(
                    width=bar_width,
                    height=max(0.01, hist_axes.y_axis.unit_size * pf(tracker.get_value())),
                    fill_color=c,
                    fill_opacity=0.8,
                    stroke_color=c,
                    stroke_width=1,
                ).move_to(hist_axes.c2p(idx + 0.5, 0), aligned_edge=DOWN)
            )

        bars = [make_bar(i) for i in range(n_states)]

        hist_label = Text("Measured", font_size=20, color=GREY_B).next_to(hist_axes, UP, buff=0.1)

        # ── RIGHT: combined curve plot ────────────────────────────
        curve_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1.1, 0.25],
            x_length=5,
            y_length=3,
            axis_config={"include_ticks": True, "include_numbers": True, "font_size": 18},
        ).shift(RIGHT * 3 + DOWN * 0.5)

        x_label = Text("Noise level", font_size=20, color=GREY_B).next_to(
            curve_axes.x_axis, DOWN, buff=0.35
        )

        # Plot all three curves
        dsr_curve = curve_axes.plot(
            dsr_func,
            x_range=[0, 1, 0.01],
            color=DSR_COLOR,
            stroke_width=3,
        )
        hf_curve = curve_axes.plot(
            hf_func,
            x_range=[0, 1, 0.01],
            color=HF_COLOR,
            stroke_width=3,
        )
        tvdf_curve = curve_axes.plot(
            tvdf_func,
            x_range=[0, 1, 0.01],
            color=TVDF_COLOR,
            stroke_width=3,
            stroke_opacity=0.6,
        )

        # Moving dots
        dsr_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), dsr_func(tracker.get_value())),
                radius=0.07,
                color=DSR_COLOR,
            )
        )
        hf_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), hf_func(tracker.get_value())),
                radius=0.07,
                color=HF_COLOR,
            )
        )
        tvdf_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), tvdf_func(tracker.get_value())),
                radius=0.07,
                color=TVDF_COLOR,
            )
        )

        # ── Legend ────────────────────────────────────────────────
        legend = (
            VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=DSR_COLOR, stroke_width=3),
                    Text("DSR", font_size=18, color=DSR_COLOR),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=HF_COLOR, stroke_width=3),
                    Text("HF", font_size=18, color=HF_COLOR),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=TVDF_COLOR, stroke_width=3),
                    Text("TVDF", font_size=18, color=TVDF_COLOR),
                ).arrange(RIGHT, buff=0.15),
            )
            .arrange(RIGHT, buff=0.6)
            .next_to(curve_axes, UP, buff=0.15)
        )

        # Key point labels
        perfect_label = Text("Perfect", font_size=20, color=EXPECTED_COLOR).next_to(
            curve_axes.c2p(0, 1), UL, buff=0.15
        )
        # HF/TVDF bottom at 0.375 — label the floor
        floor_val = n_marked * p_uniform  # 3/8 = 0.375
        floor_dash = DashedLine(
            curve_axes.c2p(0, floor_val),
            curve_axes.c2p(1, floor_val),
            color=GREY_B,
            stroke_width=1,
            dash_length=0.08,
        )
        floor_label = MathTex(
            r"0.375",
            font_size=20,
            color=GREY_B,
        ).next_to(curve_axes.c2p(0, floor_val), LEFT, buff=0.15)

        # ── Assemble ──────────────────────────────────────────────
        self.play(
            Create(hist_axes),
            Create(curve_axes),
            FadeIn(bar_labels),
            FadeIn(hist_label),
            FadeIn(x_label),
            run_time=1.5,
        )
        for b in bars:
            self.add(b)

        # Draw curves one by one
        self.play(Create(dsr_curve), run_time=1)
        self.play(Create(hf_curve), run_time=1)
        self.play(Create(tvdf_curve), run_time=1)
        self.play(
            FadeIn(legend),
            FadeIn(perfect_label),
            Create(floor_dash),
            FadeIn(floor_label),
        )

        self.add(dsr_dot, hf_dot, tvdf_dot)
        self.wait(0.5)

        # ── Main sweep animation (MultiState) ────────────────────
        self.play(
            tracker.animate.set_value(1),
            run_time=8,
            rate_func=linear,
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ══════════════════════════════════════════════════════════════════
# Scene 7: Combined Degradation — Spread Distribution (QFT-like)
# ══════════════════════════════════════════════════════════════════
class CombinedDegradationSpread(Scene):
    """3 qubits, QFT-like spread ideal distribution.
    All three metrics are visually distinct:
      - DSR:  non-linear, drops to 0 (threshold behavior)
      - HF:   non-linear, very optimistic (sqrt compresses deviations)
      - TVDF: linear, stricter than HF
    """

    def construct(self):
        self.camera.background_color = BG_COLOR

        section_title = Text(
            "Metric Comparison — Spread Distribution (QFT-like)",
            font_size=30,
            color=WHITE,
        ).to_edge(UP, buff=0.4)
        self.play(Write(section_title))

        tracker = ValueTracker(0)

        # ── QFT-like ideal distribution (8 states) ───────────────
        ideal = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03])
        n_states = len(ideal)
        p_uniform = 1.0 / n_states  # 0.125
        bar_names_3q = [f"{i:03b}" for i in range(n_states)]

        # Expected outcome = state with highest ideal prob (index 0)
        expected_idx = 0

        # ── Probability functions ─────────────────────────────────
        def p_meas(i, t):
            """Measured probability for state i at noise level t."""
            return ideal[i] * (1 - t) + p_uniform * t

        def dsr_func(t):
            p_exp_bar = p_meas(expected_idx, t)
            p_comp = max(p_meas(j, t) for j in range(n_states) if j != expected_idx)
            denom = p_exp_bar + p_comp
            if denom == 0:
                return 0
            return max(0, min(1, (p_exp_bar - p_comp) / denom))

        def hf_func(t):
            bc = sum(np.sqrt(ideal[i] * p_meas(i, t)) for i in range(n_states))
            return float(bc**2)

        def tvdf_func(t):
            tvd = 0.5 * sum(abs(ideal[i] - p_meas(i, t)) for i in range(n_states))
            return max(0.0, 1.0 - tvd)

        # ── LEFT: animated histogram (8 bars) ────────────────────
        hist_axes = Axes(
            x_range=[0, n_states, 1],
            y_range=[0, 0.35, 0.1],
            x_length=5,
            y_length=3,
            axis_config={"include_ticks": True, "font_size": 16},
        ).shift(LEFT * 3 + DOWN * 0.5)

        bar_labels = VGroup()
        for i, name in enumerate(bar_names_3q):
            label = (
                Text(name, font_size=12, color=WHITE)
                .move_to(hist_axes.c2p(i + 0.5, 0) + DOWN * 0.25)
                .rotate(45 * DEGREES)
            )
            bar_labels.add(label)

        bar_width = 0.45

        # Color gradient: highest-prob states brighter
        bar_colors = [
            interpolate_color(OTHER_COLOR, EXPECTED_COLOR, ideal[i] / ideal[0])
            for i in range(n_states)
        ]

        def make_bar(index):
            return always_redraw(
                lambda idx=index: Rectangle(
                    width=bar_width,
                    height=max(0.01, hist_axes.y_axis.unit_size * p_meas(idx, tracker.get_value())),
                    fill_color=bar_colors[idx],
                    fill_opacity=0.8,
                    stroke_color=bar_colors[idx],
                    stroke_width=1,
                ).move_to(hist_axes.c2p(idx + 0.5, 0), aligned_edge=DOWN)
            )

        bars = [make_bar(i) for i in range(n_states)]

        hist_label = Text("Measured", font_size=20, color=GREY_B).next_to(hist_axes, UP, buff=0.1)

        # ── RIGHT: combined curve plot ────────────────────────────
        curve_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1.1, 0.25],
            x_length=5,
            y_length=3,
            axis_config={"include_ticks": True, "include_numbers": True, "font_size": 18},
        ).shift(RIGHT * 3 + DOWN * 0.5)

        x_label = Text("Noise level", font_size=20, color=GREY_B).next_to(
            curve_axes.x_axis, DOWN, buff=0.35
        )

        # Plot all three curves
        dsr_curve = curve_axes.plot(
            dsr_func,
            x_range=[0, 1, 0.01],
            color=DSR_COLOR,
            stroke_width=3,
        )
        hf_curve = curve_axes.plot(
            hf_func,
            x_range=[0, 1, 0.01],
            color=HF_COLOR,
            stroke_width=3,
        )
        tvdf_curve = curve_axes.plot(
            tvdf_func,
            x_range=[0, 1, 0.01],
            color=TVDF_COLOR,
            stroke_width=3,
        )

        # Moving dots
        dsr_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), dsr_func(tracker.get_value())),
                radius=0.07,
                color=DSR_COLOR,
            )
        )
        hf_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), hf_func(tracker.get_value())),
                radius=0.07,
                color=HF_COLOR,
            )
        )
        tvdf_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), tvdf_func(tracker.get_value())),
                radius=0.07,
                color=TVDF_COLOR,
            )
        )

        # ── Legend ────────────────────────────────────────────────
        legend = (
            VGroup(
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=DSR_COLOR, stroke_width=3),
                    Text("DSR", font_size=18, color=DSR_COLOR),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=HF_COLOR, stroke_width=3),
                    Text("HF", font_size=18, color=HF_COLOR),
                ).arrange(RIGHT, buff=0.15),
                VGroup(
                    Line(ORIGIN, RIGHT * 0.4, color=TVDF_COLOR, stroke_width=3),
                    Text("TVDF", font_size=18, color=TVDF_COLOR),
                ).arrange(RIGHT, buff=0.15),
            )
            .arrange(RIGHT, buff=0.6)
            .next_to(curve_axes, UP, buff=0.15)
        )

        # Key point labels
        perfect_label = Text("Perfect", font_size=20, color=EXPECTED_COLOR).next_to(
            curve_axes.c2p(0, 1), UL, buff=0.15
        )

        # ── Assemble ──────────────────────────────────────────────
        self.play(
            Create(hist_axes),
            Create(curve_axes),
            FadeIn(bar_labels),
            FadeIn(hist_label),
            FadeIn(x_label),
            run_time=1.5,
        )
        for b in bars:
            self.add(b)

        # Draw curves one by one
        self.play(Create(dsr_curve), run_time=1)
        self.play(Create(hf_curve), run_time=1)
        self.play(Create(tvdf_curve), run_time=1)
        self.play(FadeIn(legend), FadeIn(perfect_label))

        self.add(dsr_dot, hf_dot, tvdf_dot)
        self.wait(0.5)

        # ── Main sweep animation (Spread) ─────────────────────────
        self.play(
            tracker.animate.set_value(1),
            run_time=8,
            rate_func=linear,
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
