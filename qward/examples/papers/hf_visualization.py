"""
HF (Hellinger Fidelity) — 3Blue1Brown-Style Visualization
==========================================================
ManimCE animation showing Hellinger Fidelity degradation under noise.

Formula:
    HF = (Σ √(p_ideal(x) · p_measured(x)))²

Render:
    uv run manim -pql qward/examples/papers/hf_visualization.py HFDegradation
High quality:
    uv run manim -pqh qward/examples/papers/hf_visualization.py HFDegradation
"""

from manim import *
import numpy as np

# ── Shared constants ──────────────────────────────────────────────
BG_COLOR = "#1C1C2E"
EXPECTED_COLOR = GREEN
COMPETITOR_COLOR = RED_C
OTHER_COLOR = GREY_B
HF_COLOR = TEAL_B
BAR_NAMES = ["00", "01", "10", "11"]


# ══════════════════════════════════════════════════════════════════
# HF Degradation — animated noise sweep
# ══════════════════════════════════════════════════════════════════
class HFDegradation(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        section_title = Text(
            "Hellinger Fidelity Degradation Under Noise",
            font_size=34, color=HF_COLOR,
        ).to_edge(UP, buff=0.4)
        self.play(Write(section_title))

        # ── Formula card ──────────────────────────────────────────
        formula = MathTex(
            r"\text{HF} = \left(\sum_x \sqrt{p_{\text{ideal}}(x)"
            r"\cdot p_{\text{meas}}(x)}\right)^2",
            font_size=32, color=HF_COLOR,
        ).next_to(section_title, DOWN, buff=0.25)
        self.play(Write(formula), run_time=1.5)
        self.wait(0.5)

        tracker = ValueTracker(0)

        # ── Probability functions ─────────────────────────────────
        # Ideal: |01⟩ → p_ideal = [0, 1, 0, 0]
        # Measured at noise t: p_01 = 1 - 0.75t, others = 0.25t
        def p_expected(t):
            return 1.0 - 0.75 * t

        def p_other(t):
            return 0.25 * t

        def hf_func(t):
            # HF = (√(0·p00) + √(1·p01) + √(0·p10) + √(0·p11))²
            #    = (√p01)² = p01 = 1 - 0.75t
            p01 = p_expected(t)
            return max(0, p01)

        # ── LEFT: animated histogram bars ─────────────────────────
        hist_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.1, 0.25],
            x_length=4, y_length=3,
            axis_config={"include_ticks": True, "font_size": 20},
        ).shift(LEFT * 3.2 + DOWN * 0.7)

        bar_labels = VGroup()
        for i, name in enumerate(BAR_NAMES):
            label = Text(name, font_size=18, color=WHITE).move_to(
                hist_axes.c2p(i + 0.5, 0) + DOWN * 0.3
            )
            bar_labels.add(label)

        bar_width = 0.7

        def make_bar(index, prob_func, color):
            return always_redraw(lambda idx=index, pf=prob_func, c=color: Rectangle(
                width=bar_width,
                height=max(0.01, hist_axes.y_axis.unit_size * pf(tracker.get_value())),
                fill_color=c, fill_opacity=0.8, stroke_color=c, stroke_width=1,
            ).move_to(hist_axes.c2p(idx + 0.5, 0), aligned_edge=DOWN))

        bar0 = make_bar(0, p_other, OTHER_COLOR)
        bar1 = make_bar(1, p_expected, EXPECTED_COLOR)
        bar2 = make_bar(2, p_other, OTHER_COLOR)
        bar3 = make_bar(3, p_other, OTHER_COLOR)

        hist_label = Text("Measured", font_size=22, color=GREY_B).next_to(
            hist_axes, UP, buff=0.1
        )

        # ── RIGHT: HF curve ──────────────────────────────────────
        curve_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1.1, 0.25],
            x_length=4, y_length=3,
            axis_config={"include_ticks": True, "include_numbers": True, "font_size": 18},
        ).shift(RIGHT * 3.2 + DOWN * 0.7)

        x_label = Text("Noise level", font_size=20, color=GREY_B).next_to(
            curve_axes.x_axis, DOWN, buff=0.35
        )

        # Plot HF(t) curve — linear: 1 - 0.75t
        hf_curve = curve_axes.plot(
            lambda t: hf_func(t), x_range=[0, 1, 0.01],
            color=HF_COLOR, stroke_width=3,
        )

        # Moving dot on curve
        moving_dot = always_redraw(
            lambda: Dot(
                curve_axes.c2p(tracker.get_value(), hf_func(tracker.get_value())),
                radius=0.08, color=WHITE,
            )
        )

        # Vertical dashed line from dot to x-axis
        vert_dash = always_redraw(lambda: DashedLine(
            curve_axes.c2p(tracker.get_value(), 0),
            curve_axes.c2p(tracker.get_value(), hf_func(tracker.get_value())),
            color=GREY_B, stroke_width=1, dash_length=0.08,
        ))

        # HF value label
        hf_value_label = always_redraw(lambda: MathTex(
            rf"\text{{HF}} = {hf_func(tracker.get_value()):.2f}",
            font_size=28, color=HF_COLOR,
        ).next_to(curve_axes, UP, buff=0.15))

        # ── Assemble and animate ──────────────────────────────────
        self.play(
            Create(hist_axes), Create(curve_axes),
            FadeIn(bar_labels), FadeIn(hist_label),
            FadeIn(x_label),
            run_time=1.5,
        )
        self.add(bar0, bar1, bar2, bar3)
        self.play(Create(hf_curve), run_time=1.5)
        self.add(moving_dot, vert_dash, hf_value_label)
        self.wait(0.5)

        # Key point labels on curve
        perfect_label = Text("Perfect", font_size=20, color=EXPECTED_COLOR).next_to(
            curve_axes.c2p(0, 1), UL, buff=0.15
        )
        random_label = Text("Random", font_size=20, color=COMPETITOR_COLOR).next_to(
            curve_axes.c2p(1, 0.25), DR, buff=0.15
        )
        self.play(FadeIn(perfect_label), FadeIn(random_label))

        # ── Main sweep animation ──────────────────────────────────
        self.play(
            tracker.animate.set_value(1),
            run_time=6, rate_func=linear,
        )
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
