#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_deck.py -- regenerate SIGIL_Edge_Pitch.pptx

The previous deck predated the container rule, the calibration law and the
certificates, and its title slide claimed results were "verified on real
Snapdragon silicon", which is not true -- nothing here has run on Snapdragon
hardware. Both are fixed.

Design system is lifted from the original deck so the two are visually
continuous: Arial throughout, navy 0B1E33, muted 5A6773, panel F2F4F7, green
1B7F5A on EAF5F0, red C4342B on FDF0EF, 10 x 5.625in.

The one chart uses 1F5FA8 / C9861A rather than the deck's green/red. Green
against red is the classic colour-vision-deficiency failure: validated, that
pair separates by only dE 8.1, right on the floor. The blue/amber pair
separates by 27.6 under protanopia and 32.9 for normal vision. Checked with the
dataviz validator rather than by eye.

    python build_deck.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import math

NAVY   = RGBColor(0x0B, 0x1E, 0x33)
MUTED  = RGBColor(0x5A, 0x67, 0x73)
PANEL  = RGBColor(0xF2, 0xF4, 0xF7)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
PALE   = RGBColor(0x9F, 0xB3, 0xC8)
GREEN  = RGBColor(0x1B, 0x7F, 0x5A)
GREENB = RGBColor(0xEA, 0xF5, 0xF0)
RED    = RGBColor(0xC4, 0x34, 0x2B)
REDB   = RGBColor(0xFD, 0xF0, 0xEF)
BLUE   = RGBColor(0x1F, 0x5F, 0xA8)   # validated pair, see docstring
AMBER  = RGBColor(0xC9, 0x86, 0x1A)
RULE   = RGBColor(0xD8, 0xDE, 0xE5)

W, H = Inches(10), Inches(5.625)
M = Inches(0.62)
FONT = "Arial"


def deck():
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if dark:
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = NAVY
    return s


def tb(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.paragraphs[0].alignment = align
    return tf


# Arial characters per inch, calibrated from a render: 45 characters at 13pt
# overflowed a 3.80in box, so roughly 154/size_pt characters fit per inch.
CHARS_PER_INCH_AT_1PT = 154.0


def assert_fits(text: str, width_in: float, size: float, where: str = "") -> None:
    """
    Refuse a string that will not fit on one line in a box of this width.

    Only for strings that MUST NOT wrap -- aligned numeric rows, labels inside
    a fixed card. The container slide shipped with "1.131 bpw" wrapping onto a
    second line and pushing the next row out of its panel, which is invisible
    in the code and obvious in the render.
    """
    budget = width_in * CHARS_PER_INCH_AT_1PT / size
    if len(text) > budget:
        raise ValueError(
            f"{where or 'text'} is {len(text)} chars but only ~{budget:.0f} fit "
            f"in {width_in:.2f}in at {size}pt: {text!r}")


def assert_card_fits(lines, width_in: float, height_in: float,
                     pad_in: float = 0.32, where: str = "") -> None:
    """
    Refuse a card whose wrapped text will spill past its panel.

    `lines` is [(text, size_pt, space_before_pt), ...]. Wrapped line count is
    estimated from the same characters-per-inch figure as assert_fits, and
    line height from 1.22 x the point size. Approximate on purpose -- it is a
    guard rail, not a layout engine, and it wants to fire slightly early.

    Three cards shipped with text hanging below their panels because the code
    looked fine and only the render showed it.
    """
    used = 0.0
    for text, size, before in lines:
        per_line = max(1.0, (width_in - 0.4) * CHARS_PER_INCH_AT_1PT / size)
        n = max(1, math.ceil(len(text) / per_line))
        used += (before + n * size * 1.22) / 72.0
    if used > height_in - pad_in:
        raise ValueError(
            f"{where or 'card'} needs ~{used + pad_in:.2f}in but the panel is "
            f"{height_in:.2f}in")


def para(tf, text, size, bold=False, color=NAVY, space_before=0, space_after=0,
         align=None, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    if align is not None:
        p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return p


def rect(slide, x, y, w, h, fill, line=None, radius=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    if radius:
        try:
            shape.adjustments[0] = radius
        except Exception:
            pass
    return shape


MAX_HEADING = 46
"""
Characters a 28pt Arial-bold heading fits on one line at this slide width.
Past it the title wraps onto a second line and lands on top of the subtitle,
which is what happened to the Dream-RSI slide. Asserted rather than eyeballed
because the overflow is only visible after a render.
"""


def heading(slide, text, sub=None):
    if len(text) > MAX_HEADING:
        raise ValueError(f"heading is {len(text)} chars, over the {MAX_HEADING} "
                         f"that fit on one line: {text!r}")
    if sub and len(sub) > 62:
        raise ValueError(f"subtitle is {len(sub)} chars, over 62: {sub!r}")
    tf = tb(slide, M, Inches(0.46), W - 2 * M, Inches(0.7))
    para(tf, text, 28, True, NAVY, first=True)
    y = Inches(1.08)
    if sub:
        tf2 = tb(slide, M, y, W - 2 * M, Inches(0.36))
        para(tf2, sub, 15, True, GREEN, first=True)
        y = Inches(1.52)
    return y


def dot(slide, cx, cy, d, fill, line=None, lw=1.5):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(cx - d / 2), int(cy - d / 2),
                                int(d), int(d))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #

def s_title(prs):
    s = blank(prs, dark=True)
    tf = tb(s, M, Inches(1.62), W - 2 * M, Inches(1.2))
    para(tf, "SIGIL-Edge", 46, True, WHITE, first=True)
    tf2 = tb(s, M, Inches(2.72), Inches(7.6), Inches(0.5))
    para(tf2, "On-device Indic document intelligence for Snapdragon-powered "
              "HP PCs", 16, False, PALE, first=True)
    rect(s, M, Inches(3.42), Inches(1.1), Pt(3), GREEN)
    tf3 = tb(s, M, Inches(3.78), Inches(8.4), Inches(1.0))
    para(tf3, "The binding quantity is the stored bit width on the wire, not TOPS.",
         15, True, WHITE, first=True)
    para(tf3, "Every hardware number here is Qualcomm's own measurement. The "
              "one device it lacks gets a zero-download Workbench job -- "
              "nothing lands on your laptop.", 12.5, False, PALE, space_before=7)


def s_hp_target(prs):
    """
    The machines the challenge names. This goes early because a submission
    judged on "optimised for Snapdragon-powered HP PCs" should say which ones,
    and be specific about where the evidence runs out.
    """
    s = blank(prs)
    y = heading(s, "Seven machines, and what each can hold",
                "The challenge names Snapdragon-powered HP PCs")

    rows = [
        ("OmniBook 3 14-HZ000", "Snapdragon X", "8", "X Plus 8-Core CRD *", False),
        ("OmniBook 5 16-bf000", "Snapdragon X", "16", "X Plus 8-Core CRD *", False),
        ("OmniBook Ultra 14-kg000", "Snapdragon X2 Plus", "16", "NONE", None),
        ("OmniBook Ultra 14", "Snapdragon X2 Elite", "16/32/64", "X2 Elite CRD", True),
        ("ProBook 4 G1q 14", "Snapdragon X", "16/32", "X Plus 8-Core CRD *", False),
        ("EliteBook 6 G1q 14", "Snapdragon X Elite", "32", "X Elite CRD", True),
        ("EliteBook Ultra G1q8 14", "Snapdragon X Plus", "16", "X Plus 8-Core CRD", True),
    ]
    cols = ((Inches(0.00), Inches(2.62), PP_ALIGN.LEFT),
            (Inches(2.66), Inches(2.06), PP_ALIGN.LEFT),
            (Inches(4.76), Inches(0.98), PP_ALIGN.RIGHT),
            (Inches(5.86), Inches(2.90), PP_ALIGN.LEFT))
    hdr = ("HP machine", "SoC", "RAM GB", "AI Hub device")
    for (cx, cw, al), t in zip(cols, hdr):
        h = tb(s, M + cx, y, cw, Inches(0.22), align=al)
        para(h, t, 9.5, True, MUTED, first=True, align=al)
    for i, (name, soc, ram, dev, exact) in enumerate(rows):
        ry = y + Inches(0.28) + i * Inches(0.255)
        col = RED if exact is None else (NAVY if exact else MUTED)
        for (cx, cw, al), t in zip(cols, (name, soc, ram, dev)):
            assert_fits(t, cw / 914400, 11, f"HP row {t!r}")
            c = tb(s, M + cx, ry, cw, Inches(0.24), align=al)
            para(c, t, 11, exact is None, col, first=True, align=al)

    yy = y + Inches(2.16)
    rect(s, M, yy, Inches(4.28), Inches(1.26), REDB)
    tf = tb(s, M + Inches(0.22), yy + Inches(0.14), Inches(3.86), Inches(1.0))
    para(tf, "Two gaps we name rather than paper over", 10.5, True, RED,
         first=True)
    para(tf, "X2 Plus has no AI Hub device -- measure on metal.", 11, False,
         NAVY, space_before=6)
    para(tf, "Never substitute the Elite. 8 GB leaves ~4-5 GB after Windows -- "
             "that machine is the design target.", 11, False, NAVY,
         space_before=5)

    rect(s, M + Inches(4.58), yy, Inches(4.28), Inches(1.26), PANEL)
    tf2 = tb(s, M + Inches(4.80), yy + Inches(0.14), Inches(3.86), Inches(1.0))
    para(tf2, "What Qualcomm has measured", 10.5, True, MUTED, first=True)
    para(tf2, "X Plus 8-Core: nothing", 16, True, RED, space_before=4)
    para(tf2, "X Elite 491 entries, X2 Elite 487. X Plus 8-Core stands in "
              "for four of these seven.", 11, False, NAVY, space_before=4)

    tf3 = tb(s, M, yy + Inches(1.46), W - 2 * M, Inches(0.6))
    para(tf3, "So those numbers have to be generated. One command profiles a "
              "real-dimension layer on that CRD: one upload, a few KB of JSON "
              "back, nothing downloaded.", 12, True, GREEN, first=True)


def s_mechanism(prs):
    s = blank(prs)
    y = heading(s, "The binding constraint is bandwidth, not TOPS",
                "Decode re-reads every weight once per token, at batch 1")
    cards = [("80 TOPS", "INT8 Hexagon NPU", "abundant", GREEN, GREENB),
             ("152 GB/s", "LPDDR5X", "the constraint", RED, REDB),
             ("R\u00b2 \u2265 0.99", "decode time vs bytes,\nQualcomm's own data",
              "measured", NAVY, PANEL)]
    cw = Inches(2.82)
    for i, (big, small, tag, fg, bg) in enumerate(cards):
        x = M + i * Inches(2.94)
        rect(s, x, y, cw, Inches(1.62), bg)
        tf = tb(s, x + Inches(0.22), y + Inches(0.2), cw - Inches(0.44), Inches(1.3))
        para(tf, big, 24, True, fg, first=True)
        para(tf, small, 12.5, False, NAVY, space_before=5)
        para(tf, tag.upper(), 9.5, True, fg, space_before=6)
    yy = y + Inches(1.92)
    tf = tb(s, M, yy, W - 2 * M, Inches(1.4))
    para(tf, "Tested on Qualcomm's measurements, not asserted", 14, True, NAVY,
         first=True)
    para(tf, "Qwen3 0.6B to 8B on the X Elite NPU, under QAIRT: decode time is "
             "a straight line in bytes moved, under every way of counting the "
             "LM head and KV cache.", 12.5, False, MUTED, space_before=6)
    para(tf, "If the weight stream is what binds, then how the weights are "
             "PACKED is a throughput decision -- not a packaging detail.",
         13.5, True, NAVY, space_before=11)


def s_engines(prs):
    """
    Qualcomm's own X2 Elite numbers for one model, three engines, two phases.
    Two panels on separate scales on purpose: decode and prefill differ by 60x,
    and a shared axis would flatten the decode bars into slivers. Each panel is
    one series, so one colour; the engines are named on the axis, and each bar
    carries its value because there are only three.
    """
    s = blank(prs)
    y = heading(s, "The bus sets decode; the engine sets prefill",
                "Qwen3-4B on X2 Elite, each engine on its best runtime")
    panels = [("decode, tokens/s", [("CPU", 33.7), ("GPU", 33.5), ("NPU", 36.2)],
               40.0, "spread 1.08x -- the bus is shared"),
              ("prefill, tokens/s", [("CPU", 461), ("GPU", 464), ("NPU", 2307)],
               2400.0, "spread 5.0x -- the arithmetic is not")]
    pw, ph = Inches(4.28), Inches(2.02)
    for i, (title, bars, top, note) in enumerate(panels):
        x = M + i * Inches(4.58)
        rect(s, x, y, pw, ph, PANEL)
        t = tb(s, x + Inches(0.22), y + Inches(0.14), Inches(3.8), Inches(0.24))
        para(t, title, 10.5, True, MUTED, first=True)
        lab_w = Inches(0.52)
        bx = x + Inches(0.22) + lab_w
        bw_max = pw - Inches(0.22) - lab_w - Inches(0.86)
        for j, (name, v) in enumerate(bars):
            cy = y + Inches(0.56) + j * Inches(0.38)
            tl = tb(s, x + Inches(0.22), cy, lab_w - Inches(0.08), Inches(0.26),
                    align=PP_ALIGN.RIGHT)
            para(tl, name, 11, True, NAVY, first=True, align=PP_ALIGN.RIGHT)
            w = max(int(bw_max * v / top), Pt(3))
            rect(s, bx, cy + Inches(0.03), w, Inches(0.2), BLUE)
            tv = tb(s, bx + w + Inches(0.08), cy, Inches(0.8), Inches(0.26))
            para(tv, f"{v:,}", 11, False, NAVY, first=True)
        tn = tb(s, x + Inches(0.22), y + Inches(1.66), Inches(3.8), Inches(0.26))
        para(tn, note, 10.5, True, GREEN, first=True)
    tf = tb(s, M, y + Inches(2.24), W - 2 * M, Inches(1.2))
    para(tf, "Prefill over decode is the scaling book's critical batch: 63-71 on "
             "the NPU across three model sizes, 14-17 on the CPU. It is how many "
             "draft tokens a verifier checks for the price of one step.",
         12.5, False, NAVY, first=True)
    para(tf, "So speculative verification belongs on the NPU.", 13.5, True,
         GREEN, space_before=7)
    para(tf, "The software still matters: on X Elite, QAIRT decodes the same "
             "model 3.2x faster than Genie on the same NPU.", 11.5, False,
         MUTED, space_before=9)


def s_workbench(prs):
    """
    The zero-download Workbench path as four steps, and what is and is not
    proven about it. The last step is the point: what comes back is JSON.
    """
    s = blank(prs)
    y = heading(s, "Zero-download testing on AI Hub Workbench",
                "Aimed at the one X-series device Qualcomm has not measured")
    steps = [("Build", "one Qwen3-4B layer at real dimensions, 84 MB zipped"),
             ("Upload once", "plus 34 MB of calibration data"),
             ("Profile", "X Plus 8-Core CRD at fp16, w8a16, w4a16"),
             ("Download", "profile JSON only, a 2 MB hard ceiling")]
    bw, gap = Inches(2.0), Inches(0.22)
    for i, (big, small) in enumerate(steps):
        x = M + i * (bw + gap)
        last = i == len(steps) - 1
        rect(s, x, y, bw, Inches(1.18), GREENB if last else PANEL)
        t = tb(s, x + Inches(0.16), y + Inches(0.14), bw - Inches(0.32),
               Inches(0.96))
        para(t, big, 14, True, GREEN if last else NAVY, first=True)
        para(t, small, 11, False, NAVY, space_before=5)
        if not last:
            ta = tb(s, x + bw, y + Inches(0.42), gap, Inches(0.3),
                    align=PP_ALIGN.CENTER)
            para(ta, ">", 13, True, MUTED, first=True, align=PP_ALIGN.CENTER)
    yy = y + Inches(1.40)
    cw, ch = Inches(4.28), Inches(1.46)
    rect(s, M, yy, cw, ch, PANEL)
    t1 = tb(s, M + Inches(0.22), yy + Inches(0.14), Inches(3.84), Inches(1.2))
    para(t1, "Checked without a token; runs with yours", 10.5, True, MUTED,
         first=True)
    para(t1, "Every call matches the installed client, parameter by "
             "parameter. The archive passes the client's own check. The graph "
             "matches a NumPy reference to 1e-7.", 11.5, False, NAVY,
         space_before=5)
    right = M + Inches(4.58)
    rect(s, right, yy, cw, ch, GREENB)
    t2 = tb(s, right + Inches(0.22), yy + Inches(0.14), Inches(3.84), Inches(1.2))
    para(t2, "Written down before it runs", 10.5, True, GREEN, first=True)
    para(t2, "~21 tok/s", 18, True, GREEN, space_before=4)
    para(t2, "Qwen3-4B w4a16 at 4K context, if X Plus 8-Core shares X Elite's "
             "135 GB/s bus.", 11.5, False, NAVY, space_before=4)


def s_container(prs):
    s = blank(prs)
    y = heading(s, "The container rule",
                "Up to 35% of decode throughput, free, on identical weights")
    cw = Inches(4.28)
    ch = Inches(1.92)

    # Left card: one text box per cell, so a long number cannot wrap and shove
    # the row below it out of the panel.
    rect(s, M, y, cw, ch, PANEL)
    t = tb(s, M + Inches(0.22), y + Inches(0.16), Inches(3.0), Inches(0.24))
    para(t, "Bonsai-27B, the same 1-bit weights", 10.5, True, MUTED, first=True)
    cols = ((Inches(0.22), Inches(1.30), PP_ALIGN.LEFT),
            (Inches(1.52), Inches(1.55), PP_ALIGN.RIGHT),
            (Inches(3.10), Inches(0.96), PP_ALIGN.RIGHT))
    hdr = ("container", "bytes", "bpw")
    for (cx, cwid, al), htxt in zip(cols, hdr):
        h = tb(s, M + cx, y + Inches(0.44), cwid, Inches(0.22), align=al)
        para(h, htxt, 9, False, MUTED, first=True, align=al)
    for i, (name, nbytes, bpw, col) in enumerate(
            (("GGUF Q1_0", "3,803,452,480", "1.131", BLUE),
             ("MLX 1-bit", "5,129,115,752", "1.525", AMBER))):
        ry = y + Inches(0.70) + i * Inches(0.30)
        for (cx, cwid, al), txt in zip(cols, (name, nbytes, bpw)):
            assert_fits(txt, cwid / 914400, 12.5, f"container cell {txt!r}")
            c = tb(s, M + cx, ry, cwid, Inches(0.26), align=al)
            para(c, txt, 12.5, True, col, first=True, align=al)
    t2 = tb(s, M + Inches(0.22), y + Inches(1.34), Inches(3.86), Inches(0.5))
    para(t2, "Same weights, same accuracy, 35% more bytes: MLX stores a "
             "scale AND a zero-point per group.",
         11.5, False, NAVY, first=True)

    right = M + Inches(4.58)
    rect(s, right, y, cw, ch, GREENB)
    t3 = tb(s, right + Inches(0.24), y + Inches(0.16), Inches(3.8), Inches(0.24))
    para(t3, "What it costs to collect", 10.5, True, GREEN, first=True)
    for i, line in enumerate(("Nothing is retrained.", "No accuracy is spent.",
                              "No kernel is written.",
                              "It is a choice of container.")):
        c = tb(s, right + Inches(0.24), y + Inches(0.50) + i * Inches(0.31),
               Inches(3.8), Inches(0.28))
        para(c, line, 13, True, NAVY, first=True)

    yy = y + Inches(2.18)
    tf3 = tb(s, M, yy, W - 2 * M, Inches(1.0))
    para(tf3, "The same trade appears one level up: ternary Bonsai 2 ships as "
              "PTQ1_0 at 1.768 bpw and PQ2_0 at 2.143 bpw, with throughput "
              "published for both on eight machines.", 12.5, False,
         NAVY, first=True)
    para(tf3, "That is a falsification test we did not fit anything to.",
         14, True, GREEN, space_before=8)


def s_chart(prs):
    """
    One row per device, sorted by bandwidth.

    The first version put every device on a single shared axis line. Four label
    pairs collided (RTX 6000 Ada over A100, L40S over RTX PRO 6000), the
    Snapdragon markers carried no labels at all, and the crossover band ran
    underneath the text. A dot plot with one row per item cannot collide, which
    is the whole reason it is the standard form for ranked comparisons.
    """
    s = blank(prs)
    heading(s, "Predicted on 8 of 8, from bandwidth alone",
            "Sign prediction only -- no constant was fitted to this data")

    lab_r = Inches(2.34)          # right edge of the label column
    ax_l, ax_r = Inches(2.50), Inches(9.38)
    ax_w = ax_r - ax_l
    top = Inches(1.60)
    rh = Inches(0.235)

    def X(bw):
        return int(ax_l + (math.log10(bw) - 1.602) / 2.0 * ax_w)

    rows = [(300, "L4", "PTQ1_0"), (864, "L40S", "PTQ1_0"),
            (960, "RTX 6000 Ada", "PTQ1_0"), (1008, "RTX 4090", "PTQ1_0"),
            (1792, "RTX 5090", "PQ2_0"), (1792, "RTX PRO 6000", "PQ2_0"),
            (2039, "A100 SXM", "PQ2_0"), (3350, "H100 SXM", "PQ2_0")]
    n_rows = len(rows) + 1
    bot = int(top + n_rows * rh)

    # crossover band, behind everything
    bx0, bx1 = X(1008), X(1792)
    rect(s, bx0, top, bx1 - bx0, bot - top + Inches(0.06),
         RGBColor(0xE6, 0xEA, 0xEF))

    for i, (bw, name, win) in enumerate(rows):
        cy = int(top + i * rh + rh / 2)
        col = BLUE if win == "PTQ1_0" else AMBER
        # faint leader from the label to the dot
        rect(s, ax_l, cy, X(bw) - ax_l, Pt(0.75), RGBColor(0xEC, 0xEF, 0xF3))
        dot(s, X(bw), cy, Inches(0.155), col)
        tl = tb(s, Inches(0.62), cy - Inches(0.095), lab_r - Inches(0.62),
                Inches(0.2), align=PP_ALIGN.RIGHT)
        para(tl, name, 10, False, NAVY, first=True, align=PP_ALIGN.RIGHT)
        tv = tb(s, X(bw) + Inches(0.13), cy - Inches(0.095), Inches(0.9),
                Inches(0.2))
        para(tv, f"{bw:,}", 9, False, MUTED, first=True)

    # Snapdragon as a single band -- five parts, one row, no collisions
    cy = int(top + len(rows) * rh + rh / 2)
    sx0, sx1 = X(51), X(228)
    rect(s, sx0, cy - Inches(0.055), sx1 - sx0, Inches(0.11), RGBColor(0xC7, 0xD6, 0xE8))
    for bw in (51, 67, 77, 135, 152, 228):
        dot(s, X(bw), cy, Inches(0.13), WHITE, BLUE, 1.5)
    tl = tb(s, Inches(0.62), cy - Inches(0.095), lab_r - Inches(0.62),
            Inches(0.2), align=PP_ALIGN.RIGHT)
    para(tl, "Snapdragon, 9 parts", 10, True, BLUE, first=True,
         align=PP_ALIGN.RIGHT)
    tv = tb(s, X(228) + Inches(0.13), cy - Inches(0.095), Inches(2.6), Inches(0.2))
    para(tv, "51-228   ->   4.4-19.7x below", 9, True, BLUE, first=True)

    # axis
    rect(s, ax_l, bot, ax_w, Pt(1), RULE)
    for bw, lab in ((50, "50"), (100, "100"), (200, "200"), (500, "500"),
                    (1000, "1,000"), (2000, "2,000"), (4000, "4,000")):
        tx = tb(s, X(bw) - Inches(0.42), int(bot + Inches(0.06)), Inches(0.84),
                Inches(0.24), align=PP_ALIGN.CENTER)
        para(tx, lab, 9, False, MUTED, first=True, align=PP_ALIGN.CENTER)
    tfb = tb(s, bx0, int(bot + Inches(0.30)), bx1 - bx0, Inches(0.24),
             align=PP_ALIGN.CENTER)
    para(tfb, "crossover 1,008-1,792", 9, True, MUTED, first=True,
         align=PP_ALIGN.CENTER)
    tfa = tb(s, ax_l, int(bot + Inches(0.30)), Inches(3.0), Inches(0.24))
    para(tfa, "memory bandwidth, GB/s (log)", 9.5, False, MUTED, first=True)

    # legend
    ly = int(bot + Inches(0.66))
    items = [(BLUE, True, "PTQ1_0 wins  -  narrower stream, 1.768 bpw"),
             (AMBER, True, "PQ2_0 wins  -  cheaper unpack, 2.143 bpw"),
             (BLUE, False, "Snapdragon, predicted")]
    lx = M
    for col, filled, text in items:
        dot(s, lx + Inches(0.06), ly + Inches(0.07), Inches(0.13),
            col if filled else WHITE, None if filled else col, 1.5)
        tl = tb(s, lx + Inches(0.22), ly - Inches(0.02), Inches(3.1), Inches(0.24))
        para(tl, text, 10, False, MUTED, first=True)
        lx += Inches(3.18)

    tf = tb(s, M, int(bot + Inches(1.02)), W - 2 * M, Inches(0.5))
    para(tf, "Where the narrower container wins it realises 0.46 of the ideal "
             "byte advantage -- independently close to our LUT arithmetic "
             "efficiency of 0.60, calibrated against different hardware.",
         11.5, False, NAVY, first=True)


def s_folding(prs):
    s = blank(prs)
    y = heading(s, "Zero added operators -- and it already ships",
                "Orthogonal transforms fold exactly into the stored weights")
    cw, ch = Inches(4.28), Inches(2.16)

    rect(s, M, y, cw, ch, PANEL)
    tf = tb(s, M + Inches(0.24), y + Inches(0.16), Inches(3.80), Inches(1.86))
    para(tf, "Our verification", 10.5, True, MUTED, first=True)
    para(tf, "Attention logits and block outputs are unchanged under "
             "W_q/W_k and W_v/W_o folding.", 12, False, NAVY, space_before=7)
    para(tf, "Exact to 1e-15 in float64", 14.5, True, GREEN, space_before=8)
    para(tf, "Zero operators added to the deployed graph. A RoPE guard refuses "
             "unsafe folds -- a failure that only appears past a few thousand "
             "tokens.", 11.5, False, NAVY, space_before=8)

    right = M + Inches(4.58)
    rect(s, right, y, cw, ch, GREENB)
    tf2 = tb(s, right + Inches(0.24), y + Inches(0.16), Inches(3.80), Inches(1.86))
    para(tf2, "External confirmation", 10.5, True, GREEN, first=True)
    para(tf2, "Bonsai 2 27B stores its weights in a blockwise Hadamard basis, "
              "folded in, with the inverse folded into the embedding.",
         12, False, NAVY, space_before=7)
    para(tf2, "Metadata: 0.0035% of the pack", 14.5, True, GREEN, space_before=8)
    para(tf2, "Free at runtime, negligible at rest. Their widths are exactly "
              "5, 6 and 17 blocks of 1024 -- a divisibility constraint we now "
              "enforce before export.", 11.5, False, NAVY, space_before=6)


def s_calibration(prs):
    s = blank(prs)
    y = heading(s, "When is another refinement round real?",
                "Measured on synthetic matrices, after a correction-cycle proof")
    rect(s, M, y, W - 2 * M, Inches(0.72), NAVY)
    tf = tb(s, M, y + Inches(0.17), W - 2 * M, Inches(0.42), align=PP_ALIGN.CENTER)
    para(tf, "kept  ~  1  -  1.75 * d / n", 21, True, WHITE,
         first=True, align=PP_ALIGN.CENTER)
    tf0 = tb(s, M, y + Inches(0.80), W - 2 * M, Inches(0.3))
    para(tf0, "d = matrix input dimension     n = calibration tokens     "
              "kept = fraction of the measured gain that survives on held-out data",
         11, False, MUTED, first=True)

    yy = y + Inches(1.22)
    rows = [("n/d = 0.5", "calibration 2.00x", "held-out 0.77x -- WORSE", RED, REDB),
            ("n/d = 8", "calibration 1.35x", "held-out 1.27x  -  77% kept", GREEN, GREENB)]
    for i, (a, b, c, fg, bg) in enumerate(rows):
        x = M + i * Inches(4.58)
        rect(s, x, yy, Inches(4.28), Inches(0.94), bg)
        tf2 = tb(s, x + Inches(0.22), yy + Inches(0.14), Inches(3.9), Inches(0.7))
        para(tf2, a, 13.5, True, fg, first=True)
        para(tf2, b + "     ->     " + c, 12, False, NAVY, space_before=5)
    tf3 = tb(s, M, yy + Inches(1.18), W - 2 * M, Inches(0.9))
    para(tf3, "Flat in d (75-78% for d = 32 to 256) and flat in bit width "
              "(75-79% for 8 down to 2 bits) -- a law in the ratio, not a curve.",
         12.5, False, NAVY, first=True)
    para(tf3, "For Bonsai 2 the law predicts that a 128 x 512 calibration set "
              "keeps only 54% of any gain on down_proj -- the matrix that "
              "always binds, because it alone sees the intermediate dimension.",
         13, True, NAVY, space_before=7)


def s_certificates(prs):
    s = blank(prs)
    y = heading(s, "A quantised model fails silently",
                "So ship something the device can check before it loads")
    rect(s, M, y, W - 2 * M, Inches(0.98), REDB)
    tf = tb(s, M + Inches(0.24), y + Inches(0.14), W - 2 * M - Inches(0.48),
            Inches(0.74))
    para(tf, "'Ordinary MLX loaders do not apply the required transforms.'",
         13.5, True, RED, first=True)
    para(tf, "A loader that does not know about the rotated basis returns WRONG "
             "OUTPUT rather than an error. It loads, it runs, it reads fluently, "
             "and it is wrong.", 12, False, NAVY, space_before=5)

    yy = y + Inches(1.22)
    cols = [("~3 KB", "JSON beside the weights: scheme, group AXIS, "
                      "transform, sign digest, seeded probes"),
            ("7 of 7", "injected faults refused, at four problem sizes and seeds"),
            ("~0.5 ms", "per check on a laptop CPU: no calibration data, "
                        "no reference weights, no network")]
    for i, (big, small) in enumerate(cols):
        x = M + i * Inches(2.94)
        assert_card_fits([(big, 19, 0), (small, 11.5, 6)], 2.82, 1.42,
                         where=f"certificate card {big!r}")
        rect(s, x, yy, Inches(2.82), Inches(1.42), PANEL)
        tf2 = tb(s, x + Inches(0.2), yy + Inches(0.16), Inches(2.42), Inches(1.16))
        para(tf2, big, 19, True, GREEN, first=True)
        para(tf2, small, 11.5, False, NAVY, space_before=6)
    tf3 = tb(s, M, yy + Inches(1.62), W - 2 * M, Inches(0.6))
    para(tf3, "Detection generalises. Naming which fault occurred does not "
              "(4/7 to 6/7 away from its fitted harness), so the tool returns a "
              "ranked hint and says so.", 12.5, False, NAVY, first=True)


def s_dream(prs):
    s = blank(prs)
    y = heading(s, "Your benchmark history is a simulator",
                "Score a cheaper sweep against runs you already paid for")
    rect(s, M, y, W - 2 * M, Inches(0.74), PANEL)
    tf = tb(s, M + Inches(0.24), y + Inches(0.14), W - 2 * M - Inches(0.48),
            Inches(0.5))
    para(tf, "Every 90-minute sweep writes results_t4.json: 18 configurations, "
             "each with its answer. A different search policy can be scored "
             "against that file for nothing.", 12.5, False, NAVY, first=True)

    yy = y + Inches(1.00)
    rows = [("exhaustive", "18", "1.0x", NAVY),
            ("coordinate", "6", "3.0x", NAVY),
            ("greedy", "4", "4.5x", GREEN)]
    print_w = Inches(4.28)
    rect(s, M, yy, print_w, Inches(1.52), WHITE, RULE)
    tfh = tb(s, M + Inches(0.2), yy + Inches(0.12), Inches(3.9), Inches(0.24))
    para(tfh, "evaluations to the same answer", 11, True, MUTED, first=True)
    # One text box per cell: a single formatted string wraps at the box edge
    # and the rows walk over each other.
    cols = (Inches(0.20), Inches(1.80), Inches(2.70))
    for i, (name, n, x, col) in enumerate(rows):
        ty = yy + Inches(0.46) + i * Inches(0.32)
        for cx, txt, al in ((cols[0], name, PP_ALIGN.LEFT),
                            (cols[1], n, PP_ALIGN.RIGHT),
                            (cols[2], x, PP_ALIGN.RIGHT)):
            t = tb(s, M + cx, ty, Inches(1.2), Inches(0.28), align=al)
            para(t, txt, 12, col is GREEN, col, first=True, align=al)

    rect(s, M + Inches(4.58), yy, print_w, Inches(1.52), GREENB)
    tf2 = tb(s, M + Inches(4.82), yy + Inches(0.16), Inches(3.8), Inches(1.2))
    para(tf2, "The honest limit", 11, True, GREEN, first=True)
    para(tf2, "A replay is valid only INSIDE the realized space. A policy "
              "asking for a run nobody made is a MISS, not evidence.", 12,
         False, NAVY, space_before=6)
    para(tf2, "Unreliable runs are never ranked.", 12, True,
         GREEN, space_before=6)

    tf3 = tb(s, M, yy + Inches(1.78), W - 2 * M, Inches(0.8))
    para(tf3, "Both shipped pools are flagged as unable to rank policies "
              "fairly -- one has its optimum at the first configuration tried, "
              "the other is almost flat. The tool says so before it shows a "
              "table, because a comparison that looks decisive and is not is "
              "worse than none.", 12, False, NAVY, first=True)


def s_rejected(prs):
    s = blank(prs)
    y = heading(s, "What we tested and rejected",
                "Reported, not buried")
    items = [
        ("Learned rotation on a Gaussianity objective",
         "lost to a random Hadamard on every integer condition"),
        ("Isotropy-based out-of-distribution routing",
         "mathematically impossible: ||XR|| = ||X|| for orthogonal R"),
        ("Low-bit models resist iterative refinement",
         "predicted from the correction-cycle floor; it converges identically "
         "at 8 through 2 bits"),
        ("'A 27B model at 1-bit is 3.4 GB'  -  our own claim",
         "12% under the real 3.80 GB file, and the self-test guarding it had "
         "been calibrated to the invented number"),
        ("'Qualcomm publishes no X-series numbers'  -  our own claim",
         "withdrawn: 978 measured X Elite and X2 Elite entries. The real gap is "
         "X Plus 8-Core, which has none"),
    ]
    for i, (a, b) in enumerate(items):
        yy = y + i * Inches(0.64)
        rect(s, M, yy, Pt(3), Inches(0.52), RED)
        tf = tb(s, M + Inches(0.20), yy, W - 2 * M - Inches(0.2), Inches(0.56))
        para(tf, a, 12.5, True, NAVY, first=True)
        para(tf, b, 11, False, MUTED, space_before=2)
    tf2 = tb(s, M, y + Inches(3.26), W - 2 * M, Inches(0.5))
    para(tf2, "Nine predictions that a source would be barren. Nine wrong. "
              "Every prediction of absence has failed; every measurement has held.",
         12.5, True, GREEN, first=True)


def s_deploy(prs):
    s = blank(prs)
    y = heading(s, "Three deployment paths, no bespoke runtime")
    cards = [("GenieX", "NPU + GGUF serving,\nOpenAI-compatible endpoint", "BSD-3"),
             ("llmware + ONNXRuntime-QNN",
              "document parsing and RAG\non the Snapdragon NPU", "Apache-2.0"),
             ("AI Hub Workbench",
              "compile and profile on real\nX-series devices, zero download", "free tier")]
    for i, (name, what, lic) in enumerate(cards):
        x = M + i * Inches(2.94)
        rect(s, x, y, Inches(2.82), Inches(1.74), PANEL)
        tf = tb(s, x + Inches(0.22), y + Inches(0.2), Inches(2.4), Inches(1.4))
        para(tf, name, 14.5, True, NAVY, first=True)
        para(tf, what, 12, False, NAVY, space_before=7)
        para(tf, lic.upper(), 9.5, True, GREEN, space_before=8)
    yy = y + Inches(2.04)
    tf2 = tb(s, M, yy, W - 2 * M, Inches(1.4))
    para(tf2, "PREFILL runs on the Hexagon NPU -- INT4, static shapes, via "
              "GenieX or ONNX Runtime QNN. It is compute-bound, which is what "
              "the NPU is for.", 12.5, False, NAVY, first=True)
    para(tf2, "DECODE is bound by the bus: on X2 Elite at INT4, the NPU and "
              "CPU decode alike. The ternary container runs on the ARM CPU, "
              "because stock QNN has no ternary matmul.", 12.5, False,
         NAVY, space_before=6)
    para(tf2, "Claiming the whole pipeline runs on the NPU is the claim that "
              "gets taken apart. Claiming the NPU is unusable is equally wrong.",
         12.5, True, RED, space_before=8)


def s_usecase(prs):
    s = blank(prs)
    y = heading(s, "Indic document intelligence",
                "Privacy and offline operation are the product, not a feature")
    tf = tb(s, M, y, Inches(5.4), Inches(1.6))
    para(tf, "Statutory forms, land records, exam papers, health documents.",
         13.5, False, NAVY, first=True)
    para(tf, "Hindi - Marathi - Gujarati - Punjabi - Telugu "
             "- Kannada - Tamil - Malayalam", 12.5, True, NAVY,
         space_before=8)
    para(tf, "Exactly the documents that must not leave the device, in exactly "
             "the markets where connectivity is unreliable and Snapdragon share "
             "is highest.", 12, False, MUTED, space_before=8)

    x2 = M + Inches(5.7)
    rect(s, x2, y, Inches(3.16), Inches(2.02), GREENB)
    tf2 = tb(s, x2 + Inches(0.22), y + Inches(0.18), Inches(2.74), Inches(1.7))
    para(tf2, "An unclaimed win", 11, True, GREEN, first=True)
    para(tf2, "The Bonsai 2 vision tower ships unrotated and unquantised: 1.7% "
              "of parameters, 13.5% of bytes.", 12, False, NAVY, space_before=7)
    para(tf2, "0.83 GB available", 15, True, GREEN, space_before=7)
    para(tf2, "UNMEASURED -- a hypothesis with an experiment; the tower may "
              "carry vision.", 10.5, False, MUTED, space_before=5)

    yy = y + Inches(2.28)
    rect(s, M, yy, Inches(4.28), Inches(1.06), PANEL)
    tf3 = tb(s, M + Inches(0.22), yy + Inches(0.14), Inches(3.9), Inches(0.82))
    para(tf3, "Runs on device", 12.5, True, GREEN, first=True)
    para(tf3, "Gemma 4 E2B / E4B  -  Mistral  -  Qwen3 / Qwen3.5  "
              "-  Bonsai ternary GGUF", 12, False, NAVY, space_before=5)
    rect(s, M + Inches(4.58), yy, Inches(4.28), Inches(1.06), REDB)
    tf4 = tb(s, M + Inches(4.80), yy + Inches(0.14), Inches(3.9), Inches(0.82))
    para(tf4, "Cannot run on device, at any ratio", 12.5, True, RED,
         first=True)
    para(tf4, "Closed-weight frontier models. A network-cost verifier of "
              "last resort, not something to compress.", 12, False, NAVY,
         space_before=4)


def s_repro(prs):
    s = blank(prs, dark=True)
    tf = tb(s, M, Inches(0.72), W - 2 * M, Inches(0.7))
    para(tf, "Reproduce all of it", 28, True, WHITE, first=True)
    rect(s, M, Inches(1.42), Inches(1.1), Pt(3), GREEN)

    stats = [("872", "self-tests across\n14 modules"),
             ("17 / 17", "scaling-book answers\nreproduced"),
             ("0", "models downloaded\nto your laptop"),
             ("292", "adversarial checks\nin one command")]
    y = Inches(1.86)
    for i, (big, small) in enumerate(stats):
        x = M + i * Inches(2.20)
        tfx = tb(s, x, y, Inches(2.0), Inches(1.0))
        para(tfx, big, 24, True, WHITE, first=True)
        para(tfx, small, 11.5, False, PALE, space_before=6)

    yy = Inches(3.24)
    tf2 = tb(s, M, yy, W - 2 * M, Inches(1.4))
    para(tf2, "python roofline.py law", 13, True, GREEN, first=True)
    para(tf2, "python aihub_workbench.py run --yes", 13, True, GREEN,
         space_before=4)
    para(tf2, "python snapdragon_engine.py container --validate", 13, True,
         GREEN, space_before=4)
    para(tf2, "python stress_all.py", 13, True, GREEN, space_before=4)
    para(tf2, "Every source, status and correction is in PROVENANCE.md -- "
              "including the corrections to our own published numbers.", 12,
         False, PALE, space_before=10)


# --------------------------------------------------------------------------- #
# Overflow audit -- measured, not estimated
# --------------------------------------------------------------------------- #
#
# `assert_card_fits` is a per-call-site guard and it only guards the sites that
# remember to call it. Slide 2's "Nothing." card never did, and shipped with the
# word "figures." hanging below its panel -- visible only in a render, exactly
# like the three cards before it.
#
# This replaces per-site discipline with a measurement over the FINISHED deck.
# Every text box is measured with real glyph advances from Liberation Sans,
# which is metrically identical to Arial and is what LibreOffice substitutes
# when it renders this file -- so the wrap computed here is the wrap that gets
# rendered, not a characters-per-inch approximation.

LINE_SPACING = 1.22          # matches assert_card_fits, and the observed render
_FONT_PATHS = {
    False: "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    True:  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
}


def _font(bold: bool, size_pt: float):
    from PIL import ImageFont
    # Render at 4x for sub-point accuracy, then scale the advance back down.
    return ImageFont.truetype(_FONT_PATHS[bold], max(1, int(round(size_pt * 4))))


def _wrapped_lines(text: str, width_pt: float, size_pt: float, bold: bool) -> int:
    """How many lines this text occupies in a box `width_pt` wide."""
    if not text.strip():
        return 1
    f = _font(bold, size_pt)

    def adv(s: str) -> float:
        return f.getlength(s) / 4.0

    lines, cur = 1, ""
    for word in text.split(" "):
        trial = word if not cur else cur + " " + word
        if adv(trial) <= width_pt or not cur:
            cur = trial
        else:
            lines += 1
            cur = word
    return lines


def _is_panel(shape) -> bool:
    """A filled background rectangle: an autoshape carrying no text."""
    if shape.has_text_frame and shape.text_frame.text.strip():
        return False
    try:
        return shape.shape_type is not None and shape.fill.type is not None
    except Exception:
        return False


def _enclosing_panel(shape, shapes):
    """
    The tightest filled rectangle whose top-left corner is above and left of
    this text box. A text box inside a card may overflow its own declared
    height harmlessly -- what matters is whether it escapes the PANEL, which
    is the thing the reader sees an edge on.
    """
    best = None
    for other in shapes:
        if other is shape:
            continue
        # A panel is an autoshape with no text of its own. Skipping everything
        # with a text frame was wrong: python-pptx gives EVERY autoshape a text
        # frame, so that test skipped the panels themselves and the audit went
        # quiet on the one card it was written to catch.
        if not _is_panel(other):
            continue
        if other.width < 12700 * 20 or other.height < 12700 * 10:
            continue                       # a rule or a tick, not a panel
        # It must actually CONTAIN the text box's top-left corner. Without the
        # vertical test a row stripe that ends above the box counted as its
        # panel and the available height came out negative.
        if not (other.left <= shape.left
                and other.top <= shape.top < other.top + other.height
                and other.left + other.width >= shape.left + shape.width - 9144):
            continue
        if best is None or other.height < best.height:
            best = other
    return best


def _bottom_limit(shape, shapes, slide_h_emu, margin_emu):
    """How far down this text box may extend, in points from its own top."""
    panel = _enclosing_panel(shape, shapes)
    if panel is not None:
        top_pad = shape.top - panel.top
        return (panel.height - top_pad - min(top_pad, 91440)) / 12700.0
    # Free-floating: stop at whatever comes next below, else the bottom margin.
    floor = slide_h_emu - margin_emu
    for other in shapes:
        if other is shape or other.top <= shape.top:
            continue
        if other.left + other.width <= shape.left or other.left >= shape.left + shape.width:
            continue                       # no horizontal overlap, cannot collide
        floor = min(floor, other.top)
    return max(shape.height, floor - shape.top) / 12700.0


def audit_overflow(prs, verbose: bool = True) -> list:
    """
    Every text box whose content escapes the panel that contains it, or -- for
    text on the bare slide -- runs into whatever sits below it or off the page.
    """
    try:
        _font(False, 12)
    except Exception as exc:                                  # pragma: no cover
        if verbose:
            print(f"  overflow audit SKIPPED: no Liberation Sans ({exc})")
        return []
    bad = []
    margin = int(0.30 * 914400)
    for idx, slide in enumerate(prs.slides, 1):
        shapes = list(slide.shapes)
        for shape in shapes:
            if not shape.has_text_frame:
                continue
            tf = shape.text_frame
            box_w_pt = (shape.width - tf.margin_left - tf.margin_right) / 12700.0
            if box_w_pt <= 0:
                continue
            used = 0.0
            for p in tf.paragraphs:
                if not p.runs:
                    continue
                text = "".join(r.text for r in p.runs)
                size = (p.runs[0].font.size or 0) / 12700.0 or 12.0
                bold = bool(p.runs[0].font.bold)
                n = _wrapped_lines(text, box_w_pt, size, bold)
                used += ((p.space_before or 0) / 12700.0
                         + n * size * LINE_SPACING
                         + (p.space_after or 0) / 12700.0)
            have = _bottom_limit(shape, shapes, prs.slide_height, margin)
            if used > have + 1.0:              # 1pt of slack for rounding
                first = "".join(r.text for r in tf.paragraphs[0].runs)[:44]
                bad.append((idx, round(used, 1), round(have, 1), first))
    if verbose:
        if bad:
            print("  TEXT ESCAPES ITS PANEL:")
            for slide_no, used, have, first in bad:
                print(f"    slide {slide_no:>2}: needs {used:.1f}pt, "
                      f"{have:.1f}pt available -- {first!r}")
        else:
            print("  overflow audit: every text box stays inside its panel")
    return bad


def main():
    prs = deck()
    for fn in (s_title, s_hp_target, s_mechanism, s_engines, s_workbench,
               s_container, s_chart, s_folding, s_calibration, s_certificates,
               s_dream, s_rejected, s_deploy, s_usecase, s_repro):
        fn(prs)
    bad = audit_overflow(prs)
    if bad:
        raise SystemExit("  refusing to write a deck with text outside its panels")
    out = "SIGIL_Edge_Pitch.pptx"
    prs.save(out)
    print(f"  wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
