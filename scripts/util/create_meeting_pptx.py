"""Generate Advisor Meeting Presentation — Meeting with Prof. Ming Jin."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHATS_DIR = os.path.dirname(SCRIPT_DIR)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Colour palette (VT-inspired) ─────────────────────────────────────
VT_MAROON    = RGBColor(0x86, 0x1F, 0x41)
VT_ORANGE    = RGBColor(0xE8, 0x77, 0x22)
DARK_BG      = RGBColor(0x1B, 0x1B, 0x2F)
ACCENT_BLUE  = RGBColor(0x00, 0x7A, 0xCC)
ACCENT_GREEN = RGBColor(0x2E, 0xA0, 0x43)
ACCENT_RED   = RGBColor(0xCC, 0x33, 0x33)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY   = RGBColor(0xCC, 0xCC, 0xCC)
DARK_TEXT     = RGBColor(0x33, 0x33, 0x33)
MED_GRAY     = RGBColor(0x66, 0x66, 0x66)
VERY_LIGHT   = RGBColor(0xF5, 0xF5, 0xF5)
LIGHT_BLUE   = RGBColor(0xE8, 0xF4, 0xFD)
TBL_HDR      = RGBColor(0x00, 0x56, 0x8A)
LIGHT_MAROON = RGBColor(0xF2, 0xE6, 0xEB)
AMBER        = RGBColor(0xFF, 0xA5, 0x00)

# ── Logo paths ────────────────────────────────────────────────────────
LOGO_MAROON = os.path.join(
    CHATS_DIR, "Virginia-Tech-Logos",
    "Registered Virginia Tech Logo 9.12.2023",
    "Horizontal", "Digital RGB", "Maroon",
    "Horizontal_VT_Maroon_RGB.png"
)
LOGO_WHITE = os.path.join(
    CHATS_DIR, "Virginia-Tech-Logos",
    "Registered Virginia Tech Logo 9.12.2023",
    "Horizontal", "Digital RGB", "White",
    "Horizontal_VT_White_RGB.png"
)

HAS_MAROON_LOGO = os.path.exists(LOGO_MAROON)
HAS_WHITE_LOGO  = os.path.exists(LOGO_WHITE)
print(f"Maroon logo found: {HAS_MAROON_LOGO}")
print(f"White  logo found: {HAS_WHITE_LOGO}")

TOTAL_SLIDES = 17
DEEP_PURPLE = RGBColor(0x5B, 0x2C, 0x6F)

# ── Helpers ───────────────────────────────────────────────────────────

def _add_logo(slide, dark_bg=False):
    logo = LOGO_WHITE if dark_bg else LOGO_MAROON
    exists = HAS_WHITE_LOGO if dark_bg else HAS_MAROON_LOGO
    if exists:
        slide.shapes.add_picture(logo, Inches(11.1), Inches(0.15), height=Inches(0.55))
    else:
        tb = slide.shapes.add_textbox(Inches(11.1), Inches(0.15), Inches(2.0), Inches(0.55))
        tf = tb.text_frame; p = tf.paragraphs[0]
        p.text = "[VT LOGO]"
        p.font.size = Pt(12)
        p.font.color.rgb = WHITE if dark_bg else VT_MAROON
        p.font.bold = True


def _set_bg(slide, color=VERY_LIGHT):
    fill = slide.background.fill; fill.solid(); fill.fore_color.rgb = color


def _title_bar(slide, text, y=Inches(0.2), color=VT_MAROON):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), y, prs.slide_width, Inches(0.85))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    tf = s.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
    p.alignment = PP_ALIGN.LEFT


def _subtitle_bar(slide, y=Inches(1.1)):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), y, prs.slide_width, Inches(0.04))
    s.fill.solid(); s.fill.fore_color.rgb = VT_ORANGE; s.line.fill.background()


def _txt(slide, text, l, t, w, h, sz=18, bold=False, color=DARK_TEXT, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(sz); p.font.bold = bold; p.font.color.rgb = color; p.alignment = align
    return tf


def _bullets(slide, items, l, t, w, h, sz=16, color=DARK_TEXT, spacing=6):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(spacing); p.space_before = Pt(2)
        r = p.add_run(); r.text = item; r.font.size = Pt(sz); r.font.color.rgb = color
    return tf


def _card(slide, title, body, l, t, w, h, accent=VT_MAROON):
    c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    c.fill.solid(); c.fill.fore_color.rgb = WHITE
    c.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD); c.line.width = Pt(1)
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, Inches(0.07), h)
    b.fill.solid(); b.fill.fore_color.rgb = accent; b.line.fill.background()
    _txt(slide, title, l + Inches(0.2), t + Inches(0.08), w - Inches(0.35), Inches(0.35),
         sz=16, bold=True, color=accent)
    _txt(slide, body, l + Inches(0.2), t + Inches(0.42), w - Inches(0.35), h - Inches(0.5),
         sz=13, color=MED_GRAY)


def _badge(slide, text, l, t, w=Inches(1.2), h=Inches(0.45), color=ACCENT_RED):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    tf = s.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(11); p.font.color.rgb = WHITE; p.font.bold = True
    p.alignment = PP_ALIGN.CENTER


def _slide_number(slide, num, total):
    _txt(slide, f"{num}/{total}", Inches(12.5), Inches(7.1), Inches(0.8), Inches(0.3),
         sz=10, color=LIGHT_GRAY, align=PP_ALIGN.RIGHT)


def _section_divider(title, subtitle=""):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(sl, VT_MAROON)
    _add_logo(sl, dark_bg=True)
    _txt(sl, title, Inches(1), Inches(2.5), Inches(11), Inches(1.2),
         sz=42, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        _txt(sl, subtitle, Inches(1), Inches(3.8), Inches(10), Inches(0.6),
             sz=20, color=VT_ORANGE, align=PP_ALIGN.LEFT)
    sep = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(4.6), Inches(2.5), Inches(0.04))
    sep.fill.solid(); sep.fill.fore_color.rgb = VT_ORANGE; sep.line.fill.background()
    return sl


def _status_indicator(slide, text, l, t, color):
    """Small colored status pill."""
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, Inches(1.4), Inches(0.35))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    tf = s.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(10); p.font.color.rgb = WHITE; p.font.bold = True
    p.alignment = PP_ALIGN.CENTER


# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
_set_bg(sl, DARK_BG)
_add_logo(sl, dark_bg=True)

_txt(sl, "Advisor Meeting", Inches(0.8), Inches(1.2), Inches(11), Inches(0.9),
     sz=48, bold=True, color=WHITE)
_txt(sl, "Mehmet Koruturk  &  Prof. Ming Jin",
     Inches(0.8), Inches(2.3), Inches(11), Inches(0.8), sz=26, color=LIGHT_GRAY)

sep = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.4), Inches(3), Inches(0.04))
sep.fill.solid(); sep.fill.fore_color.rgb = VT_ORANGE; sep.line.fill.background()

_txt(sl, "Virginia Tech — RoleLab  |  March 2026",
     Inches(0.8), Inches(3.7), Inches(8), Inches(0.4), sz=18, color=MED_GRAY)

# Agenda items as badges
topics = [
    ("PhD Applications", ACCENT_BLUE),
    ("Master's Thesis", ACCENT_GREEN),
    ("SmartGridComm 2026", VT_ORANGE),
    ("VT PEC Conference", VT_MAROON),
]
for i, (topic, col) in enumerate(topics):
    x = Inches(0.8) + Inches(i * 3.0)
    bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(5.0), Inches(2.7), Inches(0.7))
    bx.fill.solid(); bx.fill.fore_color.rgb = col; bx.line.fill.background()
    tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = f"{i+1}. {topic}"
    p.font.size = Pt(15); p.font.color.rgb = WHITE; p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

_slide_number(sl, 1, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2 — AGENDA
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Agenda"); _subtitle_bar(sl)

agenda_items = [
    ("1.", "PhD Application Updates", "TAMU acceptance, MIT/CMU decisions, JHU pending", ACCENT_BLUE),
    ("2.", "Master's Thesis Progress", "SustainRL-Bench: project overview, experiments, timeline", ACCENT_GREEN),
    ("3.", "SmartGridComm 2026 Paper", "Current status, novelty, contributions, next steps", VT_ORANGE),
    ("4.", "VT PEC Annual Conference", "Group presentation need and help request", VT_MAROON),
]

for i, (num, title, desc, col) in enumerate(agenda_items):
    y = Inches(1.5) + Inches(i * 1.35)
    # Number badge
    nb = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), y + Inches(0.05), Inches(0.55), Inches(0.55))
    nb.fill.solid(); nb.fill.fore_color.rgb = col; nb.line.fill.background()
    tf = nb.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = num; p.font.size = Pt(18); p.font.color.rgb = WHITE
    p.font.bold = True; p.alignment = PP_ALIGN.CENTER
    # Title & desc
    _txt(sl, title, Inches(1.4), y, Inches(10), Inches(0.4), sz=22, bold=True, color=col)
    _txt(sl, desc, Inches(1.4), y + Inches(0.45), Inches(10), Inches(0.4), sz=15, color=MED_GRAY)

_slide_number(sl, 2, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 3 — SECTION DIVIDER: PhD Applications
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("1. PhD Application Updates", "Fall 2026 Admissions")
_slide_number(sl, 3, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 4 — PhD Application Status
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "PhD Application Status — Fall 2026"); _subtitle_bar(sl)

# Cards for each school
_card(sl, "Texas A&M University (TAMU)",
      "Accepted to PhD program!\nDepartment: Electrical & Computer Engineering\nThis is currently our strongest option.",
      Inches(0.5), Inches(1.5), Inches(5.8), Inches(1.6), accent=ACCENT_GREEN)
_status_indicator(sl, "ACCEPTED", Inches(5.0), Inches(1.55), ACCENT_GREEN)

_card(sl, "Massachusetts Institute of Technology (MIT)",
      "Application was not successful.\nResult: Rejected",
      Inches(0.5), Inches(3.4), Inches(5.8), Inches(1.1), accent=ACCENT_RED)
_status_indicator(sl, "REJECTED", Inches(5.0), Inches(3.45), ACCENT_RED)

_card(sl, "Carnegie Mellon University (CMU)",
      "Application was not successful.\nResult: Rejected",
      Inches(6.8), Inches(3.4), Inches(5.8), Inches(1.1), accent=ACCENT_RED)
_status_indicator(sl, "REJECTED", Inches(11.3), Inches(3.45), ACCENT_RED)

_card(sl, "Johns Hopkins University (JHU)",
      "Decision still pending.\nExpecting to hear back soon.\nWill update as soon as result arrives.",
      Inches(6.8), Inches(1.5), Inches(5.8), Inches(1.6), accent=AMBER)
_status_indicator(sl, "PENDING", Inches(11.3), Inches(1.55), AMBER)

# Summary note
_txt(sl, "Next Steps: Waiting on JHU decision. Will evaluate offers once all results are in.",
     Inches(0.5), Inches(5.0), Inches(12), Inches(0.5), sz=16, bold=True, color=VT_MAROON)

# Ask
_card(sl, "Discussion Points",
      "- Would appreciate your advice on comparing programs\n"
      "- TAMU vs. JHU (if accepted): research fit, advisor match, funding\n"
      "- Any connections or recommendations for the TAMU ECE department?",
      Inches(0.5), Inches(5.6), Inches(12), Inches(1.5), accent=ACCENT_BLUE)

_slide_number(sl, 4, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 5 — SECTION DIVIDER: Master's Thesis
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("2. Master's Thesis", "SustainRL-Bench: RL Benchmark on SustainGym Environments")
_slide_number(sl, 5, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 6 — Thesis Project Overview
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "SustainRL-Bench — Project Overview"); _subtitle_bar(sl)

_txt(sl, "Benchmarking RL algorithms on three SustainGym energy environments across three experimental axes",
     Inches(0.5), Inches(1.3), Inches(12), Inches(0.5), sz=17, color=MED_GRAY)

# Three environment cards
env_data = [
    ("EV Charging", "ACN-Data / Caltech\n54 stations, 288 timesteps\nProfit - Carbon - Excess", ACCENT_GREEN),
    ("Building HVAC", "EnergyPlus RC Model\nMulti-zone thermal control\nComfort vs Energy tradeoff", ACCENT_BLUE),
    ("Cogeneration", "ONNX Surrogate Plant\n3 Gas Turbines + Steam\nFuel - Ramp - Violations", VT_ORANGE),
]
for i, (env, desc, col) in enumerate(env_data):
    x = Inches(0.5) + Inches(i * 4.2)
    _card(sl, env, desc, x, Inches(1.9), Inches(3.8), Inches(1.6), accent=col)

# Three axes
_txt(sl, "Three Experimental Axes:", Inches(0.5), Inches(3.8), Inches(12), Inches(0.4),
     sz=18, bold=True, color=VT_MAROON)

axes_data = [
    ("Noise Robustness", "Observation / Action / Environment\nnoise decomposition across\n10+ noise levels per env", VT_MAROON),
    ("Safe RL (CMDP)", "PPOLag, CPO, OnCRPO, FOCOPS,\nSACLag via OmniSafe framework\n4 cost limits tested", ACCENT_BLUE),
    ("Multi-Agent RL", "PettingZoo ParallelEnv + Ray RLlib\nPer-component agent decomposition\nShared reward structure", ACCENT_GREEN),
]
for i, (title, desc, col) in enumerate(axes_data):
    x = Inches(0.5) + Inches(i * 4.2)
    _card(sl, title, desc, x, Inches(4.3), Inches(3.8), Inches(1.7), accent=col)

# Algorithms row
_txt(sl, "Algorithms:  PPO  |  SAC  |  TD3  (SB3)    +    PPOLag / CPO / OnCRPO / FOCOPS / SACLag  (OmniSafe)    +    APPO / SAC / PPO  (RLlib)",
     Inches(0.5), Inches(6.3), Inches(12), Inches(0.5), sz=13, color=MED_GRAY)

_slide_number(sl, 6, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 7 — Thesis Current Status & Timeline
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Thesis — Current Status & Next Steps"); _subtitle_bar(sl)

# Experiments status
_txt(sl, "Experiment Completion Status", Inches(0.5), Inches(1.3), Inches(6), Inches(0.4),
     sz=20, bold=True, color=VT_MAROON)

exp_items = [
    ("Standard RL Training (PPO/SAC/TD3)", "All 3 envs: noise sweeps complete (obs, action, env)", ACCENT_GREEN, "DONE"),
    ("Safe RL (OmniSafe)", "5 algos x 3 envs x 4 cost limits = 60 runs", ACCENT_GREEN, "DONE"),
    ("Multi-Agent RL (RLlib)", "EV: APPO | Building: SAC | Cogen: PPO", ACCENT_GREEN, "DONE"),
    ("Model Testing & Evaluation", "Robustness testing across noise conditions", AMBER, "IN PROGRESS"),
    ("Results Analysis & Plots", "Training curves, bar charts, heatmaps, Pareto fronts", AMBER, "IN PROGRESS"),
]

for i, (title, desc, col, status) in enumerate(exp_items):
    y = Inches(1.8) + Inches(i * 0.85)
    _txt(sl, f"  {title}", Inches(0.5), y, Inches(7), Inches(0.35), sz=15, bold=True, color=DARK_TEXT)
    _txt(sl, f"     {desc}", Inches(0.5), y + Inches(0.32), Inches(7), Inches(0.3), sz=12, color=MED_GRAY)
    _status_indicator(sl, status, Inches(7.8), y + Inches(0.05), col)

# Writing status
_txt(sl, "Thesis Writing Status", Inches(0.5), Inches(5.8), Inches(6), Inches(0.4),
     sz=20, bold=True, color=VT_MAROON)

writing_items = [
    "Ch.1 Introduction — drafted (research questions, objectives)",
    "Ch.2 Literature Review — drafted",
    "Ch.3 Preliminaries — drafted (MDP, algorithms, CMDP)",
    "Ch.4 Methodology — drafted (SustainRL-Bench framework)",
    "Ch.5 Results & Ch.6 Conclusion — pending final experiments",
]
_bullets(sl, writing_items, Inches(0.5), Inches(6.2), Inches(7), Inches(1.5), sz=12, color=DARK_TEXT)

# Right side: timeline card
_card(sl, "Plan & Timeline",
      "1. Complete remaining experiments (testing)\n"
      "2. Finalize results analysis & generate all figures\n"
      "3. Write Ch.5 (Results) & Ch.6 (Conclusion)\n"
      "4. Full document review & polish\n"
      "5. Send to Prof. Jin for review\n\n"
      "Goal: Submit complete draft for your review ASAP",
      Inches(9.2), Inches(1.3), Inches(3.8), Inches(3.5), accent=VT_MAROON)

# Key ask
_card(sl, "Key Message",
      "Experiments are largely complete.\n"
      "Will finish writing and send the full\n"
      "thesis draft to you for review and\n"
      "feedback as soon as possible.",
      Inches(9.2), Inches(5.2), Inches(3.8), Inches(2.0), accent=ACCENT_BLUE)

_slide_number(sl, 7, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 8 — SECTION DIVIDER: SmartGridComm
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("3. SmartGridComm 2026", "IEEE SmartGridComm  |  Deadline: June 19, 2026")
_slide_number(sl, 8, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 9 — SmartGridComm: Current Status
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "SmartGridComm 2026 — Paper Status"); _subtitle_bar(sl)

_txt(sl, "Target: IEEE SmartGridComm 2026  |  Deadline: June 19, 2026  |  Format: 6 pages IEEE",
     Inches(0.5), Inches(1.3), Inches(12), Inches(0.4), sz=15, color=MED_GRAY)

# Status overview
_txt(sl, "Paper Preparation Status", Inches(0.5), Inches(1.8), Inches(6), Inches(0.4),
     sz=20, bold=True, color=VT_MAROON)

status_items = [
    ("Experimental Infrastructure", "Complete — all training scripts, envs, SLURM pipelines ready", ACCENT_GREEN, "READY"),
    ("Standard RL Experiments", "All 3 algos x 3 envs x noise sweeps trained", ACCENT_GREEN, "DONE"),
    ("Safe RL Experiments", "OmniSafe: 5 algos x 3 envs x 4 cost limits", ACCENT_GREEN, "DONE"),
    ("MARL Experiments", "Ray RLlib: per-env agent decomposition trained", ACCENT_GREEN, "DONE"),
    ("Non-RL Baselines", "Rule-based & optimization baselines needed for comparison", ACCENT_RED, "TODO"),
    ("Paper Writing", "Outline ready, need to draft 6-page manuscript", AMBER, "IN PROGRESS"),
]

for i, (title, desc, col, status) in enumerate(status_items):
    y = Inches(2.3) + Inches(i * 0.75)
    _txt(sl, f"  {title}", Inches(0.5), y, Inches(8), Inches(0.3), sz=14, bold=True, color=DARK_TEXT)
    _txt(sl, f"     {desc}", Inches(0.5), y + Inches(0.28), Inches(8), Inches(0.3), sz=12, color=MED_GRAY)
    _status_indicator(sl, status, Inches(8.8), y + Inches(0.02), col)

# Key gap highlight
_card(sl, "Critical Gap: Non-RL Baselines",
      "Reviewers will ask: 'Why use RL at all?'\n"
      "Need: Do-Nothing, Rule-Based, and ideally\n"
      "one optimization baseline (MPC or LP)\n"
      "for each environment.",
      Inches(10.5), Inches(2.3), Inches(2.5), Inches(2.0), accent=ACCENT_RED)

# Timeline
_card(sl, "Paper Timeline",
      "Mar-Apr: Complete baselines + testing\n"
      "Apr-May: Core experiments (5 seeds)\n"
      "May-Jun: Analysis & writing\n"
      "Jun 12-19: Final review & submit",
      Inches(10.5), Inches(4.6), Inches(2.5), Inches(2.0), accent=VT_MAROON)

_slide_number(sl, 9, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 10 — SmartGridComm: Novelty & Contributions
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "SmartGridComm — Novelty & Contributions"); _subtitle_bar(sl)

# Left column: What IS novel
_txt(sl, "Novel Contributions (What We Can Claim)", Inches(0.5), Inches(1.3), Inches(6), Inches(0.4),
     sz=18, bold=True, color=ACCENT_GREEN)

novel_items = [
    ("C1: First Unified Multi-Paradigm Benchmark",
     "Standard RL + Safe RL + MARL across EV Charging,\n"
     "Building HVAC, and Cogeneration under one protocol",
     "HIGH"),
    ("C2: Systematic Noise Robustness Decomposition",
     "Three independent noise channels (obs/action/env)\n"
     "characterized across three energy domains",
     "HIGH"),
    ("C3: First Safe RL Benchmark for EV & Cogen",
     "Novel CMDP formulations with domain-specific cost\n"
     "signals (excess charge, power, constraint violations)",
     "MED-HIGH"),
    ("C4: Cross-Paradigm Comparison",
     "Standard RL vs Safe RL vs MARL tradeoff analysis\n"
     "on the same environments and metrics",
     "HIGH"),
    ("C5: Open-Source Reproducible Suite",
     "Full code, SLURM scripts, standardized evaluation\n"
     "protocol for community reproducibility",
     "MEDIUM"),
]

for i, (title, desc, conf) in enumerate(novel_items):
    y = Inches(1.8) + Inches(i * 1.05)
    col = ACCENT_GREEN if conf == "HIGH" else (ACCENT_BLUE if "MED" in conf else MED_GRAY)
    _txt(sl, title, Inches(0.5), y, Inches(7), Inches(0.3), sz=13, bold=True, color=VT_MAROON)
    _txt(sl, desc, Inches(0.5), y + Inches(0.3), Inches(6.5), Inches(0.6), sz=11, color=MED_GRAY)
    _badge(sl, f"Conf: {conf}", Inches(7.0), y + Inches(0.05), w=Inches(1.3), h=Inches(0.3), color=col)

# Right column: What is NOT novel
_txt(sl, "Not Novel (Don't Over-Claim)", Inches(8.8), Inches(1.3), Inches(4), Inches(0.4),
     sz=18, bold=True, color=ACCENT_RED)

not_novel = [
    "Environments themselves (SustainGym, NeurIPS 2023)",
    "Individual algorithms (PPO, SAC, TD3, OmniSafe algos)",
    "PettingZoo/RLlib frameworks",
    "Basic noise injection techniques",
]
_bullets(sl, not_novel, Inches(8.8), Inches(1.8), Inches(4), Inches(2.0), sz=13, color=MED_GRAY, spacing=8)

# Key positioning
_card(sl, "Positioning Statement",
      "We don't claim new algorithms or environments.\n"
      "Our contribution is the BENCHMARK PROTOCOL:\n"
      "the first systematic, multi-paradigm evaluation\n"
      "of RL in sustainable energy systems, with\n"
      "noise robustness as a unique axis.",
      Inches(8.8), Inches(4.0), Inches(4.2), Inches(2.5), accent=VT_MAROON)

_slide_number(sl, 10, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 11 — Paper Titles & Stories (1/2): Options A & B
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Possible Paper Titles & Main Stories (1/2)"); _subtitle_bar(sl)

_txt(sl, "Which framing gives us the strongest SmartGridComm paper? Key insights from our experiments below.",
     Inches(0.5), Inches(1.25), Inches(12), Inches(0.4), sz=14, color=MED_GRAY)

# ── OPTION A ──────────────────────────────────────────────────────────
_badge(sl, "OPTION A", Inches(0.5), Inches(1.7), w=Inches(1.1), h=Inches(0.35), color=VT_MAROON)
_badge(sl, "RECOMMENDED", Inches(1.75), Inches(1.7), w=Inches(1.6), h=Inches(0.35), color=ACCENT_GREEN)

_txt(sl, '"SustainRL-Bench: A Multi-Paradigm Benchmark for\n'
         'Reinforcement Learning in Sustainable Energy Systems"',
     Inches(0.5), Inches(2.1), Inches(8), Inches(0.55), sz=15, bold=True, color=VT_MAROON)

_txt(sl, "Main Story: We present the first benchmark that systematically evaluates standard RL, "
     "safe (constrained) RL, and multi-agent RL on the same three energy environments under a unified "
     "protocol. Our preliminary results already reveal several non-obvious findings: (1) noise vulnerability "
     "is highly asymmetric across channels — Building HVAC suffers 256% performance loss under observation "
     "noise at 0.4 but is nearly immune to action noise (<5% variance across all levels), while EV charging "
     "counter-intuitively improves by 69-116% under moderate observation noise, suggesting a regularization "
     "effect unique to high-dimensional charging coordination; (2) safe RL algorithms exhibit a wide "
     "performance-safety spectrum — IPO achieves near-zero constraint cost on EV charging (10^-33) with "
     "reward >9.0, but shows 100x cost variance across seeds, whereas PCPO provides stable constraint "
     "satisfaction (~3.0) on Building at the expense of reward; (3) cross-paradigm comparison reveals "
     "that the optimal RL paradigm is strongly environment-dependent. The paper's anchor claim is the "
     "benchmark protocol itself: no prior work covers this 3-env x 3-paradigm x 3-noise intersection. "
     "This is the broadest scope, telling the full thesis story in 6 pages.",
     Inches(0.5), Inches(2.65), Inches(8), Inches(2.1), sz=11, color=DARK_TEXT)

_card(sl, "Strengths",
      "- Maximum novelty surface (3x3x3)\n"
      "- Named benchmark aids citation\n"
      "- Directly extends SustainGym (NeurIPS 2023)\n"
      "- 735+ training runs already done",
      Inches(8.5), Inches(1.7), Inches(4.5), Inches(1.3), accent=ACCENT_GREEN)

_card(sl, "Risks",
      "- 6 pages tight for 3 axes\n"
      "- 'Broad but shallow' concern\n"
      "- Needs non-RL baselines",
      Inches(8.5), Inches(3.15), Inches(4.5), Inches(1.1), accent=AMBER)

# ── OPTION B ──────────────────────────────────────────────────────────
_badge(sl, "OPTION B", Inches(0.5), Inches(5.0), w=Inches(1.1), h=Inches(0.35), color=ACCENT_BLUE)

_txt(sl, '"How Robust is Energy RL? Decomposing Observation, Action,\n'
         'and Environment Noise Across Sustainable Energy Domains"',
     Inches(0.5), Inches(5.4), Inches(8), Inches(0.55), sz=15, bold=True, color=ACCENT_BLUE)

_txt(sl, "Main Story: We introduce a structured noise decomposition framework for energy RL that independently "
     "varies observation, action, and environment noise across three SustainGym domains. Our data reveals "
     "a striking asymmetry: in Building HVAC, observation noise is catastrophic (53% loss at sigma=0.10, "
     "256% at 0.40) while action noise has <5% impact across all levels — a finding with direct implications "
     "for sensor investment vs. actuator precision. In EV charging, the relationship inverts: moderate observation "
     "noise (sigma=0.05-0.10) produces a 69-116% performance improvement, suggesting that noise acts as "
     "implicit regularization for the 54-dimensional pilot signal coordination. In Cogeneration, environment "
     "noise (ONNX model uncertainty up to 3x) shows negligible reward impact, indicating the surrogate "
     "model's dynamics are locally flat. We also find a non-monotonic pattern at low noise (sigma=0.01 "
     "improves Building PPO by 22% over clean baseline), challenging the assumption that any noise is "
     "harmful. SAC outperforms PPO at baseline (-22.81 vs -30.99 in Building) but shows similar "
     "degradation slopes, suggesting robustness is architecture-independent. This framing has the strongest "
     "methodological novelty and is the most actionable for practitioners deploying RL in real energy systems.",
     Inches(0.5), Inches(5.9), Inches(8), Inches(1.8), sz=11, color=DARK_TEXT)

_card(sl, "Strengths",
      "- Strongest methodology novelty\n"
      "- Counter-intuitive findings (noise helps)\n"
      "- Most actionable for practitioners\n"
      "- Data-rich: 10+ noise levels x 3 envs",
      Inches(8.5), Inches(5.0), Inches(4.5), Inches(1.3), accent=ACCENT_GREEN)

_card(sl, "Risks",
      "- Drops safe RL & MARL axes\n"
      "- Needs complete env noise sweeps\n"
      "- Building-heavy data, others thinner",
      Inches(8.5), Inches(6.45), Inches(4.5), Inches(1.1), accent=AMBER)

_slide_number(sl, 11, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 12 — Paper Titles & Stories (2/2): Options C & D
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Possible Paper Titles & Main Stories (2/2)"); _subtitle_bar(sl)

# ── OPTION C ──────────────────────────────────────────────────────────
_badge(sl, "OPTION C", Inches(0.5), Inches(1.3), w=Inches(1.1), h=Inches(0.35), color=VT_ORANGE)

_txt(sl, '"Safe Reinforcement Learning for Sustainable Energy Systems:\n'
         'A Constrained Optimization Benchmark Across Three Domains"',
     Inches(0.5), Inches(1.7), Inches(8), Inches(0.55), sz=15, bold=True, color=VT_ORANGE)

_txt(sl, "Main Story: We present the first CMDP-based safe RL benchmark for energy systems, evaluating "
     "five algorithms (PPOLag, CPO, OnCRPO, FOCOPS, SACLag) across EV charging, building HVAC, and "
     "cogeneration with domain-specific cost signals. Our 177 safe RL experiments reveal a rich "
     "performance-safety landscape: on EV charging, IPO can achieve near-perfect constraint satisfaction "
     "(cost ~ 10^-33) with strong reward (>9.0), but exhibits extreme seed sensitivity (100x cost "
     "variance), while CPO provides more consistent tradeoffs (reward 5.02 +/- 1.12, cost 7.97 +/- 5.79 "
     "under tight constraints). On Building, SACLag achieves zero constraint cost but at severe reward "
     "penalty (-537 vs -22.81 unconstrained SAC), revealing that Lagrangian off-policy methods can be "
     "overly conservative when energy and comfort objectives are orthogonal. PCPO shows the best "
     "balance on Building with stable cost (~3.0) across 5 seeds. The Pareto frontiers across four "
     "cost limits (1, 5, 25, 1000) demonstrate that constraint tightness interacts differently with "
     "each algorithm family — a finding absent from the robotics-focused safe RL literature. This framing "
     "offers the cleanest novelty claim (no prior CMDP benchmarks for EV/cogen) and the deepest analysis "
     "possible within 6 pages.",
     Inches(0.5), Inches(2.25), Inches(8), Inches(2.4), sz=11, color=DARK_TEXT)

_card(sl, "Strengths",
      "- Cleanest novelty (first for EV/cogen)\n"
      "- 177 experiments = statistical depth\n"
      "- Pareto analysis is visually compelling\n"
      "- Deep 6-page treatment possible",
      Inches(8.5), Inches(1.3), Inches(4.5), Inches(1.4), accent=ACCENT_GREEN)

_card(sl, "Risks",
      "- Drops noise + MARL axes\n"
      "- Cogen safe RL has scaling issues\n"
      "- Reviewers may want robustness too",
      Inches(8.5), Inches(2.85), Inches(4.5), Inches(1.1), accent=AMBER)

# ── OPTION D ──────────────────────────────────────────────────────────
_badge(sl, "OPTION D", Inches(0.5), Inches(4.9), w=Inches(1.1), h=Inches(0.35), color=DEEP_PURPLE)

_txt(sl, '"From Single-Agent to Multi-Agent RL in Energy Systems:\n'
         'A Cross-Domain Comparison on EV Charging, HVAC, and Cogeneration"',
     Inches(0.5), Inches(5.3), Inches(8), Inches(0.55), sz=15, bold=True, color=DEEP_PURPLE)

_txt(sl, "Main Story: We compare centralized single-agent RL against per-component multi-agent "
     "decomposition across three environments with vastly different agent structures: 54 agents "
     "(EV, per-station), 3-6 agents (Building, per-zone), and 4 agents (Cogen, per-turbine via "
     "GT1/GT2/GT3/ST decomposition). The key question is whether natural physical decomposition "
     "translates to MARL benefits. Our results suggest the answer is environment-dependent: the "
     "54-agent EV charging with shared-reward-divided-by-N creates a challenging credit assignment "
     "problem, while the 4-agent cogen decomposition with per-turbine cost attribution shows cleaner "
     "learning dynamics. Multi-agent cogeneration control is entirely novel in the literature. However, "
     "this framing has the thinnest experimental support — MARL iterations are limited (Cogen: only 100 "
     "iterations) and algorithm diversity is low (one algo per env). This option requires the most "
     "additional work to be convincing.",
     Inches(0.5), Inches(5.8), Inches(8), Inches(1.8), sz=11, color=DARK_TEXT)

_card(sl, "Strengths",
      "- Novel application (MARL cogen)\n"
      "- Cross-domain MARL comparison is rare\n"
      "- Natural physical decomposition angle",
      Inches(8.5), Inches(4.9), Inches(4.5), Inches(1.1), accent=ACCENT_GREEN)

_card(sl, "Risks",
      "- Thinnest data (100 iters for Cogen)\n"
      "- Only 1 algo per env\n"
      "- Most additional work needed",
      Inches(8.5), Inches(6.15), Inches(4.5), Inches(1.1), accent=ACCENT_RED)

_slide_number(sl, 12, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 13 — Preliminary Abstract Drafts (Option A)
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Preliminary Abstract Drafts"); _subtitle_bar(sl)

_badge(sl, "OPTION A", Inches(0.5), Inches(1.3), w=Inches(1.1), h=Inches(0.35), color=VT_MAROON)
_badge(sl, "RECOMMENDED", Inches(1.75), Inches(1.3), w=Inches(1.6), h=Inches(0.35), color=ACCENT_GREEN)
_txt(sl, "SustainRL-Bench: A Multi-Paradigm Benchmark for RL in Sustainable Energy Systems",
     Inches(3.5), Inches(1.28), Inches(9), Inches(0.4), sz=14, bold=True, color=VT_MAROON)

abstract_a = (
    "Reinforcement learning (RL) is increasingly applied to energy system control, yet evaluations "
    "remain fragmented across isolated environments, algorithm families, and operational settings. "
    "We present SustainRL-Bench, the first benchmark that systematically evaluates standard RL "
    "(PPO, SAC, TD3), constrained safe RL (PPOLag, CPO, OnCRPO, FOCOPS, SACLag), and cooperative "
    "multi-agent RL across three diverse SustainGym energy domains: electric vehicle charging "
    "(54-station network), building HVAC (multi-zone RC thermal model), and cogeneration plant "
    "dispatch (ONNX surrogate with three gas turbines). Our evaluation protocol introduces a "
    "structured noise decomposition framework that independently varies observation, action, and "
    "environment uncertainty to assess deployment robustness.\n\n"
    "Across 735+ training runs spanning 10+ noise levels, 5 safe RL algorithms at 4 constraint "
    "tightness levels, and 3 multi-agent decompositions, we uncover several findings that challenge "
    "prevailing assumptions: (1) noise vulnerability is channel- and domain-specific — building HVAC "
    "policies degrade by 256% under observation noise but remain stable (<5% variance) under action "
    "noise, while EV charging policies counter-intuitively improve by 69-116% under moderate "
    "observation noise due to implicit regularization; (2) safe RL algorithms exhibit a 100x cost "
    "variance across seeds on identical configurations, with interior-point methods achieving "
    "near-zero constraint violation in some runs but diverging in others; (3) low-level noise "
    "(sigma=0.01) can improve over clean baselines by 22%, revealing non-monotonic robustness "
    "profiles. These findings provide actionable guidance for practitioners selecting RL paradigms "
    "and noise mitigation strategies for energy system deployment. We release SustainRL-Bench as an "
    "open-source benchmark suite."
)

# Abstract in a light box
ab = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                         Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.3))
ab.fill.solid(); ab.fill.fore_color.rgb = WHITE
ab.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD); ab.line.width = Pt(1)
_txt(sl, abstract_a, Inches(0.8), Inches(1.9), Inches(11.8), Inches(5.1),
     sz=12, color=DARK_TEXT)

_txt(sl, "~230 words  |  Key selling points: breadth (3x3x3), counter-intuitive noise findings, "
     "safe RL seed variance, non-monotonic robustness",
     Inches(0.5), Inches(7.15), Inches(12), Inches(0.3), sz=10, color=MED_GRAY)

_slide_number(sl, 13, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 14 — Preliminary Abstract Draft (Option B)
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Preliminary Abstract Drafts (cont.)"); _subtitle_bar(sl)

_badge(sl, "OPTION B", Inches(0.5), Inches(1.3), w=Inches(1.1), h=Inches(0.35), color=ACCENT_BLUE)
_txt(sl, "How Robust is Energy RL? Decomposing Noise Across Sustainable Energy Domains",
     Inches(1.75), Inches(1.28), Inches(11), Inches(0.4), sz=14, bold=True, color=ACCENT_BLUE)

abstract_b = (
    "Deep reinforcement learning has demonstrated promising results for energy system control, "
    "yet the robustness of learned policies under realistic deployment uncertainty remains poorly "
    "understood. We introduce a structured noise decomposition framework that independently "
    "characterizes policy sensitivity along three orthogonal channels — observation noise (sensor "
    "uncertainty), action noise (actuator imprecision), and environment noise (sim-to-real dynamics "
    "gap) — across three SustainGym energy domains: electric vehicle charging, building HVAC, and "
    "cogeneration.\n\n"
    "Our analysis across PPO, SAC, and TD3 at 10+ noise levels per environment reveals that noise "
    "vulnerability profiles are strikingly domain-specific and asymmetric across channels. In "
    "building HVAC, observation noise causes catastrophic degradation (53% at sigma=0.10, 256% at "
    "sigma=0.40), while action noise has negligible impact (<5% variance from sigma=0.0 to 0.40) — "
    "implying that sensor accuracy, not actuator precision, should be the investment priority for "
    "HVAC deployment. In EV charging, we observe the opposite: moderate observation noise improves "
    "policy performance by 69-116%, acting as implicit regularization for the 54-dimensional "
    "coordination problem. In cogeneration, environment noise up to 3x baseline (including 45% ONNX "
    "model output perturbation) produces minimal reward impact, indicating robust local dynamics. "
    "We further identify a non-monotonic pattern: noise at sigma=0.01 improves Building PPO by 22% "
    "over the clean baseline, suggesting that optimal deployment may require deliberate noise "
    "injection. SAC consistently outperforms PPO at baseline but shows comparable degradation slopes, "
    "indicating that robustness is architecture-independent. Our findings provide the first "
    "channel-specific noise sensitivity map for energy RL and offer concrete deployment guidelines "
    "for practitioners."
)

ab = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                         Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.3))
ab.fill.solid(); ab.fill.fore_color.rgb = WHITE
ab.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD); ab.line.width = Pt(1)
_txt(sl, abstract_b, Inches(0.8), Inches(1.9), Inches(11.8), Inches(5.1),
     sz=12, color=DARK_TEXT)

_txt(sl, "~250 words  |  Key selling points: noise-as-regularization discovery, asymmetric channel "
     "vulnerability, non-monotonic profiles, practitioner deployment guidelines",
     Inches(0.5), Inches(7.15), Inches(12), Inches(0.3), sz=10, color=MED_GRAY)

_slide_number(sl, 14, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 15 — Preliminary Abstract Draft (Option C)
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Preliminary Abstract Drafts (cont.)"); _subtitle_bar(sl)

_badge(sl, "OPTION C", Inches(0.5), Inches(1.3), w=Inches(1.1), h=Inches(0.35), color=VT_ORANGE)
_txt(sl, "Safe RL for Sustainable Energy: A Constrained Optimization Benchmark",
     Inches(1.75), Inches(1.28), Inches(11), Inches(0.4), sz=14, bold=True, color=VT_ORANGE)

abstract_c = (
    "Reinforcement learning controllers for energy systems must satisfy hard operational constraints "
    "— network capacity limits in EV charging, thermal comfort bounds in buildings, and equipment "
    "safety envelopes in cogeneration plants — yet most RL benchmarks optimize reward without "
    "explicit constraint handling. We present the first systematic constrained RL benchmark for "
    "sustainable energy systems, formulating each of three SustainGym environments as a Constrained "
    "Markov Decision Process (CMDP) with physically motivated cost signals: excess charging "
    "violations scaled by network capacity (EV), energy consumption orthogonal to the comfort "
    "objective (Building), and dynamic equipment constraint violations plus supply non-delivery "
    "(Cogeneration).\n\n"
    "We evaluate five safe RL algorithms — PPOLag, CPO, OnCRPO, FOCOPS, and SACLag — across four "
    "constraint tightness levels (cost limits 1, 5, 25, 1000) totaling 177 training runs. Our "
    "results reveal that the reward-safety Pareto landscape varies dramatically across domains and "
    "algorithms: interior-point methods (IPO) achieve near-perfect constraint satisfaction "
    "(cost ~ 10^-33) on EV charging with competitive reward (>9.0) in favorable runs, but exhibit "
    "extreme instability (100x cost variance across seeds); projection-based methods (PCPO) provide "
    "the most stable tradeoffs on Building with consistent cost (~3.0 +/- 0.05 across 5 seeds); "
    "while Lagrangian off-policy methods (SACLag) can become overly conservative, achieving zero "
    "cost at 24x reward penalty on Building. Under loose constraints (limit=1000), Lagrangian "
    "multiplier explosion degrades all Lagrangian methods without explicit upper bounding. These "
    "findings demonstrate that safe RL is viable for energy systems but requires careful algorithm "
    "selection matched to domain-specific constraint structures."
)

ab = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                         Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.3))
ab.fill.solid(); ab.fill.fore_color.rgb = WHITE
ab.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD); ab.line.width = Pt(1)
_txt(sl, abstract_c, Inches(0.8), Inches(1.9), Inches(11.8), Inches(5.1),
     sz=12, color=DARK_TEXT)

_txt(sl, "~240 words  |  Key selling points: first CMDP for EV/cogen, seed variance as finding, "
     "Lagrangian instability insight, projection vs. interior-point tradeoff",
     Inches(0.5), Inches(7.15), Inches(12), Inches(0.3), sz=10, color=MED_GRAY)

_slide_number(sl, 15, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 16 — SECTION DIVIDER: VT PEC
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("4. VT PEC Annual Conference", "Group Presentation Opportunity")
_slide_number(sl, 16, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 17 — VT PEC Conference
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "VT PEC Annual Conference — Group Presentation"); _subtitle_bar(sl)

_txt(sl, "Virginia Tech Power & Energy Center (PEC) Annual Conference",
     Inches(0.5), Inches(1.3), Inches(12), Inches(0.5), sz=20, bold=True, color=VT_MAROON)

_txt(sl, "Our research group needs to present at this year's PEC Annual Conference.",
     Inches(0.5), Inches(1.9), Inches(12), Inches(0.5), sz=17, color=DARK_TEXT)

# Situation card
_card(sl, "Current Situation",
      "- We need at least one presentation from our group (RoleLab)\n"
      "- The PEC conference is an important venue for visibility within VT\n"
      "- Good opportunity to showcase our RL-for-energy research\n"
      "- My SustainRL-Bench work could be a strong candidate for this",
      Inches(0.5), Inches(2.6), Inches(5.8), Inches(2.2), accent=ACCENT_BLUE)

# What I need
_card(sl, "What I Need",
      "- Guidance on the presentation format & requirements\n"
      "- Help deciding what to present (thesis work? SmartGridComm subset?)\n"
      "- Feedback on whether the timing works with my other deadlines\n"
      "- Any coordination needed with other group members",
      Inches(6.8), Inches(2.6), Inches(5.8), Inches(2.2), accent=VT_MAROON)

# Action items
_txt(sl, "Action Items", Inches(0.5), Inches(5.3), Inches(6), Inches(0.4),
     sz=20, bold=True, color=VT_MAROON)

action_items = [
    "Confirm whether I should prepare the group presentation",
    "Get details on PEC conference date, format, and abstract deadline",
    "Discuss scope: full thesis overview vs. focused SmartGridComm results",
    "Coordinate with other RoleLab members if needed",
]
_bullets(sl, action_items, Inches(0.5), Inches(5.7), Inches(12), Inches(1.5), sz=15, color=DARK_TEXT, spacing=6)

_slide_number(sl, 17, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════
out_path = os.path.join(SCRIPT_DIR, "advisor_meeting_march2026.pptx")
prs.save(out_path)
print(f"\nPresentation saved to: {out_path}")
print(f"Total slides: {TOTAL_SLIDES}")