"""Generate SmartGridComm Deep Dive Presentation — 30+ slides with VT logo."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Colour palette (VT-inspired + professional) ────────────────────────
VT_MAROON   = RGBColor(0x86, 0x1F, 0x41)
VT_ORANGE   = RGBColor(0xE8, 0x77, 0x22)
DARK_BG     = RGBColor(0x1B, 0x1B, 0x2F)
ACCENT_BLUE = RGBColor(0x00, 0x7A, 0xCC)
ACCENT_GREEN= RGBColor(0x2E, 0xA0, 0x43)
ACCENT_RED  = RGBColor(0xCC, 0x33, 0x33)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xCC, 0xCC, 0xCC)
DARK_TEXT    = RGBColor(0x33, 0x33, 0x33)
MED_GRAY    = RGBColor(0x66, 0x66, 0x66)
VERY_LIGHT  = RGBColor(0xF5, 0xF5, 0xF5)
LIGHT_BLUE  = RGBColor(0xE8, 0xF4, 0xFD)
TBL_HDR     = RGBColor(0x00, 0x56, 0x8A)
LIGHT_MAROON= RGBColor(0xF2, 0xE6, 0xEB)

# ── Logo paths ──────────────────────────────────────────────────────────
LOGO_MAROON = os.path.join(
    SCRIPT_DIR,
    "Virginia-Tech-Logos",
    "Registered Virginia Tech Logo 9.12.2023",
    "Horizontal", "Digital RGB", "Maroon",
    "Horizontal_VT_Maroon_RGB.png"
)
LOGO_WHITE = os.path.join(
    SCRIPT_DIR,
    "Virginia-Tech-Logos",
    "Registered Virginia Tech Logo 9.12.2023",
    "Horizontal", "Digital RGB", "White",
    "Horizontal_VT_White_RGB.png"
)

HAS_MAROON_LOGO = os.path.exists(LOGO_MAROON)
HAS_WHITE_LOGO  = os.path.exists(LOGO_WHITE)
print(f"Maroon logo found: {HAS_MAROON_LOGO} ({LOGO_MAROON})")
print(f"White  logo found: {HAS_WHITE_LOGO}  ({LOGO_WHITE})")

# ── Helper functions ────────────────────────────────────────────────────

def _add_logo(slide, dark_bg=False):
    """Place VT logo in top-right corner."""
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
    """Thin accent bar under the title bar."""
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
    """Dark maroon section divider slide."""
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


TOTAL_SLIDES = 35  # approximate

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
_set_bg(sl, DARK_BG)
_add_logo(sl, dark_bg=True)

_txt(sl, "SustainRL-Bench", Inches(0.8), Inches(1.0), Inches(11), Inches(0.9),
     sz=48, bold=True, color=WHITE)
_txt(sl, "A Multi-Paradigm Benchmark for Reinforcement\nLearning in Sustainable Energy Systems",
     Inches(0.8), Inches(2.0), Inches(11), Inches(1.2), sz=26, color=LIGHT_GRAY)
sep = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.5), Inches(3), Inches(0.04))
sep.fill.solid(); sep.fill.fore_color.rgb = VT_ORANGE; sep.line.fill.background()
_txt(sl, "Mehmet Koruturk\nVirginia Tech — RoleLab", Inches(0.8), Inches(3.8), Inches(5), Inches(0.8),
     sz=20, color=LIGHT_GRAY)
_txt(sl, "Target: IEEE SmartGridComm 2026  |  Deadline: June 19, 2026",
     Inches(0.8), Inches(5.0), Inches(8), Inches(0.4), sz=15, color=MED_GRAY)

# Three env labels
for i, (env, col) in enumerate([("EV Charging", ACCENT_GREEN), ("Building HVAC", ACCENT_BLUE), ("Cogeneration", VT_ORANGE)]):
    x = Inches(8.0) + Inches(i * 1.7)
    bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(4.8), Inches(1.5), Inches(0.8))
    bx.fill.solid(); bx.fill.fore_color.rgb = col; bx.line.fill.background()
    tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = env; p.font.size = Pt(13); p.font.color.rgb = WHITE
    p.font.bold = True; p.alignment = PP_ALIGN.CENTER

_slide_number(sl, 1, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2 — AGENDA
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Agenda"); _subtitle_bar(sl)

sections = [
    "1.  Project Overview & Research Questions",
    "2.  Three SustainGym Environments",
    "3.  Noise Robustness Framework",
    "4.  Literature Positioning & Gap",
    "5.  Novelty & Contributions (Deep Dive)",
    "6.  Gap Analysis (Detailed Audit)",
    "7.  Strategic Execution Roadmap",
    "8.  Baseline Implementation Plan",
    "9.  Experiment Design & Protocols",
    "10. Evaluation & Reporting Framework",
    "11. Paper Writing Blueprint",
    "12. Risk Assessment & Contingencies",
    "13. Timeline & Next Steps",
]
for i, s in enumerate(sections):
    _txt(sl, s, Inches(1.0), Inches(1.5) + Inches(i * 0.43), Inches(10), Inches(0.4),
         sz=17, color=DARK_TEXT if i < 4 else VT_MAROON, bold=(i >= 4))
_slide_number(sl, 2, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 3 — PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Project Overview"); _subtitle_bar(sl)

_txt(sl, "Core Research Question", Inches(0.5), Inches(1.4), Inches(6), Inches(0.4),
     sz=22, bold=True, color=VT_MAROON)
_txt(sl, "How do different RL paradigms (standard, safe, multi-agent)\nperform across diverse sustainable energy domains under\nrealistic uncertainty conditions?",
     Inches(0.5), Inches(1.9), Inches(7), Inches(1.2), sz=18, color=DARK_TEXT)

_txt(sl, "Experimental Matrix", Inches(0.5), Inches(3.5), Inches(12), Inches(0.4),
     sz=20, bold=True, color=VT_MAROON)

# 3x3 grid
paradigms = [("Standard RL", "(PPO, SAC via SB3)"), ("Safe RL", "(CMDP via OmniSafe)"), ("Multi-Agent RL", "(RLlib + PettingZoo)")]
envs_list = [("EV Charging", "54 stations"), ("Building HVAC", "RC thermal model"), ("Cogeneration", "ONNX surrogate")]
p_colors = [ACCENT_BLUE, VT_ORANGE, ACCENT_GREEN]

for j, ((par, detail), col) in enumerate(zip(paradigms, p_colors)):
    y = Inches(4.1) + Inches(j * 1.05)
    bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), y, Inches(3.2), Inches(0.85))
    bx.fill.solid(); bx.fill.fore_color.rgb = col; bx.line.fill.background()
    tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True; tf.margin_left = Inches(0.15)
    p = tf.paragraphs[0]; p.text = f"{par}"; p.font.size = Pt(14); p.font.color.rgb = WHITE; p.font.bold = True
    p2 = tf.add_paragraph(); p2.text = detail; p2.font.size = Pt(11); p2.font.color.rgb = RGBColor(0xDD,0xDD,0xDD)

    for i, (env, sub) in enumerate(envs_list):
        x = Inches(3.8) + Inches(i * 3.2)
        bx2 = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.0), Inches(0.85))
        bx2.fill.solid(); bx2.fill.fore_color.rgb = WHITE
        bx2.line.color.rgb = col; bx2.line.width = Pt(2)
        tf2 = bx2.text_frame; tf2.vertical_anchor = MSO_ANCHOR.MIDDLE; tf2.word_wrap = True
        p = tf2.paragraphs[0]; p.text = env; p.font.size = Pt(13); p.font.color.rgb = DARK_TEXT
        p.font.bold = True; p.alignment = PP_ALIGN.CENTER
        p2 = tf2.add_paragraph(); p2.text = sub; p2.font.size = Pt(11); p2.font.color.rgb = MED_GRAY
        p2.alignment = PP_ALIGN.CENTER

_slide_number(sl, 3, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 4 — RESEARCH QUESTIONS
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Five Research Questions"); _subtitle_bar(sl)

qs = [
    ("Q1", "Which RL algorithm performs best across energy domains?", "Expected: SAC for off-policy efficiency; PPO for stability under noise", ACCENT_BLUE),
    ("Q2", "Is safe (constrained) RL worth the reward cost?", "Expected: X% fewer violations at Y% reward reduction — quantify Pareto tradeoff", VT_ORANGE),
    ("Q3", "Does multi-agent RL improve over single-agent?", "Expected: env-dependent — EV charging benefits; cogen may not", ACCENT_GREEN),
    ("Q4", "Which noise type is most damaging to energy RL?", "Hypothesis: env noise (distribution shift) > obs noise > action noise", ACCENT_RED),
    ("Q5", "Does RL outperform traditional baselines under uncertainty?", "Expected: RL beats rule-based; advantage grows with increasing noise", VT_MAROON),
]
for i, (qid, q, exp, col) in enumerate(qs):
    y = Inches(1.4) + Inches(i * 1.15)
    bx = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.4), y + Inches(0.1), Inches(0.55), Inches(0.55))
    bx.fill.solid(); bx.fill.fore_color.rgb = col; bx.line.fill.background()
    tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = qid; p.font.size = Pt(13); p.font.color.rgb = WHITE; p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    _txt(sl, q, Inches(1.2), y, Inches(6.5), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
    _txt(sl, exp, Inches(1.2), y + Inches(0.38), Inches(11.5), Inches(0.5), sz=13, color=MED_GRAY)
_slide_number(sl, 4, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 5 — EV CHARGING ENV DETAIL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Environment 1: EV Charging", color=ACCENT_GREEN); _subtitle_bar(sl)
_card(sl, "Source", "ACN-Data / ACN-Sim (Caltech)\nGMM-based trace generation", Inches(0.3), Inches(1.4), Inches(4.0), Inches(1.0), ACCENT_GREEN)
_card(sl, "Episode", "288 timesteps (24-hour day, 5-min resolution)\n54 charging stations", Inches(4.5), Inches(1.4), Inches(4.0), Inches(1.0), ACCENT_GREEN)
_card(sl, "Reward", "profit - carbon_cost - excess_charge\n(USD per timestep)", Inches(8.7), Inches(1.4), Inches(4.3), Inches(1.0), ACCENT_GREEN)

_txt(sl, "Observation Space (Dict)", Inches(0.3), Inches(2.7), Inches(6), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
obs_items = [
    "timestep: scalar [0, 287]",
    "est_departures: (54,) estimated departure times",
    "demands: (54,) remaining energy demand per station",
    "prev_moer: (1,) previous marginal operating emissions rate",
    "forecasted_moer: (36,) 3-hour MOER forecast"
]
_bullets(sl, obs_items, Inches(0.3), Inches(3.1), Inches(6), Inches(2.5), sz=13, color=DARK_TEXT)

_txt(sl, "Action Space: Box(54)", Inches(6.5), Inches(2.7), Inches(6), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
act_items = [
    "Continuous [0, 1] normalized pilot signals",
    "Scaled to [0, 32] Amps in environment",
    "Network constraint projection (CVXPY+MOSEK) optional",
]
_bullets(sl, act_items, Inches(6.5), Inches(3.1), Inches(6), Inches(1.5), sz=13, color=DARK_TEXT)

_txt(sl, "MARL Setup", Inches(6.5), Inches(4.5), Inches(6), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
_txt(sl, "54 agents (one per station)\nShared reward / num_agents\nPettingZoo ParallelEnv", Inches(6.5), Inches(4.9), Inches(6), Inches(1.0), sz=13, color=MED_GRAY)

_txt(sl, "Safe RL Cost", Inches(0.3), Inches(5.8), Inches(6), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
_txt(sl, "excess_charge * 100 (network constraint violations)", Inches(0.3), Inches(6.2), Inches(6), Inches(0.5), sz=13, color=MED_GRAY)
_slide_number(sl, 5, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 6 — BUILDING ENV DETAIL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Environment 2: Building HVAC", color=ACCENT_BLUE); _subtitle_bar(sl)
_card(sl, "Source", "EnergyPlus RC thermal model\nASHRAE 90.1-2019, OfficeSmall, Hot_Dry", Inches(0.3), Inches(1.4), Inches(4.0), Inches(1.0), ACCENT_BLUE)
_card(sl, "Episode", "288 timesteps (24h, 5-min resolution)\nMultiple thermal zones", Inches(4.5), Inches(1.4), Inches(4.0), Inches(1.0), ACCENT_BLUE)
_card(sl, "Reward", "-(energy_rate * ||action||_p +\nerror_rate * ||temp_error||_p)\nNormalized to [-1, 0]", Inches(8.7), Inches(1.4), Inches(4.3), Inches(1.0), ACCENT_BLUE)

_txt(sl, "Observation: Box(n+4)", Inches(0.3), Inches(2.7), Inches(6), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
_bullets(sl, ["zone_temperatures (n)", "outdoor_temperature (1)", "ground_temperature (1)", "global_horizontal_irradiance (1)", "occupancy_power (1)"], Inches(0.3), Inches(3.1), Inches(6), Inches(2.0), sz=13)

_txt(sl, "Action: Box(n) in [-1, 1]", Inches(6.5), Inches(2.7), Inches(6), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
_bullets(sl, ["Negative = cooling, Positive = heating", "Zones without AC have fixed action=0", "reward_beta controls comfort vs energy tradeoff"], Inches(6.5), Inches(3.1), Inches(6), Inches(1.5), sz=13)

_txt(sl, "MARL: 1 agent per AC-enabled zone  |  Safe RL cost: temperature violations",
     Inches(0.3), Inches(5.5), Inches(12), Inches(0.5), sz=15, bold=True, color=MED_GRAY)
_slide_number(sl, 6, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 7 — COGEN ENV DETAIL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Environment 3: Cogeneration Plant", color=VT_ORANGE); _subtitle_bar(sl)
_card(sl, "Source", "ONNX surrogate model of CHP plant\nLazy-loaded in reset() for pickling", Inches(0.3), Inches(1.4), Inches(4.0), Inches(1.0), VT_ORANGE)
_card(sl, "Episode", "96 timesteps per day\nMixed continuous/discrete actions", Inches(4.5), Inches(1.4), Inches(4.0), Inches(1.0), VT_ORANGE)
_card(sl, "Reward", "-(fuel + ramp + non_delivery +\nconstraint_violation) / 1e7", Inches(8.7), Inches(1.4), Inches(4.3), Inches(1.0), VT_ORANGE)

_txt(sl, "MyCogenEnv Wrapper", Inches(0.3), Inches(2.7), Inches(12), Inches(0.35), sz=16, bold=True, color=DARK_TEXT)
_bullets(sl, [
    "Flattens Dict action space to single continuous Box(15)",
    "Removes Prev_Action nesting from obs Dict",
    "Discrete actions decoded with np.rint() + np.clip() (NOT truncation)",
    "Discrete(2) -> [0,1]; Discrete(12, start=1) -> [1,12]",
    "Reward scaled by 1/reward_scale (default 1e7)",
    "safe_rl=True further flattens Dict obs to Box for OmniSafe",
], Inches(0.3), Inches(3.1), Inches(12), Inches(2.5), sz=13)

_txt(sl, "MARL: 4 agents (GT1, GT2, GT3, ST) — per-agent reward  |  Safe RL cost: dyn_cv_costs / 1e7",
     Inches(0.3), Inches(5.5), Inches(12), Inches(0.5), sz=15, bold=True, color=MED_GRAY)
_slide_number(sl, 7, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 8 — NOISE FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Noise Robustness Framework"); _subtitle_bar(sl)

_txt(sl, "Three Independent Noise Channels — Key Methodological Contribution",
     Inches(0.4), Inches(1.3), Inches(12), Inches(0.4), sz=20, bold=True, color=VT_MAROON)

noise_data = [
    ("Observation Noise", "o_noisy = o + N(0, sigma_o * |o|)\nSensor uncertainty, forecast errors\nLevels: {0, 0.05, 0.10, 0.20, 0.40}", ACCENT_BLUE,
     "Agent sees\nnoisy state"),
    ("Action Noise", "a_noisy = clip(a + N(0, sigma_a), lo, hi)\nActuator imprecision, communication errors\nLevels: {0, 0.05, 0.10, 0.20, 0.40}", ACCENT_GREEN,
     "Env receives\nnoisy action"),
    ("Environment Noise", "s' = f(s, a; theta + N(0, sigma_e))\nModel mismatch, real-world stochasticity\nLevels: {0, 0.05, 0.10, 0.20, 0.40}", VT_ORANGE,
     "Dynamics are\nstochastic"),
]
for i, (title, desc, color, tag) in enumerate(noise_data):
    y = Inches(2.0) + Inches(i * 1.7)
    _card(sl, title, desc, Inches(0.4), y, Inches(8.0), Inches(1.45), color)
    bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), y + Inches(0.3), Inches(4.0), Inches(0.7))
    bx.fill.solid(); bx.fill.fore_color.rgb = color; bx.line.fill.background()
    tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = tag; p.font.size = Pt(15); p.font.color.rgb = WHITE
    p.font.bold = True; p.alignment = PP_ALIGN.CENTER

_txt(sl, "Ablation: Independent channels + joint noise → reveals interaction effects",
     Inches(0.4), Inches(7.0), Inches(12), Inches(0.35), sz=14, bold=True, color=ACCENT_RED)
_slide_number(sl, 8, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 9 — NOISE DECOMPOSITION DETAIL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Noise Decomposition: Experiment Design"); _subtitle_bar(sl)

_txt(sl, "Protocol: Independent Channel Ablation + Joint Noise", Inches(0.4), Inches(1.3), Inches(12), Inches(0.4),
     sz=20, bold=True, color=VT_MAROON)

_card(sl, "Phase A: Independent Channels",
      "For each noise_type in {obs, action, env}:\n  For each level in {0, 0.05, 0.10, 0.20, 0.40}:\n    Train with ONLY this noise type active\n    Other two channels = 0",
      Inches(0.3), Inches(1.9), Inches(6.0), Inches(1.8), ACCENT_BLUE)

_card(sl, "Phase B: Joint Noise",
      "For each level in {0, 0.05, 0.10, 0.20, 0.40}:\n  Train with ALL three channels at same level\n  Compare: joint_drop vs sum(individual_drops)\n  Reveals: additive or compounding effects?",
      Inches(6.6), Inches(1.9), Inches(6.4), Inches(1.8), VT_ORANGE)

_card(sl, "Phase C: Distribution Shift (Train-Clean-Test-Noisy)",
      "Take best clean model (noise=0)\nEvaluate under noise = {0.05, 0.10, 0.20, 0.40}\nNo additional training — evaluation only\nMeasures deployment robustness",
      Inches(0.3), Inches(4.0), Inches(6.0), Inches(1.6), ACCENT_GREEN)

_card(sl, "Key Analysis Questions",
      "1. Does obs noise > act noise > env noise (or different order)?\n2. Is joint noise additive or super-additive?\n3. How much does in-distribution training help vs clean-only?\n4. Are vulnerability profiles consistent across environments?",
      Inches(6.6), Inches(4.0), Inches(6.4), Inches(1.6), VT_MAROON)

# Placeholder for figure
bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), Inches(5.9), Inches(12.7), Inches(1.3))
bx.fill.solid(); bx.fill.fore_color.rgb = LIGHT_BLUE
bx.line.color.rgb = ACCENT_BLUE; bx.line.width = Pt(2)
_txt(sl, "[FIGURE PLACEHOLDER: 3x3 Noise Degradation Curves — rows=envs, cols=noise types]",
     Inches(1.5), Inches(6.2), Inches(10), Inches(0.5), sz=14, bold=True, color=ACCENT_BLUE, align=PP_ALIGN.CENTER)
_slide_number(sl, 9, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SECTION DIVIDER — LITERATURE
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("Literature Positioning", "Where we fit in the existing landscape")
_slide_number(sl, 10, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 11 — LITERATURE MAP
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Literature Landscape"); _subtitle_bar(sl)

benchmarks = [
    ("SustainGym (NeurIPS'23)", "3 envs, basic RL baselines", "We extend: safe RL + MARL + noise", ACCENT_BLUE),
    ("CityLearn (BuildSys'19)", "MARL for buildings only", "We add: 3 domains + safe RL", ACCENT_GREEN),
    ("BOPTEST (JBPS'21)", "Building control + MPC baselines", "We add: RL benchmark + multi-domain", VT_ORANGE),
    ("Sinergym (BuildSys'21)", "EnergyPlus RL, single domain", "We: multi-domain, multi-paradigm", VT_MAROON),
    ("OmniSafe (arXiv'23)", "Safe RL framework, robotics bench", "We: apply to energy domains", ACCENT_RED),
    ("Dulac-Arnold (ML'21)", "Real-world RL challenges", "We: instantiate for energy systems", MED_GRAY),
]
for i, (name, scope, gap, col) in enumerate(benchmarks):
    y = Inches(1.4) + Inches(i * 0.92)
    _badge(sl, name, Inches(0.3), y, Inches(3.2), Inches(0.42), col)
    _txt(sl, scope, Inches(3.7), y, Inches(4.2), Inches(0.42), sz=13, color=DARK_TEXT)
    _txt(sl, gap, Inches(8.2), y, Inches(4.8), Inches(0.42), sz=13, bold=True, color=ACCENT_GREEN)

_txt(sl, "Our Unique Position: Intersection of {multi-domain} x {multi-paradigm} x {noise decomposition}",
     Inches(0.3), Inches(7.0), Inches(12), Inches(0.35), sz=16, bold=True, color=VT_MAROON)
_slide_number(sl, 11, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SECTION DIVIDER — NOVELTY
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("Novelty & Contributions", "Deep Dive: What is new, what is not, and what to claim")
_slide_number(sl, 12, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 13 — HONEST NOVELTY AUDIT
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Honest Novelty Audit: What Is NOT Novel"); _subtitle_bar(sl)

_txt(sl, "We must be precise about what already exists — overclaiming kills credibility",
     Inches(0.4), Inches(1.3), Inches(12), Inches(0.4), sz=16, bold=True, color=ACCENT_RED)

not_novel = [
    "Environments — SustainGym published at NeurIPS 2023 (Yeh et al.)",
    "PPO / SAC / TD3 — well-established algorithms (2017-2018)",
    "OmniSafe framework — published (Ji et al. 2023)",
    "Ray RLlib / PettingZoo — published frameworks",
    "RL for EV charging / buildings / cogen — many prior papers",
    "Safe RL theory (CMDP) — Altman 1999, Achiam 2017",
    "MARL for buildings — CityLearn, GridLearn exist",
]
_bullets(sl, [f"  {x}" for x in not_novel], Inches(0.4), Inches(1.8), Inches(12), Inches(3.5), sz=15, color=DARK_TEXT)

_txt(sl, "Novelty must come from the BENCHMARK PROTOCOL and the INSIGHTS it generates — not the components.",
     Inches(0.4), Inches(5.5), Inches(12), Inches(0.5), sz=18, bold=True, color=VT_MAROON)
_slide_number(sl, 13, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 14 — NOVELTY CLAIM N1
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Novelty N1: Unified Multi-Paradigm Benchmark"); _subtitle_bar(sl)

_card(sl, "What We Must Do",
      "1. Complete standard RL (PPO, SAC) across all 3 envs with 5 seeds\n2. Complete safe RL (OnCRPO, CPO, PPOLag) across all 3 envs\n3. Complete MARL training across all 3 envs with 5 seeds\n4. Present all results in a single unified table with identical metrics",
      Inches(0.3), Inches(1.3), Inches(6.2), Inches(2.2), ACCENT_BLUE)

_card(sl, "What We Can Then Claim",
      '"To the best of our knowledge, this is the first benchmark that\nsystematically evaluates standard RL, constrained (safe) RL,\nand multi-agent RL on the same set of energy system\nenvironments under a unified evaluation protocol."',
      Inches(6.8), Inches(1.3), Inches(6.2), Inches(2.2), ACCENT_GREEN)

_card(sl, "Why This Is Credible",
      "- SustainGym: environments + basic PPO only (no safe/MARL benchmark)\n- CityLearn: MARL for buildings only (no safe RL, single domain)\n- No existing paper spans EV + Building + Cogen AND std + safe + MARL",
      Inches(0.3), Inches(3.8), Inches(6.2), Inches(1.5), VT_MAROON)

bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.8), Inches(6.2), Inches(1.5))
bx.fill.solid(); bx.fill.fore_color.rgb = ACCENT_GREEN
bx.line.fill.background()
tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True; tf.margin_left = Inches(0.3)
p = tf.paragraphs[0]; p.text = "STRENGTH: HIGH"; p.font.size = Pt(22); p.font.color.rgb = WHITE; p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph(); p2.text = "This is the paper's anchor novelty claim"; p2.font.size = Pt(14)
p2.font.color.rgb = WHITE; p2.alignment = PP_ALIGN.CENTER

_slide_number(sl, 14, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 15 — NOVELTY CLAIM N2
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Novelty N2: Structured Noise Decomposition"); _subtitle_bar(sl)

_card(sl, "What We Must Do",
      "1. Define formal noise model with 3 orthogonal channels\n2. Run experiments varying each channel independently\n3. Run joint noise experiments (all 3 simultaneously)\n4. Show decomposed impact profiles differ across envs",
      Inches(0.3), Inches(1.3), Inches(6.2), Inches(2.0), ACCENT_BLUE)

_card(sl, "What We Can Then Claim",
      '"We introduce a structured noise decomposition for energy RL,\ndecomposing uncertainty into obs, action, and env channels.\nOur analysis reveals that [environment noise dominates in EV\nwhile observation noise dominates in cogen]."',
      Inches(6.8), Inches(1.3), Inches(6.2), Inches(2.0), ACCENT_GREEN)

_card(sl, "Why This Is Credible",
      "- Dulac-Arnold (2021): defines challenges but doesn't apply to energy\n- Most energy RL papers: add noise as single unstructured perturbation\n- Decomposition into obs/action/env is a methodological contribution",
      Inches(0.3), Inches(3.6), Inches(6.2), Inches(1.5), VT_MAROON)

bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.6), Inches(6.2), Inches(1.5))
bx.fill.solid(); bx.fill.fore_color.rgb = ACCENT_GREEN; bx.line.fill.background()
tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True; tf.margin_left = Inches(0.3)
p = tf.paragraphs[0]; p.text = "STRENGTH: HIGH"; p.font.size = Pt(22); p.font.color.rgb = WHITE; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph(); p2.text = "Methodological novelty — provides actionable guidance"; p2.font.size = Pt(14); p2.font.color.rgb = WHITE; p2.alignment = PP_ALIGN.CENTER
_slide_number(sl, 15, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 16 — NOVELTY CLAIMS N3-N5
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Novelty Claims N3, N4, N5"); _subtitle_bar(sl)

claims = [
    ("N3: First Safe RL for EV Charging & Cogen", "MED-HIGH",
     "CMDP formulation with OmniSafe\nPareto analysis: reward vs. constraint cost\nNo prior CMDP work on ACN-based EV or ONNX-based cogen",
     VT_ORANGE),
    ("N4: Cross-Domain MARL Comparison", "MEDIUM",
     "54 agents (EV) vs 4 agents (Cogen) vs ~5 agents (Building)\nShared reward vs per-agent reward decomposition\nFindings may be environment-specific",
     ACCENT_BLUE),
    ("N5: RL vs Non-RL Under Uncertainty", "HIGH",
     "Rule-based + (optional) MPC baselines\n\"RL advantage grows with noise\" — actionable for practitioners\nAddresses the most fundamental reviewer concern",
     ACCENT_GREEN),
]
for i, (title, strength, desc, col) in enumerate(claims):
    y = Inches(1.3) + Inches(i * 2.0)
    _card(sl, title, desc, Inches(0.3), y, Inches(9.5), Inches(1.7), col)
    _badge(sl, f"Strength: {strength}", Inches(10.2), y + Inches(0.5), Inches(2.5), Inches(0.5), col)
_slide_number(sl, 16, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 17 — CONTRIBUTION STATEMENT
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Proposed Contribution Statement (C1-C5)"); _subtitle_bar(sl)

contribs = [
    ("C1", "Unified multi-paradigm evaluation protocol", "std RL + safe RL + MARL across 3 SustainGym envs", ACCENT_BLUE),
    ("C2", "Structured noise robustness framework", "obs/action/env decomposition with actionable guidance", ACCENT_GREEN),
    ("C3", "First safe RL benchmark for EV + Cogen", "CMDP cost signals + Pareto frontier characterization", VT_ORANGE),
    ("C4", "Cross-domain comparative analysis", "When RL beats baselines, when MARL helps, when safe RL costs", VT_MAROON),
    ("C5", "Open-source benchmark suite", "Reproducible scripts, eval tools, pre-trained models", MED_GRAY),
]
for i, (cid, title, desc, col) in enumerate(contribs):
    y = Inches(1.4) + Inches(i * 1.15)
    circle = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.4), y + Inches(0.08), Inches(0.6), Inches(0.6))
    circle.fill.solid(); circle.fill.fore_color.rgb = col; circle.line.fill.background()
    tf = circle.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = cid; p.font.size = Pt(14); p.font.color.rgb = WHITE; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
    _txt(sl, title, Inches(1.3), y, Inches(5.0), Inches(0.4), sz=18, bold=True, color=DARK_TEXT)
    _txt(sl, desc, Inches(1.3), y + Inches(0.4), Inches(11), Inches(0.6), sz=14, color=MED_GRAY)
_slide_number(sl, 17, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SECTION DIVIDER — GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("Gap Analysis", "Comprehensive audit of what's done, what's missing, what to cut")
_slide_number(sl, 18, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 19 — EXPERIMENT STATUS MATRIX (STD RL)
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Experiment Status: Standard RL"); _subtitle_bar(sl)

headers = ["Env / Algo", "Clean", "Obs Noise", "Act Noise", "Env Noise", "5 Seeds", "Overall"]
rows = [
    ["EV / PPO", "Done", "Done (10)", "Done", "In Prog", "?", "70%"],
    ["EV / SAC", "Done", "Done (10)", "Done", "In Prog", "?", "70%"],
    ["Bldg / PPO", "Partial", "Partial", "Partial", "Missing", "?", "30%"],
    ["Bldg / SAC", "Partial", "Partial", "Partial", "Missing", "?", "30%"],
    ["Cogen / PPO", "Partial", "Partial", "Missing", "Missing", "?", "20%"],
    ["Cogen / SAC", "Partial", "Partial", "Missing", "Missing", "?", "20%"],
]
status_map = {"Done": ACCENT_GREEN, "Done (10)": ACCENT_GREEN, "Partial": VT_ORANGE,
              "In Prog": VT_ORANGE, "Missing": ACCENT_RED, "?": MED_GRAY}

y0 = Inches(1.3); cw = [Inches(2.2), Inches(1.5), Inches(1.8), Inches(1.8), Inches(1.8), Inches(1.3), Inches(1.3)]
cx = [Inches(0.5)]
for w in cw[:-1]: cx.append(cx[-1] + w + Inches(0.1))

for j, (h, x, w) in enumerate(zip(headers, cx, cw)):
    bx = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y0, w, Inches(0.45))
    bx.fill.solid(); bx.fill.fore_color.rgb = TBL_HDR; bx.line.fill.background()
    tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = h; p.font.size = Pt(12); p.font.color.rgb = WHITE; p.font.bold = True; p.alignment = PP_ALIGN.CENTER

for i, row in enumerate(rows):
    y = y0 + Inches(0.5) + Inches(i * 0.5)
    bg = WHITE if i % 2 == 0 else VERY_LIGHT
    for j, (cell, x, w) in enumerate(zip(row, cx, cw)):
        bx = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Inches(0.45))
        bx.fill.solid(); bx.fill.fore_color.rgb = bg
        bx.line.color.rgb = RGBColor(0xEE,0xEE,0xEE); bx.line.width = Pt(0.5)
        tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.text = cell; p.font.size = Pt(12)
        p.font.bold = (j == 0 or j == 6)
        p.font.color.rgb = status_map.get(cell, DARK_TEXT) if j > 0 else DARK_TEXT
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
        if j == 0: tf.margin_left = Inches(0.1)

_txt(sl, "TD3 dropped from scope — PPO + SAC cover on-policy and off-policy families",
     Inches(0.5), Inches(4.7), Inches(12), Inches(0.4), sz=14, bold=True, color=VT_MAROON)
_slide_number(sl, 19, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 20 — EXPERIMENT STATUS: SAFE RL + MARL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Experiment Status: Safe RL & MARL"); _subtitle_bar(sl)

_txt(sl, "Safe RL (OmniSafe)", Inches(0.4), Inches(1.3), Inches(6), Inches(0.4), sz=20, bold=True, color=VT_ORANGE)
safe_rows = [
    ["EV / OnCRPO", "60%", "2 cost limits done, needs 5 seeds"],
    ["EV / CPO", "60%", "2 cost limits done, needs 5 seeds"],
    ["EV / PPOLag", "60%", "2 cost limits done, needs 5 seeds"],
    ["Bldg / OnCRPO", "15%", "Early results only"],
    ["Bldg / CPO+PPOLag", "0%", "Not started"],
    ["Cogen / OnCRPO", "10%", "Incomplete; debug prints in code"],
    ["Cogen / CPO+PPOLag", "0%", "Not started"],
]
for i, row in enumerate(safe_rows):
    y = Inches(1.8) + Inches(i * 0.42)
    _txt(sl, row[0], Inches(0.4), y, Inches(2.8), Inches(0.38), sz=12, bold=True, color=DARK_TEXT)
    col = ACCENT_GREEN if "60" in row[1] else (VT_ORANGE if row[1] not in ["0%"] else ACCENT_RED)
    _badge(sl, row[1], Inches(3.3), y, Inches(0.8), Inches(0.35), col)
    _txt(sl, row[2], Inches(4.3), y, Inches(3.5), Inches(0.38), sz=11, color=MED_GRAY)

_txt(sl, "Multi-Agent RL (RLlib)", Inches(8.0), Inches(1.3), Inches(5), Inches(0.4), sz=20, bold=True, color=ACCENT_GREEN)
marl_rows = [
    ["EV / APPO", "30%", "32k iters, 1 seed", "54 agents"],
    ["Bldg / SAC", "30%", "32k iters, 1 seed", "~5 agents"],
    ["Cogen / PPO", "10%", "Only 100 iters!", "4 agents"],
]
for i, row in enumerate(marl_rows):
    y = Inches(1.8) + Inches(i * 0.55)
    _txt(sl, row[0], Inches(8.0), y, Inches(2.2), Inches(0.38), sz=12, bold=True, color=DARK_TEXT)
    _badge(sl, row[1], Inches(10.2), y, Inches(0.8), Inches(0.35), VT_ORANGE)
    _txt(sl, row[2], Inches(8.0), y + Inches(0.28), Inches(2.8), Inches(0.25), sz=10, color=MED_GRAY)
    _txt(sl, row[3], Inches(11.2), y + Inches(0.05), Inches(1.5), Inches(0.3), sz=10, color=ACCENT_BLUE)

_txt(sl, "BASELINES: 0% complete for all 3 environments — CRITICAL GAP",
     Inches(0.4), Inches(5.0), Inches(12), Inches(0.4), sz=16, bold=True, color=ACCENT_RED)
_slide_number(sl, 20, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 21 — CODE-LEVEL BUGS
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Code-Level Issues to Fix"); _subtitle_bar(sl)

bugs = [
    ("Omnisafe_Building.py", "noise_action parsed but never passed to env", "MEDIUM", "Trivial fix"),
    ("Omnisafe_Cogen.py", "print(info) on every step floods logs", "LOW", "Trivial fix"),
    ("marl_tr_co.py", "Metrics file named 'building_env_metrics'", "LOW", "Rename"),
    ("marl_tr_co.py", "Only 100 iterations (insufficient)", "HIGH", "Increase to 2000+"),
    ("ttp/marl_plot.py", "All code commented out", "MEDIUM", "Reactivate"),
    ("stdrl_training.py", "No multi-seed batch support", "HIGH", "Add --seeds flag"),
    ("No file exists", "No baseline implementations", "CRITICAL", "Create baselines/"),
    ("No file exists", "No results aggregation pipeline", "HIGH", "Create aggregate_results.py"),
]
for i, (f, issue, severity, fix) in enumerate(bugs):
    y = Inches(1.3) + Inches(i * 0.72)
    sev_col = {"CRITICAL": ACCENT_RED, "HIGH": VT_ORANGE, "MEDIUM": ACCENT_BLUE, "LOW": MED_GRAY}[severity]
    _badge(sl, severity, Inches(0.3), y + Inches(0.05), Inches(1.1), Inches(0.35), sev_col)
    _txt(sl, f, Inches(1.6), y, Inches(2.8), Inches(0.35), sz=12, bold=True, color=DARK_TEXT)
    _txt(sl, issue, Inches(4.5), y, Inches(5.5), Inches(0.35), sz=12, color=MED_GRAY)
    _txt(sl, fix, Inches(10.2), y, Inches(2.8), Inches(0.35), sz=12, bold=True, color=ACCENT_GREEN)
_slide_number(sl, 21, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 22 — SCOPE REDUCTION
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Scope Reduction: What to Cut"); _subtitle_bar(sl)

cuts = [
    ("Drop TD3 entirely", "Keep PPO + SAC only", "Saves 33% of std RL runs", "LOW risk"),
    ("Reduce noise levels", "10 -> 5: {0, 0.05, 0.10, 0.20, 0.40}", "Saves 50% noise sweeps", "LOW risk"),
    ("Reduce safe RL algos", "15 -> 3: OnCRPO, CPO, PPOLag", "Huge savings", "LOW risk"),
    ("Reduce cost limits", "To 2: {1.0, 5.0}", "Saves 33% safe RL runs", "LOW risk"),
    ("Skip optimization baseline", "If time-constrained", "Saves impl. time", "MEDIUM risk"),
]
for i, (what, detail, saving, risk) in enumerate(cuts):
    y = Inches(1.4) + Inches(i * 1.0)
    _card(sl, what, f"{detail}\n{saving}", Inches(0.3), y, Inches(9.0), Inches(0.85), ACCENT_BLUE)
    risk_col = ACCENT_GREEN if "LOW" in risk else VT_ORANGE
    _badge(sl, risk, Inches(9.8), y + Inches(0.2), Inches(2.0), Inches(0.4), risk_col)

_txt(sl, "After reduction: ~735 total runs (many are fast evaluation-only)",
     Inches(0.3), Inches(6.6), Inches(12), Inches(0.4), sz=16, bold=True, color=VT_MAROON)
_slide_number(sl, 22, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SECTION DIVIDER — STRATEGIC ROADMAP
# ═══════════════════════════════════════════════════════════════════════
sl = _section_divider("Strategic Execution Roadmap", "Phase-by-phase plan from today to submission")
_slide_number(sl, 23, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 24 — PHASE 0: INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Phase 0: Infrastructure Setup (Mar 3-10)"); _subtitle_bar(sl)

tasks = [
    ("Step 0.1", "Multi-Seed Support", "Add --seeds 42,43,44,45,46 to stdrl_training.py\nOr create SLURM array job launcher", "LOW"),
    ("Step 0.2", "Experiment Tracker", "Script scanning logs/ dirs, cross-referencing\nagainst full experiment matrix", "LOW"),
    ("Step 0.3", "Domain Metric Extraction", "Add per-env metric extraction in stdrl_testing.py\nEV: profit, CO2, violations | Bldg: kWh, temp error | Cogen: fuel, delivery", "MEDIUM"),
    ("Step 0.4", "Bug Fixes", "Omnisafe_Building noise_action, Omnisafe_Cogen prints,\nmarl_tr_co filename", "TRIVIAL"),
    ("Step 0.5", "SLURM Templates", "Array job scripts for batch experiment submission", "LOW"),
]
for i, (step, name, desc, effort) in enumerate(tasks):
    y = Inches(1.3) + Inches(i * 1.15)
    _badge(sl, step, Inches(0.3), y + Inches(0.05), Inches(1.3), Inches(0.4), TBL_HDR)
    _txt(sl, name, Inches(1.8), y, Inches(2.5), Inches(0.35), sz=15, bold=True, color=DARK_TEXT)
    _txt(sl, desc, Inches(4.5), y, Inches(7.0), Inches(0.9), sz=12, color=MED_GRAY)
    eff_col = ACCENT_GREEN if "LOW" in effort or "TRIV" in effort else VT_ORANGE
    _badge(sl, effort, Inches(11.8), y + Inches(0.05), Inches(1.2), Inches(0.4), eff_col)
_slide_number(sl, 24, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 25 — PHASE 1: BASELINES
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Phase 1: Baseline Implementation (Mar 10-24)"); _subtitle_bar(sl)

base_data = [
    ("Do-Nothing", ACCENT_RED,
     "EV: max charge rate (all 1s)\nBuilding: HVAC off (all 0s)\nCogen: nominal point (midrange)", "Effort: Trivial\nExpected: worst performer"),
    ("Rule-Based", ACCENT_BLUE,
     "EV: LLLF (urgency-proportional)\nBuilding: deadband thermostat (+/-1C)\nCogen: load-following proportional", "Effort: Low\nExpected: decent performer"),
    ("Optimization (optional)", VT_ORANGE,
     "EV: LP for min-cost charging\nBuilding: 1-step MPC with RC model\nCogen: LP for fuel minimization", "Effort: Medium\nExpected: competitive ceiling"),
]
for i, (name, col, desc, notes) in enumerate(base_data):
    y = Inches(1.3) + Inches(i * 2.0)
    hdr = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), y, Inches(2.5), Inches(0.55))
    hdr.fill.solid(); hdr.fill.fore_color.rgb = col; hdr.line.fill.background()
    tf = hdr.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = name; p.font.size = Pt(16); p.font.color.rgb = WHITE; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
    _txt(sl, desc, Inches(3.0), y, Inches(5.5), Inches(1.7), sz=13, color=DARK_TEXT)
    _txt(sl, notes, Inches(8.8), y, Inches(4.0), Inches(1.7), sz=13, color=MED_GRAY)

_txt(sl, "Minimum: Do-Nothing + Rule-Based for all 3 envs = 30 eval runs (no training needed)",
     Inches(0.3), Inches(7.0), Inches(12), Inches(0.35), sz=15, bold=True, color=ACCENT_GREEN)
_slide_number(sl, 25, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 26 — PHASE 2: COMPLETE STD RL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Phase 2: Complete Standard RL (Mar 24 - Apr 21)"); _subtitle_bar(sl)

_txt(sl, "Priority Ordering by Information Value", Inches(0.4), Inches(1.3), Inches(12), Inches(0.4),
     sz=18, bold=True, color=VT_MAROON)

priorities = [
    ("P1", "Building + Cogen clean baselines (PPO, SAC, noise=0, 5 seeds)", "Foundation"),
    ("P2", "Building + Cogen obs noise sweeps (5 levels × 5 seeds)", "Extends EV results"),
    ("P3", "EVCharging env noise sweep (5 levels × 5 seeds)", "Third noise channel"),
    ("P4", "Building + Cogen action noise sweeps", "Complete 2nd channel"),
    ("P5", "Building + Cogen env noise sweeps", "Complete 3rd channel"),
]
for i, (pid, desc, why) in enumerate(priorities):
    y = Inches(1.9) + Inches(i * 0.65)
    _badge(sl, pid, Inches(0.4), y + Inches(0.05), Inches(0.7), Inches(0.35), VT_MAROON)
    _txt(sl, desc, Inches(1.3), y, Inches(8.0), Inches(0.5), sz=14, color=DARK_TEXT)
    _txt(sl, why, Inches(9.5), y, Inches(3.5), Inches(0.5), sz=13, bold=True, color=MED_GRAY)

_txt(sl, "SLURM Submission Strategy", Inches(0.4), Inches(5.3), Inches(12), Inches(0.4), sz=18, bold=True, color=VT_MAROON)
_bullets(sl, [
    "Week 1: Building clean + obs noise (highest priority)",
    "Week 2: Cogen clean + obs noise; Building action noise",
    "Week 3: EV env noise; Building env noise",
    "Week 4: Cogen action + env noise; reruns for failed jobs",
], Inches(0.4), Inches(5.8), Inches(12), Inches(1.5), sz=14, color=DARK_TEXT)
_slide_number(sl, 26, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 27 — PHASE 3: SAFE RL + MARL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Phase 3: Safe RL & MARL (Apr 21 - May 12)"); _subtitle_bar(sl)

_card(sl, "Safe RL Completion",
      "Building + Cogen: 3 algos x 2 cost limits x 5 seeds = 30 runs each\nVerify bug fixes before ARC submission\nRun quick local test per env first\nKey output: Pareto curves (reward vs constraint cost)",
      Inches(0.3), Inches(1.3), Inches(6.2), Inches(2.0), VT_ORANGE)

_card(sl, "MARL Completion",
      "Increase Cogen iterations: 100 -> 2000+\nAdd seed support to all MARL scripts\nShow convergence curves for all envs\nKey output: MARL vs single-agent bars + error bars",
      Inches(6.8), Inches(1.3), Inches(6.2), Inches(2.0), ACCENT_GREEN)

_card(sl, "Distribution Shift (Train-Clean-Test-Noisy)",
      "Take best clean model (noise=0, already trained)\nEvaluate under noise = {0.05, 0.10, 0.20, 0.40}\nNo additional training — evaluation only\n3 envs x 2 algos x 5 test noise x 5 seeds = 150 eval runs",
      Inches(0.3), Inches(3.6), Inches(12.7), Inches(1.6), ACCENT_BLUE)

# Placeholder
bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), Inches(5.5), Inches(6.2), Inches(1.7))
bx.fill.solid(); bx.fill.fore_color.rgb = LIGHT_BLUE; bx.line.color.rgb = ACCENT_BLUE; bx.line.width = Pt(2)
_txt(sl, "[FIGURE PLACEHOLDER]\nSafe RL Pareto Fronts\n(reward vs constraint cost, 3 panels)", Inches(0.8), Inches(5.8), Inches(5), Inches(1.2),
     sz=13, bold=True, color=ACCENT_BLUE, align=PP_ALIGN.CENTER)

bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(5.5), Inches(6.2), Inches(1.7))
bx.fill.solid(); bx.fill.fore_color.rgb = LIGHT_BLUE; bx.line.color.rgb = ACCENT_GREEN; bx.line.width = Pt(2)
_txt(sl, "[FIGURE PLACEHOLDER]\nMARL vs Single-Agent Comparison\n(grouped bar chart with error bars)", Inches(7.3), Inches(5.8), Inches(5), Inches(1.2),
     sz=13, bold=True, color=ACCENT_GREEN, align=PP_ALIGN.CENTER)
_slide_number(sl, 27, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 28 — EVALUATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Evaluation & Reporting Protocol"); _subtitle_bar(sl)

_card(sl, "Statistical Rigor", "5 seeds (42-46) per experiment\nBootstrap 95% CIs (10,000 resamples)\nInterquartile Mean (IQM) per Agarwal et al. NeurIPS'21\nWelch's t-test for pairwise comparisons",
      Inches(0.3), Inches(1.3), Inches(6.2), Inches(1.8), ACCENT_BLUE)
_card(sl, "Domain-Specific Metrics",
      "EV: profit ($), CO2 cost ($), violation rate (%)\nBuilding: energy (kWh), temp error (C), comfort viol. (%)\nCogen: fuel cost ($), constraint viol., delivery rate (%)\n+ Peak demand for all envs",
      Inches(6.8), Inches(1.3), Inches(6.2), Inches(1.8), ACCENT_GREEN)
_card(sl, "Normalized Comparisons",
      "Performance retention = reward(noise) / reward(clean) x 100%\nImprovement over baseline = (RL - baseline) / |baseline| x 100%\nSafe RL efficiency = constraint_reduction vs reward_retention\nMARL gain = (MARL - single_agent) / |single_agent| x 100%",
      Inches(0.3), Inches(3.4), Inches(6.2), Inches(1.8), VT_ORANGE)
_card(sl, "Visualization Standards",
      "Times New Roman, min 8pt in figures, 300 DPI\nColorblind-friendly palette (Seaborn 'colorblind')\nVary color AND line style for accessibility\nShaded IQR for training curves; error bars for bar charts",
      Inches(6.8), Inches(3.4), Inches(6.2), Inches(1.8), VT_MAROON)
_slide_number(sl, 28, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 29 — REQUIRED FIGURES
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Required Paper Figures"); _subtitle_bar(sl)

figs = [
    ("Fig 1", "Benchmark Taxonomy", "3 envs -> 3 paradigms -> noise channels -> algorithms\nDraw with matplotlib or TikZ", Inches(0.3), Inches(1.3)),
    ("Fig 2", "Training Curves (3 panels)", "X: timesteps, Y: reward, Lines: algorithms, Shade: IQR\nOne panel per environment", Inches(6.8), Inches(1.3)),
    ("Fig 3", "Noise Degradation (3x3 grid)", "Rows: envs, Cols: noise types\nX: noise level, Y: % retention, Error bars: 95% CI", Inches(0.3), Inches(3.2)),
    ("Fig 4", "Safe RL Pareto Fronts (3 panels)", "X: constraint cost, Y: reward\nPoints per (algo, cost_limit), Pareto frontier highlighted", Inches(6.8), Inches(3.2)),
    ("Fig 5", "Cross-Paradigm Comparison", "Grouped bars: DoNothing, RuleBased, PPO, SAC, OnCRPO, MARL\nError bars: 95% CI, color-coded by paradigm", Inches(0.3), Inches(5.1)),
    ("Fig 6", "Distribution Shift (optional)", "X: test noise, Y: reward\nLines: trained at different noise levels\nShows in-dist vs out-of-dist gap", Inches(6.8), Inches(5.1)),
]
for fid, title, desc, x, y in figs:
    bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(6.2), Inches(1.7))
    bx.fill.solid(); bx.fill.fore_color.rgb = LIGHT_BLUE; bx.line.color.rgb = ACCENT_BLUE; bx.line.width = Pt(1.5)
    _txt(sl, f"{fid}: {title}", x + Inches(0.2), y + Inches(0.1), Inches(5.8), Inches(0.35), sz=14, bold=True, color=ACCENT_BLUE)
    _txt(sl, desc, x + Inches(0.2), y + Inches(0.5), Inches(5.8), Inches(1.0), sz=12, color=MED_GRAY)
_slide_number(sl, 29, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 30 — REQUIRED TABLES
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Required Paper Tables"); _subtitle_bar(sl)

tables = [
    ("Table I", "Environment Specifications",
     "Episode length, obs/action space dims, reward range,\n# MARL agents, safe RL cost signal, data source"),
    ("Table II", "Main Results (Reward +/- CI)",
     "Paradigm x Algorithm x Environment\nBaselines, Std RL, Safe RL, MARL\nAll with mean +/- 95% CI"),
    ("Table III", "Domain-Specific Metrics",
     "EV: profit/CO2/violations | Bldg: kWh/temp/comfort\nCogen: fuel/constraints/delivery\nPer algorithm, per paradigm"),
    ("Table IV", "Robustness Analysis",
     "Performance retention (%) under noise\nRows: env x algo | Cols: noise type x level\nNormalized to clean baseline = 100%"),
]
for i, (tid, title, desc) in enumerate(tables):
    y = Inches(1.3) + Inches(i * 1.5)
    _card(sl, f"{tid}: {title}", desc, Inches(0.3), y, Inches(12.7), Inches(1.3), VT_MAROON)
_slide_number(sl, 30, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 31 — PAPER STRUCTURE
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Paper Structure (6-Page IEEE)"); _subtitle_bar(sl)

sections_paper = [
    ("I. Introduction", "0.75 pg", "Motivation, problem statement, C1-C5 contributions"),
    ("II. Related Work", "0.75 pg", "RL for energy, safe RL, MARL, benchmarks + gap"),
    ("III. Problem Formulation", "1.0 pg", "MDP, CMDP, Dec-POMDP + noise model + Table I"),
    ("IV. Experimental Setup", "0.75 pg", "Algorithms, hyperparams, baselines, eval protocol"),
    ("V. Results & Analysis", "2.0 pg", "Tables II-IV, Figures 2-5, key findings"),
    ("VI. Conclusion", "0.75 pg", "Insights, limitations, future work, code release"),
]
for i, (sec, pages, content) in enumerate(sections_paper):
    y = Inches(1.3) + Inches(i * 0.95)
    bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), y, Inches(3.5), Inches(0.7))
    bx.fill.solid(); bx.fill.fore_color.rgb = VT_MAROON; bx.line.fill.background()
    tf = bx.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Inches(0.15)
    p = tf.paragraphs[0]; p.text = sec; p.font.size = Pt(14); p.font.color.rgb = WHITE; p.font.bold = True
    _txt(sl, pages, Inches(4.0), y + Inches(0.1), Inches(1.0), Inches(0.4), sz=14, bold=True, color=VT_ORANGE)
    _txt(sl, content, Inches(5.2), y + Inches(0.1), Inches(7.8), Inches(0.5), sz=14, color=DARK_TEXT)
_slide_number(sl, 31, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 32 — REVIEWER ANTICIPATION
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Anticipated Reviewer Questions"); _subtitle_bar(sl)

qas = [
    ('"Why not test more algorithms?"', "PPO (on-policy) + SAC (off-policy) covers algorithm families.\nExtending is future work. 3 safe RL algos add further breadth."),
    ('"The environments are not new"', "Correct. Contribution is the benchmark protocol and findings,\nnot the environments. SustainGym cited properly."),
    ('"Why not include MPC baseline?"', "Rule-based baselines establish RL advantage. MPC assumes\nperfect information — inherently unfair comparison. Discussed."),
    ('"5 seeds is not enough"', "5 seeds with bootstrap CIs is standard per Henderson (AAAI'18).\nWe also report IQM per Agarwal (NeurIPS'21)."),
    ('"What about sim-to-real?"', "Acknowledged as limitation. Noise robustness analysis partially\naddresses this by measuring degradation under uncertainty."),
    ('"What is the practical impact?"', 'Actionable: "use SAC for noisy actuators"\n"safe RL reduces violations X% at Y% reward cost"'),
]
for i, (q, a) in enumerate(qas):
    y = Inches(1.3) + Inches(i * 1.0)
    _txt(sl, q, Inches(0.3), y, Inches(4.5), Inches(0.4), sz=14, bold=True, color=VT_MAROON)
    _txt(sl, a, Inches(5.0), y, Inches(8.0), Inches(0.8), sz=12, color=DARK_TEXT)
_slide_number(sl, 32, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 33 — RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Risk Assessment & Contingencies"); _subtitle_bar(sl)

_txt(sl, "Timeline Risks", Inches(0.3), Inches(1.3), Inches(6), Inches(0.4), sz=18, bold=True, color=ACCENT_RED)
t_risks = [
    ("HPC queue delays", "HIGH", "Submit early; use preemptable partition"),
    ("Training divergence", "MED", "Quick-test first; backup hyperparams"),
    ("OmniSafe crashes", "MED", "Local test first; fall back to PPOLag"),
    ("Cogen MARL no converge", "MED", "Report partial; focus EV+Building"),
    ("Writing time squeeze", "MED", "Start writing NOW during Phase 0"),
]
for i, (risk, prob, mit) in enumerate(t_risks):
    y = Inches(1.8) + Inches(i * 0.5)
    _txt(sl, risk, Inches(0.3), y, Inches(2.5), Inches(0.4), sz=12, color=DARK_TEXT)
    prob_col = ACCENT_RED if "HIGH" in prob else VT_ORANGE
    _badge(sl, prob, Inches(2.9), y, Inches(0.8), Inches(0.35), prob_col)
    _txt(sl, mit, Inches(3.9), y, Inches(3.0), Inches(0.4), sz=11, color=MED_GRAY)

_txt(sl, "Result Risks", Inches(7.0), Inches(1.3), Inches(6), Inches(0.4), sz=18, bold=True, color=VT_ORANGE)
r_risks = [
    ("RL < rule-based", "LOW-MED", "Report honestly; RL may win under noise"),
    ("Safe RL no benefit", "MED", "Try more cost limits; report as finding"),
    ("MARL < single-agent", "MED", "Frame: 'naive decomposition hurts'"),
    ("Noise has no effect", "LOW", "Report: 'energy RL surprisingly robust'"),
    ("Wide CIs, unclear", "MED", "Use IQM; increase to 10 seeds if feasible"),
]
for i, (risk, prob, mit) in enumerate(r_risks):
    y = Inches(1.8) + Inches(i * 0.5)
    _txt(sl, risk, Inches(7.0), y, Inches(2.5), Inches(0.4), sz=12, color=DARK_TEXT)
    prob_col = ACCENT_RED if "HIGH" in prob else (VT_ORANGE if "MED" in prob else MED_GRAY)
    _badge(sl, prob, Inches(9.5), y, Inches(1.1), Inches(0.35), prob_col)
    _txt(sl, mit, Inches(10.8), y, Inches(2.5), Inches(0.4), sz=11, color=MED_GRAY)
_slide_number(sl, 33, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 34 — CONTINGENCY TIERS
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Scope Contingency: Three Tiers"); _subtitle_bar(sl)

tiers = [
    ("Tier 1: Minimum Publishable", "~145 runs", ACCENT_RED,
     ["3 envs x 2 algos x clean only x 5 seeds (30 runs)",
      "3 envs x 2 baselines x 5 seeds (30 eval runs)",
      "Obs noise on EV + Building only (60 runs)",
      "Safe RL: EV only, 2 algos (20 runs)",
      "MARL: EV only (5 runs)"]),
    ("Tier 2: Strong Submission (TARGET)", "~735 runs", ACCENT_GREEN,
     ["Full matrix: 3 envs x 2 algos x 5 noise x 3 types x 5 seeds",
      "Safe RL: 3 envs x 3 algos x 2 limits x 5 seeds",
      "MARL: 3 envs x 1 algo x 5 seeds",
      "Distribution shift experiments",
      "All baselines evaluated"]),
    ("Tier 3: Aspirational (Top-Tier)", "~2000+ runs", VT_MAROON,
     ["Full Tier 2 + optimization baselines",
      "10 seeds per experiment",
      "All noise combinations (joint)",
      "Ablation on reward_beta, VecNormalize, etc."]),
]
for i, (name, count, col, items) in enumerate(tiers):
    y = Inches(1.3) + Inches(i * 2.0)
    hdr = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), y, Inches(4.5), Inches(0.55))
    hdr.fill.solid(); hdr.fill.fore_color.rgb = col; hdr.line.fill.background()
    tf = hdr.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.margin_left = Inches(0.15)
    p = tf.paragraphs[0]; p.text = f"{name}  ({count})"; p.font.size = Pt(15)
    p.font.color.rgb = WHITE; p.font.bold = True
    _bullets(sl, items, Inches(5.0), y, Inches(8.0), Inches(1.8), sz=13, color=DARK_TEXT, spacing=3)
_slide_number(sl, 34, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 35 — TIMELINE
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6]); _set_bg(sl); _add_logo(sl)
_title_bar(sl, "Timeline to Submission: June 19, 2026"); _subtitle_bar(sl)

phases = [
    ("Phase 0", "Infrastructure", "Mar 3-10", VT_MAROON, "Multi-seed, tracker, metrics, bug fixes, SLURM templates"),
    ("Phase 1", "Baselines", "Mar 10-24", ACCENT_RED, "Do-Nothing + Rule-Based for all 3 envs; evaluate 30 runs"),
    ("Phase 2", "Complete Std RL", "Mar 24 - Apr 21", VT_ORANGE, "Building + Cogen all noise sweeps; 5 seeds everywhere"),
    ("Phase 3", "Safe RL + MARL", "Apr 21 - May 12", ACCENT_BLUE, "OmniSafe for Bldg/Cogen; MARL convergence; dist. shift"),
    ("Phase 4", "Analysis + Figs", "May 12-26", ACCENT_GREEN, "Aggregate results; generate all figures and tables"),
    ("Phase 5", "Paper Writing", "May 26 - Jun 12", MED_GRAY, "Draft, integrate results, related work, revision"),
    ("Phase 6", "Finalize", "Jun 12-19", RGBColor(0x33,0x99,0x33), "Proofread, format check, supplementary, SUBMIT"),
]

# Timeline visualization
bar_left = Inches(3.5); bar_width = Inches(9.3)
total_days = 108  # Mar 3 to Jun 19

for i, (pid, name, dates, col, tasks) in enumerate(phases):
    y = Inches(1.3) + Inches(i * 0.82)
    # Label
    _txt(sl, f"{pid}: {name}", Inches(0.3), y, Inches(3.0), Inches(0.35), sz=13, bold=True, color=col)
    _txt(sl, dates, Inches(0.3), y + Inches(0.32), Inches(3.0), Inches(0.25), sz=10, color=MED_GRAY)
    # Tasks
    _txt(sl, tasks, Inches(3.5), y + Inches(0.05), Inches(9.5), Inches(0.6), sz=11, color=DARK_TEXT)

_txt(sl, "START WRITING DURING PHASE 0-1 — Don't wait for all results!",
     Inches(0.3), Inches(7.0), Inches(12), Inches(0.35), sz=15, bold=True, color=ACCENT_RED)
_slide_number(sl, 35, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 36 — SUMMARY & NEXT STEPS
# ═══════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
_set_bg(sl, DARK_BG); _add_logo(sl, dark_bg=True)

_txt(sl, "Summary & Immediate Next Steps", Inches(0.8), Inches(0.8), Inches(11), Inches(0.7),
     sz=38, bold=True, color=WHITE)
sep = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.6), Inches(3), Inches(0.04))
sep.fill.solid(); sep.fill.fore_color.rgb = VT_ORANGE; sep.line.fill.background()

summary = [
    "Scope is strong: 3 envs x 3 paradigms + noise decomposition",
    "5 novelty claims identified (N1-N5) with clear if/then mapping",
    "Critical gaps: baselines, statistical rigor, Building/Cogen completion",
    "~735 experiments needed for strong submission (Tier 2)",
    "Timeline tight but feasible: 3.5 months to June 19 deadline",
]
for i, item in enumerate(summary):
    y = Inches(2.0) + Inches(i * 0.6)
    dot = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.8), y + Inches(0.08), Inches(0.18), Inches(0.18))
    dot.fill.solid(); dot.fill.fore_color.rgb = VT_ORANGE; dot.line.fill.background()
    _txt(sl, item, Inches(1.2), y, Inches(11), Inches(0.5), sz=17, color=WHITE)

_txt(sl, "This Week's Actions", Inches(0.8), Inches(5.2), Inches(5), Inches(0.4), sz=22, bold=True, color=VT_ORANGE)
actions = [
    "1. Implement Do-Nothing + Rule-Based baselines",
    "2. Add multi-seed support to training scripts",
    "3. Add domain metric extraction to stdrl_testing.py",
    "4. Apply bug fixes (Omnisafe, marl_tr_co)",
    "5. Create SLURM array job templates",
    "6. Start writing Introduction + Related Work",
]
_bullets(sl, actions, Inches(0.8), Inches(5.7), Inches(11), Inches(1.8), sz=15, color=LIGHT_GRAY, spacing=4)
_slide_number(sl, 36, TOTAL_SLIDES)

# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════
output = os.path.join(SCRIPT_DIR, "smartgridcomm_deep_dive_presentation.pptx")
prs.save(output)
print(f"\nPresentation saved: {output}")
print(f"Total slides: {len(prs.slides)}")
