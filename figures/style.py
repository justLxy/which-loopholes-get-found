"""Shared style of the paper figures: ACL sizes, Times, a colorblind-safe palette and display names.

Every figure script (figures/gen_fig_*.py) reads committed results only and writes fig_<name>.pdf (vector,
fonts embedded as TrueType, as the ACL format check requires) and fig_<name>.png (300 dpi) next to itself.
Run one with `python -m figures.gen_fig_<name>` or all with `python -m figures.make_all`.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.paths import ROOT  # noqa: E402

OUT = ROOT / "figures"
COLUMN, TEXT = 3.03, 6.3       # inches: acl.sty sets \textwidth to 16 cm and \columnsep to 0.6 cm

# Okabe-Ito, safe for the common forms of colour blindness and distinct in greyscale by marker/line style
OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
         "vermillion": "#D55E00", "pink": "#CC79A7", "black": "#000000", "grey": "#8C8C8C", "light": "#D9D9D9"}
SCORE_COLORS = {"risk": OKABE["vermillion"], "fpr": OKABE["blue"], "p_hat": OKABE["green"], "static": OKABE["grey"]}
SCORE_LABELS = {"risk": "exposure", "fpr": "FPR alone", "p_hat": "propensity alone",
                "static": "static accuracy"}

FAMILY_NAMES = {
    "empty_symbols": "Empty symbols", "gibberish": "Gibberish", "answer_repetition": "Answer repetition",
    "unicode_confusables": "Unicode confusables", "latex_notation": "LaTeX notation",
    "html_markdown": "HTML/Markdown", "system_prompt_mimicry": "System-prompt mimicry",
    "instruction_hijacking": "Instruction hijacking", "prompt_injection": "Prompt injection",
    "mixed_language": "Mixed language", "adversarial_prefixes": "Adversarial prefix",
    "answer_explanation": "Answer explanation", "simple_words": "Simple words",
    "reasoning_opener": "Reasoning opener"}
JUDGE_NAMES = {
    "r1_distill_verifier_1p5b": "R1-Distill-Verifier-1.5B", "r1_distill_qwen_1p5b": "R1-Distill-Qwen-1.5B",
    "qwen2p5_math_1p5b_instruct": "Qwen2.5-Math-1.5B-Inst.", "xverify_0p5b": "xVerify-0.5B",
    "general_verifier": "General-Verifier", "qwen2p5_7b_instruct": "Qwen2.5-7B-Inst.",
    "qwen3p6_35b_a3b_fp8": "Qwen3.6-35B-A3B"}
JUDGE_ORDER = list(JUDGE_NAMES)          # core judges first, then the held-out ones


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "lines.linewidth": 1.2, "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150})


def save(fig, name):
    # no creation date in the PDF, so redrawing unchanged data gives byte-identical files
    fig.savefig(OUT / f"fig_{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / f"fig_{name}.png")
    plt.close(fig)
    return OUT / f"fig_{name}.pdf"
