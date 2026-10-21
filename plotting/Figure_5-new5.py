#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three publication figures for the newest no-penalty trimer LJ+ATM GAPS campaign.

This is PLOTTING/DIAGNOSTICS ONLY. It does NOT modify campaign_state.json,
rerun WHAM, refit the GP, submit jobs, or change saved simulation data.

Figure 1 — four-panel diagnostics
    A. Exact-versus-estimated parity (WHAM support + dense GP)
    B. Reconstruction RMSE versus physical umbrella-window count
    C. Node/support statistics versus information iteration
    D. Adaptive umbrella spring constant versus physical window

Figure 2 — configurable mixed slices through the full cubic configuration space
    Columns: Ref | direct WHAM | GP
    Number of rows is user-selectable: 2, 3, or 4.

    2 rows:
        fixed d12 = 1.10, shown as d13 vs d23
        isosceles plane d13 = d23, shown as d12 vs d13=d23
    3 rows:
        fixed d12 = 1.10, shown as d13 vs d23
        fixed d23 = 1.70, shown as d12 vs d13
        isosceles plane d13 = d23
    4 rows:
        fixed d12 = 1.10 and 1.70, both shown as d13 vs d23
        fixed d23 = 1.70, shown as d12 vs d13
        isosceles plane d13 = d23

    The requested constants are snapped to the nearest saved GP query-grid value.
    The GP panels use saved GP predictions on the canonical query grid,
    symmetry-expanded exactly. The WHAM panels use the same display-only
    symmetry-aware inverse-distance interpolation used by the production code
    for slice visualization. No interpolated values are fed back to WHAM/GP.

Figure 3 — sampling graph and distributions
    A. Full cubic d-space, symmetry expanded
    B. Permutation-reduced d(1) <= d(2) <= d(3) wedge
    Shows all completed physical-window centers, all measured-overlap graph
    edges, sampled distributions (display-downsampled only), and current query.

Default campaign:
    automatically selected as the no-penalty trimer campaign containing the newest online_analysis.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

SPYDER_INLINE_PLOTS = True
if SPYDER_INLINE_PLOTS:
    try:
        from IPython import get_ipython
        _ip = get_ipython()
        if _ip is not None:
            _ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass

if not os.environ.get("DISPLAY") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
from scipy.spatial import cKDTree


# =============================================================================
# DEFAULTS; overwritten from campaign_config.json at runtime
# =============================================================================

BASE = Path("/users/mfarshad/LJ_ADAPTIVE_US")
LATEST = None  # auto-discovered from newest no-penalty trimer analysis unless --root is supplied

D_MIN = 0.95
D_MAX = 2.50
QUERY_MIN = 0.90
QUERY_MAX = 2.55

LJ_EPSILON = 1.0
LJ_SIGMA = 1.0
LJ_CUTOFF = 2.5
ATM_NU = 4.0
TRIANGLE_PENALTY_LAMBDA = 0.0
TRIANGLE_PENALTY_BETA = 25.0

# Free-energy surface coloring: match the revised Figure 7 convention.
# These settings affect DISPLAY COLORS ONLY; the surface z-heights/energies are unchanged.
LANDSCAPE_CMAP = "viridis"
LANDSCAPE_LOW_PERCENTILE = 1.0
LANDSCAPE_HIGH_PERCENTILE = 99.0
LANDSCAPE_POWER_GAMMA = 0.68  # display-only: spreads viridis through the occupied energy range
COLOR_TESSELLATION_FACTOR = 12  # linear display subdivision only; no landscape smoothing
MAX_SURFACE_RENDER_POINTS = 240  # display-only cap per axis for fast smooth 3D rendering
PRINT_TOP_CANDIDATES = 8

MINIMUM_OVERLAP = 0.10
PLANNER_OVERLAP = 0.13
STOP_MAX_GP_STD = 0.03

# Figure-2 controls. Change FIGURE2_ROWS to 2, 3, or 4 when running in Spyder.
# The command-line option --figure2-rows overrides this value. Fixed-coordinate
# targets are snapped to the nearest saved GP query-grid value.
FIGURE2_ROWS = 2
FIGURE2_D12_LOW = 1.10
FIGURE2_D12_MID = 1.70
FIGURE2_D23 = 1.70
WHAM_DISPLAY_IDW_K = 24
WHAM_DISPLAY_IDW_POWER = 2.0

# Spyder-friendly control for the latest AL/query step included in all figures.
# Change this number directly before pressing Run in Spyder.
MAX_AL_STEP = None

# Figure-3 legend control.  All three figures now share the same source-canvas
# width and the same physical panel footprint; only the number of rows/columns
# differs.  This makes the typography genuinely comparable when the complete
# figures are inserted at the same manuscript width.
FIG3_LEGEND_Y = 0.965

MAX_EXPLORATION_SAMPLES_PER_WINDOW = 220
MAX_3D_POINTS = 24000

PLOT_VIEW_ELEV = 28
PLOT_VIEW_AZIM = -140
PLOT_3D_Z_ASPECT = 0.70

# =============================================================================
# COMMON MANUSCRIPT GEOMETRY + TYPOGRAPHY
# =============================================================================
# Fairness rule used here:
#   1) every saved figure has the SAME source width;
#   2) every individual panel occupies the SAME physical width/height;
#   3) every figure uses the SAME raw point sizes;
#   4) the save routine preserves the full canvas (no bbox_inches="tight" crop).
# Therefore, if Figures 1--3 are inserted at the same final manuscript width,
# they undergo the same scale factor and the fonts/panels remain directly
# comparable.  Extra columns simply require more of the common canvas width;
# two-column figures retain deliberate side whitespace rather than enlarging
# their panels.
# Keep every panel physically identical across all three figures.  The canvas
# is widened (not the panels) to create room for the 50%-larger typography.
# This preserves the fairness rule: same source width, same panel size, same
# point sizes, and therefore the same scaling when inserted at equal manuscript width.
COMMON_FIG_W = 18.00
COMMON_PANEL_W = 4.35
COMMON_PANEL_H = 4.35
COMMON_COL_GAP_3COL = -0.5
COMMON_COL_GAP_2COL = 1.5
COMMON_ROW_GAP = 1.4

# Exactly 50% larger than the previous 18/14/16 manuscript hierarchy.
BASE_FONT_FZ = 24
PLOT_2D_LABEL_FZ = 26.25
PLOT_2D_TICK_FZ = 20.25
PLOT_2D_TITLE_FZ = 26.25
PLOT_3D_LABEL_FZ = 26.25
PLOT_3D_TICK_FZ = 20.25
SURFACE_TICK_FZ = 20.25
PLOT_3D_TITLE_FZ = 26.25
LEGEND_FZ = 18
PANEL_LABEL_FZ = 30
PNG_DPI = 450

# Legacy scale retained for compatibility with older variants.  The current
# Figure 2 and Figure 3 functions use figure-specific geometry/font sizes chosen
# for readability rather than enforcing a single common scale.
FIG23_SCALE = 2.0

# Large 3D panels need fewer distance ticks so 2x typography stays readable.
FIG23_DISTANCE_TICK_STEP = 0.50


# Point-type convention shared with the other GAPS benchmark systems.
C_WHAM = "#ff7f0e"   # shared manuscript WHAM orange
C_GP = "#3B82F6"     # shared manuscript GP blue
C_REF = "#111111"
C_COVERAGE = "#1B9E77"

TYPE_STYLE = {
    # Same high-contrast node convention used in Müller–Brown and alanine.
    "bootstrap_seed":      dict(marker="o", color="#D81B60", label="Bootstrap"),
    "bootstrap_bridge":    dict(marker="D", color="#8E44AD", label="Bootstrap bridge"),
    "connectivity_bridge": dict(marker="s", color="#E31A1C", label="Connectivity bridge"),
    "information_target":  dict(marker="^", color="#00D4FF", label="Information target"),
}

TYPE_STYLE_FALLBACK = dict(
    marker="o",
    color="#777777",
    label="Physical window",
)


def _window_sname(window):
    """Return the stored point/window type name used by the campaign."""
    val = window.get("sname")
    if val is not None and str(val).strip():
        return str(val).strip()

    # Backward-compatible fallbacks for older saved states.
    for key in ("kind", "type", "sampling_name", "source_name"):
        val = window.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    return "physical_window"


def _style_for_sname(sname):
    """Return manuscript-consistent marker/color/label for one point type."""
    if sname in TYPE_STYLE:
        return TYPE_STYLE[sname]

    out = dict(TYPE_STYLE_FALLBACK)
    out["label"] = str(sname).replace("_", " ").strip().title()
    return out


def _centered_left(ncols, panel_w=COMMON_PANEL_W, col_gap=None,
                   fig_w=COMMON_FIG_W):
    """Left edge (inches) for a centered row of equal-width panels."""
    if col_gap is None:
        col_gap = COMMON_COL_GAP_3COL if int(ncols) >= 3 else COMMON_COL_GAP_2COL
    occupied = int(ncols) * float(panel_w) + (int(ncols) - 1) * float(col_gap)
    return 0.5 * (float(fig_w) - occupied)

# 3D styling follows the older scripts that produced the clean publication plots.
FORCE_TEXT2D_ZLABEL = True
ZLABEL_2D_X = -0.105
ZLABEL_2D_Y = 0.50
XY_TICK_PAD = 0
Z_TICK_PAD = 2
SURFACE_XY_LABELPAD = 8
FULLSPACE_XY_LABELPAD = 11
AXIS_TICK_LENGTH = 4.5
AXIS_TICK_WIDTH = 1.0
FULLSPACE_FRAME_GAP_FRAC = 0.040

_PERMUTATIONS = np.asarray([
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
], dtype=int)

# Resolve the actual Arial font files with STRICT face validation.
#
# Important: font_manager.findfont(..., style="italic") can legally return the
# closest Arial face even when Arial Italic itself is absent.  That is what can
# silently produce upright labels.  Here we search Matplotlib's registered font
# table directly and verify the selected file with FreeType before accepting it.
def _ft_face_info(path):
    face = ft2font.FT2Font(str(path))
    return str(face.family_name), str(face.style_name)


def _path_is_italic(path):
    if path is None:
        return False
    try:
        _, style_name = _ft_face_info(path)
    except Exception:
        return False
    style_l = style_name.lower()
    return ("italic" in style_l) or ("oblique" in style_l)


def _path_is_bold(path):
    if path is None:
        return False
    try:
        _, style_name = _ft_face_info(path)
    except Exception:
        return False
    return "bold" in style_name.lower()


def _find_arial_face(style="normal", weight="normal"):
    """Return an ACTUAL Arial face file matching the requested style/weight.

    No approximate fontconfig match is accepted.  If Arial Italic is not
    installed/registered, this returns None instead of silently returning Arial
    Regular.
    """
    want_italic = str(style).lower() in ("italic", "oblique")
    want_bold = str(weight).lower() in ("bold", "700", "heavy", "black")

    candidates = []
    for entry in font_manager.fontManager.ttflist:
        name_l = str(entry.name).lower()
        fname_l = str(entry.fname).lower()
        if "arial" not in name_l and "arial" not in fname_l:
            continue

        try:
            family_name, style_name = _ft_face_info(entry.fname)
        except Exception:
            continue

        if "arial" not in family_name.lower() and "arial" not in name_l:
            continue

        actual_italic = ("italic" in style_name.lower()) or ("oblique" in style_name.lower())
        actual_bold = "bold" in style_name.lower()

        # Fall back to the font-manager metadata only for weight naming, because
        # some Arial files use style names such as simply "Italic" while the
        # boldness is encoded separately.
        entry_weight = str(entry.weight).lower()
        meta_bold = (
            actual_bold or "bold" in name_l or "bold" in fname_l or
            entry_weight in ("bold", "700", "heavy", "black")
        )

        if actual_italic != want_italic:
            continue
        if meta_bold != want_bold:
            continue

        candidates.append(str(entry.fname))

    # Prefer canonical Arial family names over metric-compatible substitutes or
    # oddly named variants if several matching faces are registered.
    if not candidates:
        return None
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if _ft_face_info(p)[0].lower() == "arial" else 1,
            len(p),
            p.lower(),
        ),
    )
    return candidates[0]


def _font_file_pattern(path):
    """Fontconfig pattern that pins mathtext to one exact font file."""
    if path is None:
        return None
    return font_manager.FontProperties(fname=path).get_fontconfig_pattern()


_ARIAL_REGULAR_PATH = _find_arial_face("normal", "normal")
_ARIAL_ITALIC_PATH = _find_arial_face("italic", "normal")
_ARIAL_BOLD_PATH = _find_arial_face("normal", "bold")
_ARIAL_BOLDITALIC_PATH = _find_arial_face("italic", "bold")

if _ARIAL_REGULAR_PATH is None:
    raise RuntimeError(
        "Arial Regular is not visible to Matplotlib. True all-Arial plotting "
        "cannot be produced on this machine."
    )
if _ARIAL_ITALIC_PATH is None:
    raise RuntimeError(
        "Arial Italic is NOT installed/registered in this Python/Matplotlib "
        "environment. Previous versions could silently substitute Arial Regular, "
        "which is why the labels stayed upright. Install or expose the Arial "
        "Italic face, then rerun this script."
    )

# These checks are deliberately strict: never silently draw upright variables.
_reg_family, _reg_style = _ft_face_info(_ARIAL_REGULAR_PATH)
_it_family, _it_style = _ft_face_info(_ARIAL_ITALIC_PATH)
if not _path_is_italic(_ARIAL_ITALIC_PATH):
    raise RuntimeError(
        f"Resolved italic path is not actually italic: {_ARIAL_ITALIC_PATH} "
        f"(FreeType reports family={_it_family!r}, style={_it_style!r})."
    )

# Bold faces are only needed for panel tags/titles.  Falling back to the regular
# or italic Arial face here does not affect scientific-variable italics.
if _ARIAL_BOLD_PATH is None:
    _ARIAL_BOLD_PATH = _ARIAL_REGULAR_PATH
if _ARIAL_BOLDITALIC_PATH is None:
    _ARIAL_BOLDITALIC_PATH = _ARIAL_ITALIC_PATH

_ARIAL_REGULAR_PATTERN = _font_file_pattern(_ARIAL_REGULAR_PATH)
_ARIAL_ITALIC_PATTERN = _font_file_pattern(_ARIAL_ITALIC_PATH)
_ARIAL_BOLD_PATTERN = _font_file_pattern(_ARIAL_BOLD_PATH)
_ARIAL_BOLDITALIC_PATTERN = _font_file_pattern(_ARIAL_BOLDITALIC_PATH)

_PLOT_FONT = font_manager.FontProperties(fname=_ARIAL_REGULAR_PATH).get_name()

print("Arial Regular file:", _ARIAL_REGULAR_PATH)
print("  FreeType face     :", _reg_family, "/", _reg_style)
print("Arial Italic file :", _ARIAL_ITALIC_PATH)
print("  FreeType face     :", _it_family, "/", _it_style)


def _arial_fontprops(size, italic=False, bold=False):
    """Return FontProperties pinned to an exact validated Arial face file."""
    if italic and bold:
        path = _ARIAL_BOLDITALIC_PATH
    elif italic:
        path = _ARIAL_ITALIC_PATH
    elif bold:
        path = _ARIAL_BOLD_PATH
    else:
        path = _ARIAL_REGULAR_PATH
    fp = font_manager.FontProperties(fname=path)
    fp.set_size(float(size))
    return fp


def _apply_exact_arial_face(text_obj, size, italic=False, bold=False):
    """Force an already-created Text/Text3D object onto the exact Arial face.

    Applying the file-backed FontProperties after the axis label exists avoids
    any 3D-axis label recreation/family-resolution step changing the face.
    """
    text_obj.set_fontproperties(_arial_fontprops(size, italic=italic, bold=bold))
    text_obj.set_fontsize(float(size))
    return text_obj

matplotlib.rcParams.update({
    # Ordinary text: Arial.
    "font.family": _PLOT_FONT,
    "font.sans-serif": [_PLOT_FONT],
    "font.size": BASE_FONT_FZ,
    "axes.labelsize": PLOT_2D_LABEL_FZ,
    "axes.titlesize": PLOT_2D_TITLE_FZ,
    "xtick.labelsize": PLOT_2D_TICK_FZ,
    "ytick.labelsize": PLOT_2D_TICK_FZ,
    "legend.fontsize": LEGEND_FZ,

    # Mathtext is used only for subscript/superscript characters.  The main
    # scientific letters (d, W, K) are rendered as ordinary text with the exact
    # Arial Italic font file, while mathtext supplies the lowered/raised digits.
    # This avoids missing-glyph boxes from Unicode subscript characters.
    "mathtext.fontset": "custom",
    "mathtext.rm": _ARIAL_REGULAR_PATTERN,
    "mathtext.it": _ARIAL_ITALIC_PATTERN,
    "mathtext.bf": _ARIAL_BOLD_PATTERN,
    "mathtext.bfit": _ARIAL_BOLDITALIC_PATTERN,
    "mathtext.sf": _ARIAL_REGULAR_PATTERN,
    "mathtext.tt": _ARIAL_REGULAR_PATTERN,
    "mathtext.cal": _ARIAL_ITALIC_PATTERN,
    "mathtext.default": "regular",
    "mathtext.fallback": None,

    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# =============================================================================
# IO / CAMPAIGN SELECTION
# =============================================================================

def read_json(path):
    return json.loads(Path(path).read_text())


def query_number(path):
    m = re.search(r"query_(\d+)", str(path))
    return int(m.group(1)) if m else -1


def analysis_files(root):
    root = Path(root)
    fs = list((root / "analysis_history").rglob("online_analysis.npz"))
    if not fs:
        fs = list(root.rglob("online_analysis.npz"))
    return sorted(fs, key=lambda p: (query_number(p), p.stat().st_mtime))


def _is_no_penalty_trimer_campaign(root, cfg):
    """Accept the physical Cartesian/LAMMPS trimer family and reject penalized runs."""
    name = Path(root).name.upper()
    if not any(tok in name for tok in ("TRIMER", "LJATM", "LJ_ATM")):
        return False

    penalty = cfg.get("triangle_penalty_lambda", cfg.get("TRIANGLE_PENALTY_LAMBDA", 0.0))
    try:
        if abs(float(penalty or 0.0)) > 1.0e-15:
            return False
    except Exception:
        pass

    txt = " ".join([
        name,
        str(cfg.get("dynamics", "")),
        str(cfg.get("atm_implementation", "")),
        str(cfg.get("atm_force_implementation", "")),
        str(cfg.get("potential", "")),
    ]).upper()
    return (
        "NOPENALTY" in txt or "NOPENTY" in txt or "NO PENALTY" in txt
        or "LAMMPS" in txt or "PAIR_STYLE ATM" in txt
    )


def _latest_analysis_by_mtime(root):
    fs = analysis_files(root)
    if not fs:
        raise FileNotFoundError(f"No online_analysis.npz under {root}")
    return max(fs, key=lambda p: p.stat().st_mtime)


def resolve_root(user_root=None):
    """Use explicit --root, otherwise select the trimer campaign with newest saved analysis."""
    if user_root:
        p = Path(user_root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = []
    seen = set()
    for cfg_path in BASE.rglob("campaign_config.json"):
        root = cfg_path.parent.resolve()
        if root in seen:
            continue
        seen.add(root)
        try:
            cfg = read_json(cfg_path)
        except Exception:
            continue
        if not _is_no_penalty_trimer_campaign(root, cfg):
            continue
        try:
            analysis = _latest_analysis_by_mtime(root)
        except Exception:
            continue
        candidates.append((analysis.stat().st_mtime, root, analysis.resolve(), cfg))

    if not candidates:
        raise FileNotFoundError(
            "No no-penalty trimer campaign with online_analysis.npz was found under "
            f"{BASE}. Use --root /exact/campaign/path if needed."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    from datetime import datetime
    print("Newest no-penalty trimer analysis candidates:")
    print("-" * 96)
    for rank, (mtime, root, analysis, cfg) in enumerate(candidates[:PRINT_TOP_CANDIDATES], 1):
        stamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        wh = cfg.get("wham", {}) if isinstance(cfg.get("wham", {}), dict) else {}
        print(f"{rank:2d}. {stamp}  {root.name}")
        print(f"    analysis: {analysis}")
        print(
            "    EQ={} PROD={} SAMPLE={} MIN_COUNT={}".format(
                cfg.get("equilibration_steps"), cfg.get("production_steps"),
                cfg.get("sample_every"), wh.get("min_total_counts", cfg.get("min_total_counts"))
            )
        )
    print("-" * 96)
    return candidates[0][1]


def load_state(root):
    return read_json(Path(root) / "campaign_state.json")


def completed_windows(state):
    return [w for w in state.get("windows", []) if w.get("status") == "complete"]


def latest_analysis(root, state):
    p = state.get("latest_full_analysis")
    if p and Path(p).exists():
        return Path(p)
    fs = analysis_files(root)
    if not fs:
        raise FileNotFoundError(f"No online_analysis.npz under {root}")
    return max(fs, key=lambda q: q.stat().st_mtime)

def analysis_at_or_before(root, max_query=None):
    """Return latest analysis, or newest snapshot with query <= max_query."""
    fs = analysis_files(root)
    if not fs:
        raise FileNotFoundError(f"No online_analysis.npz under {root}")
    if max_query is None:
        return max(fs, key=lambda p: (query_number(p), p.stat().st_mtime))
    fs = [p for p in fs if query_number(p) <= int(max_query)]
    if not fs:
        raise FileNotFoundError(
            f"No online_analysis.npz with query <= {int(max_query)} under {root}"
        )
    return max(fs, key=lambda p: (query_number(p), p.stat().st_mtime))


def state_for_analysis(state, z):
    """Restrict campaign state to the physical windows represented by a saved snapshot."""
    if "connected_window_ids" not in z:
        return state
    keep = set(int(x) for x in np.asarray(z["connected_window_ids"]).ravel())
    out = dict(state)
    out["windows"] = [
        w for w in state.get("windows", [])
        if int(w.get("id", -1)) in keep and w.get("status") == "complete"
    ]
    # The live campaign target may belong to a later AL step, so do not mix it
    # into an older snapshot figure.
    out["current_query"] = None
    return out


def load_npz(path):
    return dict(np.load(path, allow_pickle=False))


def campaign_config(root):
    p = Path(root) / "campaign_config.json"
    return read_json(p) if p.exists() else {}


def apply_campaign_config(cfg):
    """Make plotting physics match the campaign that actually generated the data."""
    global D_MIN, D_MAX, QUERY_MIN, QUERY_MAX
    global LJ_EPSILON, LJ_SIGMA, LJ_CUTOFF, ATM_NU
    global TRIANGLE_PENALTY_LAMBDA, TRIANGLE_PENALTY_BETA
    global MINIMUM_OVERLAP, PLANNER_OVERLAP, STOP_MAX_GP_STD

    if cfg.get("distance_domain"):
        D_MIN, D_MAX = map(float, cfg["distance_domain"])
    if cfg.get("query_halo_domain"):
        QUERY_MIN, QUERY_MAX = map(float, cfg["query_halo_domain"])

    LJ_EPSILON = float(cfg.get("lj_epsilon", LJ_EPSILON))
    LJ_SIGMA = float(cfg.get("lj_sigma", LJ_SIGMA))
    LJ_CUTOFF = float(cfg.get("lj_cutoff", LJ_CUTOFF))
    ATM_NU = float(cfg.get("atm_nu", ATM_NU))
    TRIANGLE_PENALTY_LAMBDA = float(
        cfg.get("triangle_penalty_lambda", 0.0)
    )
    TRIANGLE_PENALTY_BETA = float(
        cfg.get("triangle_penalty_beta", TRIANGLE_PENALTY_BETA)
    )

    MINIMUM_OVERLAP = float(cfg.get("minimum_overlap", MINIMUM_OVERLAP))
    PLANNER_OVERLAP = float(cfg.get("planner_overlap", PLANNER_OVERLAP))
    stop = cfg.get("stopping", {})
    STOP_MAX_GP_STD = float(stop.get("maximum_gp_std", STOP_MAX_GP_STD))


# =============================================================================
# DISPLAY / SAVE
# =============================================================================

def display_figure_in_spyder(fig):
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        ip = None

    if ip is not None:
        try:
            from IPython.display import display
            fig.canvas.draw()
            display(fig)
        except Exception:
            try:
                plt.show()
            except Exception:
                pass
    plt.close(fig)


def save_pair(fig, outdir, stem, extra_artists=None):
    """Save PNG/PDF at the exact declared canvas size.

    Do NOT use bbox_inches="tight" here.  Tight cropping changes the physical
    output width differently for each figure and defeats the equal-panel/equal-
    typography geometry established above.  All labels are kept inside the
    deliberately reserved margins instead.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"

    fig.canvas.draw()
    save_kw = dict(facecolor="white", bbox_inches=None)

    fig.savefig(png, dpi=PNG_DPI, **save_kw)
    fig.savefig(pdf, **save_kw)
    fw, fh = fig.get_size_inches()
    print(f"saved: {png}   [canvas {fw:.2f} x {fh:.2f} in]")
    print(f"saved: {pdf}   [canvas {fw:.2f} x {fh:.2f} in]")


def style_2d(ax):
    ax.tick_params(axis="both", labelsize=PLOT_2D_TICK_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def _compact_tick_label(x, pos=None):
    """Compact numeric formatter used by the older publication plots."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) < 1.0e-12:
        x = 0.0
    if np.isclose(x, np.round(x), atol=1.0e-9):
        return f"{int(np.round(x))}"
    return f"{x:g}"


COMPACT_TICK_FORMATTER = FuncFormatter(_compact_tick_label)


def _set_ortho(ax):
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def _make_3d_ticks_visible(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.28
            axis._axinfo["tick"]["linewidth"] = {
                True: AXIS_TICK_WIDTH,
                False: AXIS_TICK_WIDTH,
            }
        except Exception:
            pass


def _pin_3d_zlabel_2d(ax, label, fontsize=PLOT_3D_LABEL_FZ,
                      x=ZLABEL_2D_X, y=ZLABEL_2D_Y, italic=True):
    """Pin only the z label in 2D axes coordinates; x/y remain native 3D labels."""
    try:
        ax.set_zlabel("")
    except Exception:
        pass
    txt = ax.text2D(
        float(x), float(y), label,
        transform=ax.transAxes,
        rotation=90,
        va="center", ha="right",
        clip_on=False,
    )
    return _apply_exact_arial_face(txt, fontsize, italic=italic)


def _expanded_limit(data_lim):
    lo, hi = float(data_lim[0]), float(data_lim[1])
    span = hi - lo
    pad = FULLSPACE_FRAME_GAP_FRAC * span if span > 0.0 else 0.05
    return lo - pad, hi + pad


def _distance_ticks(lo, hi, step=0.30):
    """Clean distance ticks for the present 0.9--2.55 coordinate range."""
    lo = float(lo); hi = float(hi); step = float(step)
    start = np.ceil((lo - 1.0e-12) / step) * step
    vals = np.arange(start, hi + 0.5 * step, step)
    vals = vals[(vals >= lo - 1.0e-10) & (vals <= hi + 1.0e-10)]
    return np.round(vals, 10)


def _apply_coordinate_ticks(ax, xlim, ylim, zlim=None, show_z=True,
                            tick_fz=PLOT_3D_TICK_FZ, tick_step=0.30):
    xt = _distance_ticks(*xlim, step=tick_step)
    yt = _distance_ticks(*ylim, step=tick_step)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

    if zlim is not None and show_z:
        ax.set_zticks(_distance_ticks(*zlim, step=tick_step))
        ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

    ax.tick_params(
        axis="x", labelsize=tick_fz, pad=XY_TICK_PAD,
        length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH,
    )
    ax.tick_params(
        axis="y", labelsize=tick_fz, pad=XY_TICK_PAD,
        length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH,
    )
    ax.tick_params(
        axis="z", labelsize=tick_fz, pad=Z_TICK_PAD,
        length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH,
    )
    _make_3d_ticks_visible(ax)


def _style_energy_surface_axis(ax, xlabel, ylabel, zlim,
                               show_ylabel=True, show_zlabel=True,
                               label_fz=PLOT_3D_LABEL_FZ,
                               tick_fz=SURFACE_TICK_FZ,
                               labelpad_scale=1.0):
    """Native x/y labels + pinned z label, matching the older successful scripts."""
    axis_labelpad = SURFACE_XY_LABELPAD * float(labelpad_scale)
    ax.set_xlabel(xlabel, labelpad=axis_labelpad)
    _apply_exact_arial_face(ax.xaxis.label, label_fz, italic=True)
    if show_ylabel:
        ax.set_ylabel(ylabel, labelpad=axis_labelpad)
        _apply_exact_arial_face(ax.yaxis.label, label_fz, italic=True)
    else:
        ax.set_ylabel("")

    # x/y ticks remain present on every panel even if the y-axis label is omitted.
    ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.tick_params(axis="x", labelsize=tick_fz, pad=XY_TICK_PAD,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="y", labelsize=tick_fz, pad=XY_TICK_PAD,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="z", labelsize=tick_fz, pad=Z_TICK_PAD * float(labelpad_scale),
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)

    # Clean, publication-friendly z ticks; no long floating point labels.
    try:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, min_n_ticks=3))
    except TypeError:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

    _set_ortho(ax)
    ax.view_init(elev=PLOT_VIEW_ELEV, azim=PLOT_VIEW_AZIM)
    _make_3d_ticks_visible(ax)
    try:
        ax.set_box_aspect((1.0, 1.0, PLOT_3D_Z_ASPECT))
    except Exception:
        pass

    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor((0.80, 0.80, 0.80, 1.0))
            axis._axinfo["grid"]["color"] = (0.80, 0.80, 0.80, 0.48)
            axis._axinfo["grid"]["linewidth"] = 0.55
        except Exception:
            pass

    z_art = None
    if show_zlabel:
        # Figure 2 only: halve the horizontal gap between W_3 and the z axis.
        z_art = _pin_3d_zlabel_2d(
            ax, r"W$_{\mathrm{3}}$ $\mathrm{(}k_{\mathrm{B}}T\mathrm{)}$", label_fz,
            x=0.5 * ZLABEL_2D_X, y=ZLABEL_2D_Y,
        )
    else:
        ax.set_zlabel("")
    return z_art


def _style_fullspace_axis(ax, xlabel, ylabel, zlabel,
                          data_xlim, data_ylim, data_zlim,
                          show_zlabel=True, show_zticks=True,
                          label_fz=PLOT_3D_LABEL_FZ,
                          tick_fz=PLOT_3D_TICK_FZ,
                          labelpad_scale=1.0,
                          tick_step=0.30):
    """Full 3D configuration-space styling from the older Figure 4/6 scripts."""
    axis_labelpad = FULLSPACE_XY_LABELPAD * float(labelpad_scale)
    ax.set_xlabel(xlabel, labelpad=axis_labelpad)
    ax.set_ylabel(ylabel, labelpad=axis_labelpad)
    _apply_exact_arial_face(ax.xaxis.label, label_fz, italic=False)
    _apply_exact_arial_face(ax.yaxis.label, label_fz, italic=False)

    xlim = _expanded_limit(data_xlim)
    ylim = _expanded_limit(data_ylim)
    zlim = _expanded_limit(data_zlim)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)

    # Use clean 0.3-spaced distance ticks on the data range, not the padded frame.
    _apply_coordinate_ticks(
        ax, data_xlim, data_ylim, data_zlim,
        show_z=show_zticks, tick_fz=tick_fz, tick_step=tick_step,
    )
    if not show_zticks:
        ax.set_zticks([])

    _set_ortho(ax)
    ax.view_init(elev=PLOT_VIEW_ELEV, azim=PLOT_VIEW_AZIM)
    try:
        ax.set_box_aspect((
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            0.60 * (zlim[1] - zlim[0]),
        ))
    except Exception:
        pass

    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
            axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.45)
            axis._axinfo["grid"]["linewidth"] = 0.55
        except Exception:
            pass

    z_art = None
    if show_zlabel:
        # Full-space plots: keep the pinned z label closer to the panel than
        # the energy-surface default.  This preserves the robust 2D pinning
        # while roughly halving the visual gap to the frame.
        z_art = _pin_3d_zlabel_2d(
            ax, zlabel, label_fz,
            x=0.5 * ZLABEL_2D_X, y=ZLABEL_2D_Y,
            italic=False,
        )
    else:
        ax.set_zlabel("")
    return z_art


def _add_axes_inches(fig, fig_w, fig_h, x, y, w, h, projection=None):
    return fig.add_axes(
        [x / fig_w, y / fig_h, w / fig_w, h / fig_h],
        projection=projection,
    )


# =============================================================================
# EXACT BENCHMARK — diagnostic only
# =============================================================================

def canonical(points):
    return np.sort(np.atleast_2d(np.asarray(points, float)), axis=1)


def symmetry_expand(points, values=None):
    p = np.atleast_2d(np.asarray(points, float))
    blocks = [p[:, perm] for perm in _PERMUTATIONS]
    out = np.vstack(blocks) if blocks else np.empty((0, 3), float)
    if values is None:
        return out
    v = np.asarray(values, float).ravel()
    return out, np.tile(v, len(_PERMUTATIONS))


def shifted_lj(r):
    r = np.asarray(r, float)
    rc = float(LJ_CUTOFF)
    rc_energy = 4.0 * LJ_EPSILON * (
        (LJ_SIGMA / rc) ** 12 - (LJ_SIGMA / rc) ** 6
    )
    out = np.zeros_like(r)
    mask = r < rc
    inv = LJ_SIGMA / r[mask]
    out[mask] = 4.0 * LJ_EPSILON * (inv ** 12 - inv ** 6) - rc_energy
    return out


def _softplus(x):
    beta = float(TRIANGLE_PENALTY_BETA)
    x = np.asarray(x, float)
    bx = beta * x
    return np.where(
        bx > 40.0,
        x,
        np.where(bx < -40.0, np.exp(bx) / beta, np.log1p(np.exp(bx)) / beta),
    )


def triangle_penalty(points):
    p = np.atleast_2d(np.asarray(points, float))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    v = np.column_stack([x - y - z, y - x - z, z - x - y])
    s = _softplus(v)
    return TRIANGLE_PENALTY_LAMBDA * np.sum(s * s, axis=1)


def atm_energy(points):
    p = np.atleast_2d(np.asarray(points, float))
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    eps = 1.0e-15
    cos1 = (a * a + b * b - c * c) / np.maximum(2.0 * a * b, eps)
    cos2 = (a * a + c * c - b * b) / np.maximum(2.0 * a * c, eps)
    cos3 = (b * b + c * c - a * a) / np.maximum(2.0 * b * c, eps)
    denom = np.maximum((a * b * c) ** 3, 1.0e-300)
    return ATM_NU * (1.0 + 3.0 * cos1 * cos2 * cos3) / denom


def exact_w3(points):
    p = np.atleast_2d(np.asarray(points, float))
    return np.sum(shifted_lj(p), axis=1) + atm_energy(p)


# =============================================================================
# FIGURE 1 — FOUR-PANEL DIAGNOSTICS
# =============================================================================

def collect_history(root, max_query=None):
    rows = []
    for p in analysis_files(root):
        if max_query is not None and query_number(p) > int(max_query):
            continue
        try:
            z = load_npz(p)
            nwin = (
                len(np.asarray(z["connected_window_ids"]).ravel())
                if "connected_window_ids" in z else np.nan
            )
            gp_dense = (
                float(np.asarray(z["diagnostic_dense_gp_exact_rmse"]).ravel()[0])
                if "diagnostic_dense_gp_exact_rmse" in z else np.nan
            )
            gp_support = (
                float(np.asarray(z["diagnostic_support_gp_exact_rmse"]).ravel()[0])
                if "diagnostic_support_gp_exact_rmse" in z else np.nan
            )
            wham_support = (
                float(np.asarray(z["diagnostic_support_wham_exact_rmse"]).ravel()[0])
                if "diagnostic_support_wham_exact_rmse" in z else np.nan
            )
            sigma = np.nan
            if "candidate_std" in z:
                cs = np.asarray(z["candidate_std"], float)
                finite = cs[np.isfinite(cs)]
                if len(finite):
                    sigma = float(np.max(finite))
            support_cells = (
                int(np.count_nonzero(np.asarray(z["support"], bool)))
                if "support" in z else np.nan
            )
            canonical_labels = (
                len(np.asarray(z["canonical_training_x"]))
                if "canonical_training_x" in z else np.nan
            )
            rows.append(dict(
                query=query_number(p),
                windows=float(nwin),
                gp_dense=gp_dense,
                gp_support=gp_support,
                wham_support=wham_support,
                sigma=sigma,
                support_cells=float(support_cells),
                canonical_labels=float(canonical_labels),
                path=p,
            ))
        except Exception as exc:
            print("WARNING: skipped history snapshot", p, exc)

    dedup = {}
    for r in rows:
        dedup[r["query"]] = r
    out = list(dedup.values())
    out.sort(key=lambda r: r["query"])
    return out


def overlap_key(i, j):
    a, b = sorted((int(i), int(j)))
    return f"{a}:{b}"


def parent_overlap_history(root, state):
    p = Path(root) / "analysis_current" / "measured_overlap_cache.json"
    cache = read_json(p) if p.exists() else {}
    xs, ys = [], []
    for w in completed_windows(state):
        wid = int(w["id"])
        parent = w.get("parent_window")
        if parent is None:
            continue
        rec = cache.get(overlap_key(wid, int(parent)))
        if not isinstance(rec, dict):
            continue
        ov = rec.get("overlap")
        if ov is None or not np.isfinite(float(ov)):
            continue
        xs.append(wid)
        ys.append(float(ov))
    return np.asarray(xs, float), np.asarray(ys, float)


def figure1_diagnostics(root, state, z, outdir, max_al_step=None):
    """Four-panel diagnostics figure with square panels and larger typography."""
    # Figure-1-only typography, consistent with the enlarged main-text figures.
    LABEL_FZ = 32.0
    TICK_FZ = 25.0
    LEGEND1_FZ = 26.0
    PANEL_TAG_FZ = 36.0

    # Richer publication blue for the series that previously used
    # Matplotlib's default blue.
    BLUE = C_GP

    # Equal-panel geometry: Figure 1 uses the same 18-in source width and the
    # same 4.35 x 4.35-in panel boxes as Figures 2 and 3.  Because Figure 1 has
    # only two columns, the extra horizontal space is kept as symmetric margins
    # rather than allowing its panels to become physically larger.
    FIG_W = COMMON_FIG_W
    PANEL_W = COMMON_PANEL_W
    PANEL_H = COMMON_PANEL_H
    # Figure 1 only: double the original inter-panel spacing.
    # Original values were 1.5 in (columns) and 1.4 in (rows).
    COL_GAP = 2.0
    ROW_GAP = 2.
    LEFT = _centered_left(2, PANEL_W, COL_GAP, FIG_W)
    BOTTOM = 1.25
    TOP_MARGIN = 1.25
    FIG_H = BOTTOM + 2 * PANEL_H + ROW_GAP + TOP_MARGIN

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = np.empty((2, 2), dtype=object)
    for r in range(2):
        y = BOTTOM + (1 - r) * (PANEL_H + ROW_GAP)
        for c in range(2):
            x = LEFT + c * (PANEL_W + COL_GAP)
            ax[r, c] = _add_axes_inches(fig, FIG_W, FIG_H, x, y, PANEL_W, PANEL_H)
            try:
                ax[r, c].set_box_aspect(1)
            except Exception:
                pass

    print(
        f"Figure 1 geometry: canvas={FIG_W:.2f} in wide; "
        f"each panel={PANEL_W:.2f} x {PANEL_H:.2f} in"
    )

    hist = collect_history(root, max_query=max_al_step)
    n = np.asarray([r["windows"] for r in hist], float)
    q = np.asarray([r["query"] for r in hist], float)
    gd = np.asarray([r["gp_dense"] for r in hist], float)
    gs = np.asarray([r["gp_support"] for r in hist], float)
    ws = np.asarray([r["wham_support"] for r in hist], float)
    support_cells = np.asarray([r["support_cells"] for r in hist], float)
    canonical_labels = np.asarray([r["canonical_labels"] for r in hist], float)

    a = ax[0, 0]
    exact_train = np.asarray(z.get("exact_training_aligned", []), float)
    wham_train = np.asarray(z.get("canonical_training_y", []), float)
    goodw = np.isfinite(exact_train) & np.isfinite(wham_train)
    rv, wv = exact_train[goodw], wham_train[goodw]
    if "parity_dense_exact" in z and "parity_dense_gp" in z:
        gx = np.asarray(z["parity_dense_exact"], float)
        gy = np.asarray(z["parity_dense_gp"], float)
    else:
        gx = exact_train.copy()
        gy = np.asarray(z.get("canonical_training_pred", []), float)
    goodg = np.isfinite(gx) & np.isfinite(gy)
    gx, gy = gx[goodg], gy[goodg]
    parts = [v for v in (rv, wv, gx, gy) if len(v)]
    lim = max(1.0, float(np.nanpercentile(np.concatenate(parts), 98.5))) if parts else 1.0
    if len(rv):
        keep = rv <= lim
        a.scatter(rv[keep], wv[keep], s=28, alpha=0.28, color=C_WHAM, label="WHAM")
    if len(gx):
        keep = gx <= lim
        a.scatter(gx[keep], gy[keep], s=28, alpha=0.25, color=BLUE, label="GP")
    a.plot([0, lim], [0, lim], "--", lw=1.2, color="0.35")
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal", adjustable="box")
    a.set_xlabel(r"$W_3^{\mathrm{Ref}}$ ($k_{\mathrm{B}}T$)", fontproperties=_arial_fontprops(LABEL_FZ, italic=True))
    a.set_ylabel(r"$W_3^{\mathrm{GP}},\ W_3^{\mathrm{WHAM}}$ ($k_{\mathrm{B}}T$)", fontproperties=_arial_fontprops(LABEL_FZ, italic=True))
    style_2d(a); a.tick_params(axis="both", labelsize=TICK_FZ)
    leg_a = a.legend(
                    frameon=False,
                    loc="upper left",
                    fontsize=LEGEND1_FZ,
                    handlelength=0.5,     # shorter line/stick
                    handletextpad=0.35,   # smaller gap between symbol and text
                    )
    
    for txt in leg_a.get_texts():
        txt.set_color("black")
        txt.set_alpha(1.0)

    # Only the legend marker samples are made fully opaque.
    # The actual scatter points in Panel A keep their original pale alpha.
    legend_handles_a = getattr(
        leg_a, "legend_handles",
        getattr(leg_a, "legendHandles", [])
    )
    for h in legend_handles_a:
        try:
            h.set_alpha(1.0)
            h.set_sizes([110])   # larger legend symbol only; scatter points unchanged
        except Exception:
            pass
    a.text(-0.075, 1.045, "A", transform=a.transAxes,
           ha="left", va="bottom",
           fontsize=PANEL_TAG_FZ, fontweight="bold", clip_on=False)

    b = ax[0, 1]
    for vals, marker, label, lw, color in (
        # (gs, "^", "GP support", 1.7, "#ff7f0e"),
        (ws, "s", "WHAM", 2.0, C_WHAM),
        (gd, "o", "GP", 2.2, BLUE),
    ):
        m = np.isfinite(n) & np.isfinite(vals)
        if np.any(m):
            b.plot(
                n[m], vals[m],
                marker=marker, ms=9.6, lw=lw,
                color=color, label=label
            )
    b.set_xlabel("Number of windows", fontsize=LABEL_FZ)
    b.set_ylabel(r"RMSE ($k_{\mathrm{B}}T$)", fontsize=LABEL_FZ)
    style_2d(b); b.tick_params(axis="both", labelsize=TICK_FZ)
    leg_b = b.legend(
                    frameon=False,
                    loc="upper right",
                    fontsize=LEGEND1_FZ,
                    handlelength=0.9,     # shorter line/stick
                    handletextpad=0.35,   # smaller gap between symbol and text
                    )
    for txt in leg_b.get_texts():
        txt.set_color("black")
        txt.set_alpha(1.0)
    b.text(-0.075, 1.045, "B", transform=b.transAxes,
           ha="left", va="bottom",
           fontsize=PANEL_TAG_FZ, fontweight="bold", clip_on=False)

    c = ax[1, 0]
    # Cumulative graph cost versus active-learning / information iteration.
    # The solid curve is the actual cumulative number of physical windows/nodes.
    # The dashed curve is the ideal cost of exactly one new node per target.
    good_n = np.isfinite(q) & np.isfinite(n)
    qn = q[good_n]
    nn = n[good_n]
    if len(nn):
        c.plot(
            qn, nn,
            marker="o", ms=9.6, lw=2.2, color=BLUE, markeredgecolor="none",
            label="Physical graph nodes",
        )
        offset = float(nn[0] - qn[0])
        c.plot(
            qn, qn + offset,
            "--", lw=1.35, color="0.45",
            label="One node per target",
        )
        c.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    c.set_xlabel("AL step", fontsize=LABEL_FZ)
    c.set_ylabel("Number of windows", fontsize=LABEL_FZ)
    style_2d(c); c.tick_params(axis="both", labelsize=TICK_FZ)
    c.text(-0.075, 1.045, "C", transform=c.transAxes,
           ha="left", va="bottom",
           fontsize=PANEL_TAG_FZ, fontweight="bold", clip_on=False)

    d = ax[1, 1]
    done = completed_windows(state)
    ids = np.asarray([int(w["id"]) for w in done], float)
    ks = np.asarray([float(w.get("spring_k")) if w.get("spring_k") is not None else np.nan for w in done], float)
    good = np.isfinite(ids) & np.isfinite(ks) & (ks > 0)

    # Apply the generic 2D styling first.  The discrete K ticks are then set
    # explicitly so the generic MaxNLocator cannot overwrite them.
    style_2d(d)
    d.tick_params(axis="both", labelsize=TICK_FZ)
    if np.any(good):
        d.plot(ids[good], ks[good], marker="o", ms=9.0, lw=1.8, color=BLUE, markeredgecolor="none")
        d.set_yscale("log", base=2)
        k_levels = np.unique(np.round(ks[good], 10)); k_levels = np.sort(k_levels)
        if len(k_levels) > 10:
            idx = np.linspace(0, len(k_levels) - 1, 8, dtype=int)
            k_levels = k_levels[idx]
        d.set_yticks(k_levels)
        k_tick_labels = [
            f"{int(round(v))}" if abs(v - round(v)) < 1e-8 else f"{v:g}"
            for v in k_levels
        ]
        d.set_yticklabels(k_tick_labels, fontproperties=_arial_fontprops(TICK_FZ))
        try:
            d.yaxis.offsetText.set_visible(False)
        except Exception:
            pass
    d.set_xlabel("Number of windows", fontsize=LABEL_FZ)
    d.set_ylabel("K", fontproperties=_arial_fontprops(LABEL_FZ, italic=True))
    d.text(-0.075, 1.045, "D", transform=d.transAxes,
           ha="left", va="bottom",
           fontsize=PANEL_TAG_FZ, fontweight="bold", clip_on=False)

    save_pair(fig, outdir, "Figure1_GAPS_Diagnostics_4Panel")
    display_figure_in_spyder(fig)



def unique_symmetry_points(points, values):
    """Expand symmetry-equivalent points, then keep each labeled coordinate once."""
    p, v = symmetry_expand(points, values)
    rounded = np.round(p, 12)
    _, keep = np.unique(rounded, axis=0, return_index=True)
    keep = np.sort(keep)
    return p[keep], np.asarray(v, float)[keep]


def idw_interpolate(training_x, training_y, target_points,
                    k=WHAM_DISPLAY_IDW_K, power=WHAM_DISPLAY_IDW_POWER):
    """Display-only symmetry-aware inverse-distance interpolation of WHAM labels."""
    tx, ty = unique_symmetry_points(
        np.asarray(training_x, float),
        np.asarray(training_y, float),
    )
    q = np.atleast_2d(np.asarray(target_points, float))
    if not len(tx):
        return np.full(len(q), np.nan, float)

    tree = cKDTree(tx)
    kk = min(int(k), len(tx))
    dist, idx = tree.query(q, k=kk)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    vals = ty[idx]
    exact = dist[:, 0] < 1.0e-12
    w = 1.0 / np.maximum(dist, 1.0e-8) ** float(power)
    pred = np.sum(w * vals, axis=1) / np.sum(w, axis=1)
    pred[exact] = vals[exact, 0]
    return pred


def saved_gp_grid_values(z, target_points):
    """Read saved GP values at canonical query-grid points; no GP refit."""
    bp = np.asarray(z["gp_base_points"], float)
    bm = np.asarray(z["gp_base_mean"], float).ravel()
    tree = cKDTree(bp)
    qcanon = canonical(target_points)
    dist, idx = tree.query(qcanon, k=1)
    if np.nanmax(dist) > 1.0e-7:
        raise RuntimeError(
            "Figure-2 target points are not on the saved GP query grid; "
            f"largest canonical-grid mismatch = {float(np.nanmax(dist)):.3e}"
        )
    return bm[idx]


def _nearest_saved_axis_value(axis, target):
    axis = np.asarray(axis, float)
    return float(axis[np.argmin(np.abs(axis - float(target)))])


def _figure2_slice_specs(nrows, axis):
    """
    Return ordered Figure-2 slice definitions.

    2 rows:
        1) fixed d12 = low
        2) isosceles d13 = d23

    3 rows:
        1) fixed d12 = low
        2) fixed d12 = mid
        3) isosceles d13 = d23

    4 rows:
        1) fixed d12 = low
        2) fixed d12 = mid
        3) fixed d23 = chosen value
        4) isosceles d13 = d23
    """
    nrows = int(nrows)
    if nrows not in (2, 3, 4):
        raise ValueError(f"Figure 2 rows must be 2, 3, or 4; got {nrows}")

    d12_low = _nearest_saved_axis_value(axis, FIGURE2_D12_LOW)
    d12_mid = _nearest_saved_axis_value(axis, FIGURE2_D12_MID)
    d23_fix = _nearest_saved_axis_value(axis, FIGURE2_D23)

    if nrows == 2:
        # Compact manuscript default: keep the most complementary two slices
        # from the 3-row figure (current rows 2 and 3).
        return [
            ("fixed_d12", d12_mid),
            ("isosceles", None),
        ]

    if nrows == 3:
        return [
            ("fixed_d12", d12_low),
            ("fixed_d12", d12_mid),
            ("isosceles", None),
        ]

    return [
        ("fixed_d12", d12_low),
        ("fixed_d12", d12_mid),
        ("fixed_d23", d23_fix),
        ("isosceles", None),
    ]


def _build_figure2_slice(axis, kind, value):
    """
    Build display coordinates and 3D d-space target points for one Figure-2 row.

    Display axes are intentionally chosen as follows:
      fixed d12: x=d23, y=d13
      fixed d23: x=d13, y=d12
      isosceles:  x=d13=d23, y=d12
    """
    if kind == "fixed_d12":
        # x = d23, y = d13, with d12 fixed.
        X, Y = np.meshgrid(axis, axis, indexing="xy")
        pts = np.column_stack([
            np.full(X.size, float(value)),
            Y.ravel(),
            X.ravel(),
        ])
        row_label = rf"fixed d$_{{\mathrm{{12}}}}$={float(value):.2f}"
        xlabel = r"d$_{\mathrm{23}}$"
        ylabel = r"d$_{\mathrm{13}}$"

    elif kind == "fixed_d23":
        # x = d13, y = d12, with d23 fixed.
        X, Y = np.meshgrid(axis, axis, indexing="xy")
        pts = np.column_stack([
            Y.ravel(),
            X.ravel(),
            np.full(X.size, float(value)),
        ])
        row_label = rf"fixed d$_{{\mathrm{{23}}}}$={float(value):.2f}"
        xlabel = r"d$_{\mathrm{13}}$"
        ylabel = r"d$_{\mathrm{12}}$"

    elif kind == "isosceles":
        # x = s=d13=d23, y = d12.
        X, Y = np.meshgrid(axis, axis, indexing="xy")
        pts = np.column_stack([
            Y.ravel(),
            X.ravel(),
            X.ravel(),
        ])
        row_label = r"d$_{\mathrm{13}}$ = d$_{\mathrm{23}}$"
        xlabel = r"d$_{\mathrm{13}}$ = d$_{\mathrm{23}}$"
        ylabel = r"d$_{\mathrm{12}}$"

    else:
        raise ValueError(f"Unknown Figure-2 slice kind: {kind}")

    return X, Y, pts, row_label, xlabel, ylabel


def refine_surface_for_display(X, Y, Z, factor=COLOR_TESSELLATION_FACTOR):
    """Linear display-only subdivision for finer color gradation; no spline/Gaussian smoothing."""
    if int(factor) <= 1:
        return np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)

    x = np.asarray(X[0, :], float)
    y = np.asarray(Y[:, 0], float)
    z = np.asarray(Z, float)
    nx = (len(x) - 1) * int(factor) + 1
    ny = (len(y) - 1) * int(factor) + 1
    xf = np.linspace(x[0], x[-1], nx)
    yf = np.linspace(y[0], y[-1], ny)

    zx = np.full((len(y), nx), np.nan, float)
    for j in range(len(y)):
        good = np.isfinite(z[j, :])
        if np.count_nonzero(good) >= 2:
            zx[j, :] = np.interp(xf, x[good], z[j, good])

    zf = np.full((ny, nx), np.nan, float)
    for i in range(nx):
        good = np.isfinite(zx[:, i])
        if np.count_nonzero(good) >= 2:
            zf[:, i] = np.interp(yf, y[good], zx[good, i])

    Xf, Yf = np.meshgrid(xf, yf, indexing="xy")
    return Xf, Yf, zf


def cap_surface_for_render(X, Y, Z, max_points=MAX_SURFACE_RENDER_POINTS):
    """Display-only mesh cap; preserves the refined surface values that are retained."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    Z = np.asarray(Z, float)
    nr, nc = Z.shape

    if nr <= int(max_points) and nc <= int(max_points):
        return X, Y, Z

    ir = np.arange(nr) if nr <= int(max_points) else np.unique(
        np.linspace(0, nr - 1, int(max_points)).round().astype(int)
    )
    ic = np.arange(nc) if nc <= int(max_points) else np.unique(
        np.linspace(0, nc - 1, int(max_points)).round().astype(int)
    )
    return X[np.ix_(ir, ic)], Y[np.ix_(ir, ic)], Z[np.ix_(ir, ic)]


def continuous_surface_facecolors(Z, vmin, vmax, cmap_name=LANDSCAPE_CMAP):
    """Linear viridis colors from cell-centered energy; no energy smoothing."""
    z = np.asarray(Z, float)
    zcell = 0.25 * (
        z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:]
    )
    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    cmap = plt.get_cmap(cmap_name)
    fc = cmap(norm(zcell))
    bad = ~np.isfinite(zcell)
    if np.any(bad):
        fc[bad, 3] = 0.0
    return fc


def figure2_mixed_slices(z, outdir, nrows=FIGURE2_ROWS):
    """Exact/WHAM/GP mixed slices using the proven older publication layout."""
    required = [
        "gp_base_points", "gp_base_mean",
        "canonical_training_x", "canonical_training_y",
        "exact_reference_shift",
    ]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"Figure 2 missing saved arrays: {missing}")

    base = np.asarray(z["gp_base_points"], float)
    all_axis = np.unique(np.round(base.ravel(), 12))
    axis = all_axis[(all_axis >= D_MIN - 1.0e-10) & (all_axis <= D_MAX + 1.0e-10)]
    axis = np.asarray(axis, float)
    if len(axis) < 4:
        raise RuntimeError("Could not recover the saved GP display grid.")

    specs = _figure2_slice_specs(nrows, axis)
    exact_shift = float(np.asarray(z["exact_reference_shift"]).ravel()[0])
    train_x = np.asarray(z["canonical_training_x"], float)
    train_y = np.asarray(z["canonical_training_y"], float)

    rows = []
    for kind, value in specs:
        X, Y, pts, row_label, xlabel, ylabel = _build_figure2_slice(axis, kind, value)
        ex = (exact_w3(pts) + exact_shift).reshape(X.shape)
        wh = idw_interpolate(
            train_x, train_y, pts,
            k=WHAM_DISPLAY_IDW_K,
            power=WHAM_DISPLAY_IDW_POWER,
        ).reshape(X.shape)
        gp = saved_gp_grid_values(z, pts).reshape(X.shape)

        finite = [a[np.isfinite(a)] for a in (ex, wh, gp) if np.any(np.isfinite(a))]
        if not finite:
            raise RuntimeError(f"No finite values for Figure-2 row {row_label}")

        common_shift = float(np.nanmin(np.concatenate(finite)))
        ex = ex - common_shift
        wh = wh - common_shift
        gp = gp - common_shift

        vals = np.concatenate([a[np.isfinite(a)] for a in (ex, wh, gp)])
        zmin = 0.0
        zmax = max(1.0, 1.03 * float(np.nanmax(vals)))

        # COLOR mapping only: shared Ref/WHAM/GP normalization for this row.
        # The surface heights retain the full unmodified energy range.
        cmin = float(np.nanpercentile(vals, LANDSCAPE_LOW_PERCENTILE))
        cmax = float(np.nanpercentile(vals, LANDSCAPE_HIGH_PERCENTILE))
        if not np.isfinite(cmin):
            cmin = float(np.nanmin(vals))
        if not np.isfinite(cmax) or cmax <= cmin + 1.0e-12:
            cmax = float(np.nanmax(vals))
        rows.append((X, Y, xlabel, ylabel, ex, wh, gp, zmin, zmax, cmin, cmax))

    nrows = len(rows)

    # One GLOBAL Figure-2 color normalization: the same W3 energy maps to the
    # same viridis color in every Ref/WHAM/GP panel and every slice row.
    color_vals = []
    for _, _, _, _, ex, wh, gp, _, _, _, _ in rows:
        for arr in (ex, wh, gp):
            color_vals.append(np.asarray(arr, float)[np.isfinite(arr)])
    color_vals = np.concatenate([v for v in color_vals if len(v)])
    figure2_cmin = float(np.nanpercentile(color_vals, LANDSCAPE_LOW_PERCENTILE))
    figure2_cmax = float(np.nanpercentile(color_vals, LANDSCAPE_HIGH_PERCENTILE))
    if not np.isfinite(figure2_cmin):
        figure2_cmin = float(np.nanmin(color_vals))
    if not np.isfinite(figure2_cmax) or figure2_cmax <= figure2_cmin + 1.0e-12:
        figure2_cmax = float(np.nanmax(color_vals))

    # Same display-only nonlinear normalization used by the revised Figure 7.
    # gamma < 1 expands the lower/intermediate-energy viridis range and avoids
    # the visually dominant two-band appearance without changing any energies.
    figure2_norm = PowerNorm(
        gamma=LANDSCAPE_POWER_GAMMA,
        vmin=figure2_cmin,
        vmax=figure2_cmax,
        clip=True,
    )
    print(
        f"Figure-2 color scale: [{figure2_cmin:.4g}, {figure2_cmax:.4g}] kBT; "
        f"percentiles={LANDSCAPE_LOW_PERCENTILE:g}-{LANDSCAPE_HIGH_PERCENTILE:g}, "
        f"PowerNorm gamma={LANDSCAPE_POWER_GAMMA:g}"
    )

    # Figure 2: prioritize a clean publication layout rather than forcing
    # geometric/font equality with the other figures.  These dimensions give
    # the 3D surfaces enough room while keeping labels and ticks readable.
    PANEL_W = 5.35
    PANEL_H = 4.10
    COL_GAP = COMMON_COL_GAP_3COL
    ROW_GAP = 0.55
    LEFT = 0.72
    RIGHT = 1.05
    BOTTOM = 0.72
    TOP_MARGIN = 0.68
    FIG_W = LEFT + 3 * PANEL_W + 2 * COL_GAP + RIGHT
    FIG_H = BOTTOM + nrows * PANEL_H + (nrows - 1) * ROW_GAP + TOP_MARGIN

    # Figure-2-specific typography.  These values are intentionally moderate:
    # large enough for publication, but not so large that mplot3d labels collide.
    label_fz = 26.25
    tick_fz = 20.25
    title_fz = 26.25
    panel_fz = 30

    print(
        f"Figure 2 geometry (clean layout): canvas={FIG_W:.2f} in wide; "
        f"each panel={PANEL_W:.2f} x {PANEL_H:.2f} in; rows={nrows}"
    )

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    methods = (
        r"$\mathit{W}_3^{\mathrm{Ref}}$",
        r"$\mathit{W}_3^{\mathrm{WHAM}}$",
        r"$\mathit{W}_3^{\mathrm{GP}}$",
    )
    row_tags = ("A", "B", "C", "D")
    extra_artists = []

    def col_x(c):
        return LEFT + c * (PANEL_W + COL_GAP)

    def row_y(r):
        # r=0 is the top row.
        return BOTTOM + (nrows - 1 - r) * (PANEL_H + ROW_GAP)

    for r, (X, Y, xlabel, ylabel, ex, wh, gp, zmin, zmax, _cmin, _cmax) in enumerate(rows):
        cmin, cmax = figure2_cmin, figure2_cmax
        for c, Z in enumerate((ex, wh, gp)):
            ax = _add_axes_inches(
                fig, FIG_W, FIG_H,
                col_x(c), row_y(r), PANEL_W, PANEL_H,
                projection="3d",
            )

            ZZ = np.asarray(Z, float)
            Xplot, Yplot, Zplot = refine_surface_for_display(X, Y, ZZ)
            Xplot, Yplot, Zplot = cap_surface_for_render(Xplot, Yplot, Zplot)
            ax.plot_surface(
                Xplot, Yplot, Zplot,
                cmap=LANDSCAPE_CMAP, norm=figure2_norm,
                linewidth=0, edgecolor="none",
                antialiased=True, shade=False,
                rstride=1, cstride=1,
                rasterized=True,
            )

            xlim = (float(np.nanmin(X)), float(np.nanmax(X)))
            ylim = (float(np.nanmin(Y)), float(np.nanmax(Y)))

            # Keep the full y tick set through 2.5, but extend only the visible
            # y-axis frame a little beyond the data.  This moves the terminal
            # Give the far y-axis end a little extra projected room so the
            # y=2.5 label does not crowd the z=0 label at the back-left corner.
            # This changes only the displayed frame; the actual y ticks/data
            # remain on the true physical range.
            y_frame_pad = 0.11 * (ylim[1] - ylim[0])
            ylim_display = (ylim[0], ylim[1] + y_frame_pad)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim_display)
            ax.set_zlim(zmin, zmax)

            # Ticks still come from the true data range, so 2.5 remains present.
            xt = _distance_ticks(*xlim, step=FIG23_DISTANCE_TICK_STEP)
            yt = _distance_ticks(*ylim, step=FIG23_DISTANCE_TICK_STEP)

            ax.set_xticks(xt)
            ax.set_yticks(yt)
            ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
            ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

            z_art = _style_energy_surface_axis(
                ax,
                xlabel=xlabel,
                ylabel=ylabel,
                zlim=(zmin, zmax),
                show_ylabel=True,
                show_zlabel=(c == 0),
                label_fz=label_fz,
                tick_fz=tick_fz,
                labelpad_scale=1.45,
            )
            
            # Force 0, 2, 4, 6 on the upper row
            if r == 0:
                ax.set_zlim(zmin, max(zmax, 6.0))
                ax.set_zticks([0, 2, 4, 6])
                ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
                
            # Keep identical x/y tick padding.  Unequal padding moves the two
            # front-corner labels in different projected directions and worsens
            # their visual alignment.
            ax.tick_params(axis="x", pad=XY_TICK_PAD)
            ax.tick_params(axis="y", pad=XY_TICK_PAD)
            ax.tick_params(axis="z", pad=Z_TICK_PAD)

            # Use a symmetric azimuth for the surface panels.  At -135 degrees,
            # the projected x and y axes are mirror-symmetric, which brings the
            # front x=1 and y=1 tick labels onto essentially the same baseline.
            ax.view_init(elev=PLOT_VIEW_ELEV, azim=-135)

            if z_art is not None:
                extra_artists.append(z_art)

            if r == 0:
                ax.set_title(
                    methods[c],
                    fontproperties=_arial_fontprops(title_fz),
                    pad=3,
                )
            else:
                ax.set_title("")

            # Sequential panel labels, consistent with the earlier benchmark figures.
            panel_letter = chr(ord("A") + 3*r + c)
            tag = ax.text2D(
                -0.025, 0.92, panel_letter,
                transform=ax.transAxes,
                ha="left", va="top",
                fontproperties=_arial_fontprops(panel_fz, bold=True),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.3),
                clip_on=False,
            )
            extra_artists.append(tag)

    save_pair(
        fig, outdir,
        f"Figure2_{nrows}Rows_MixedSlices_Ref_WHAM_GP",
        extra_artists=extra_artists,
    )
    display_figure_in_spyder(fig)


# =============================================================================
# FIGURE 3 — FULL / PERMUTATION-REDUCED NODES + CONNECTIVITY + DISTRIBUTIONS
# =============================================================================

def sample_path(root, wid, rep):
    return (
        Path(root) / "windows" / f"window_{int(wid):06d}"
        / f"rep_{int(rep):02d}" / "samples.dat"
    )


def load_window_samples(root, wid, nrep):
    chunks = []
    for rep in range(int(nrep)):
        p = sample_path(root, wid, rep)
        if not p.exists():
            continue
        a = np.loadtxt(p, comments="#", ndmin=2)
        if a.shape[1] >= 4:
            chunks.append(np.asarray(a[:, 1:4], float))
    return np.vstack(chunks) if chunks else np.empty((0, 3), float)


def collect_display_samples(root, state, nrep):
    chunks = []
    for w in completed_windows(state):
        wid = int(w["id"])
        s = load_window_samples(root, wid, nrep)
        if len(s) > MAX_EXPLORATION_SAMPLES_PER_WINDOW:
            idx = np.linspace(0, len(s)-1, MAX_EXPLORATION_SAMPLES_PER_WINDOW, dtype=int)
            s = s[idx]
        if len(s):
            chunks.append(s)
    if not chunks:
        return np.empty((0, 3), float)
    out = np.vstack(chunks)
    if len(out) > MAX_3D_POINTS:
        idx = np.linspace(0, len(out)-1, MAX_3D_POINTS, dtype=int)
        out = out[idx]
    return out


def measured_graph(root, state):
    ids = [int(w["id"]) for w in completed_windows(state)]
    adj = {i: set() for i in ids}
    p = Path(root) / "analysis_current" / "measured_overlap_cache.json"
    cache = read_json(p) if p.exists() else {}
    for key, rec in cache.items():
        try:
            i, j = map(int, key.split(":"))
            ov = float(rec["overlap"])
        except Exception:
            continue
        if i in adj and j in adj and ov >= MINIMUM_OVERLAP:
            adj[i].add(j); adj[j].add(i)
    return adj


def draw_graph_3d(ax, state, adj, canonical_space=False, visual_scale=1.0):
    done = completed_windows(state)
    byid = {int(w["id"]): w for w in done}
    for i in sorted(adj):
        for j in sorted(adj[i]):
            if j <= i or i not in byid or j not in byid:
                continue
            p = np.vstack([byid[i]["center"], byid[j]["center"]])
            if canonical_space:
                pp = canonical(p)
                ax.plot(pp[:,0], pp[:,1], pp[:,2], color="k", lw=0.65 * visual_scale, alpha=0.32)
            else:
                for perm in _PERMUTATIONS:
                    pp = p[:, perm]
                    ax.plot(pp[:,0], pp[:,1], pp[:,2], color="k", lw=0.50 * visual_scale, alpha=0.18)


def draw_cube(ax, lo, hi, alpha=0.35, lw=0.8, ls="--"):
    vals = [float(lo), float(hi)]
    corners = [(x, y, z) for x in vals for y in vals for z in vals]
    for i, a in enumerate(corners):
        for b in corners[i+1:]:
            if sum(abs(a[k]-b[k]) > 1.0e-12 for k in range(3)) == 1:
                ax.plot(
                    [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color="0.20", alpha=alpha, lw=lw, ls=ls,
                )


def draw_reduced_wedge(ax, lo, hi, alpha=0.55, lw=1.0):
    """Boundary tetrahedron of lo <= d(1) <= d(2) <= d(3) <= hi."""
    v = np.asarray([
        [lo, lo, lo],
        [lo, lo, hi],
        [lo, hi, hi],
        [hi, hi, hi],
    ], float)
    for i in range(4):
        for j in range(i+1, 4):
            ax.plot(
                [v[i,0], v[j,0]], [v[i,1], v[j,1]], [v[i,2], v[j,2]],
                color="0.20", alpha=alpha, lw=lw, ls="--",
            )


def figure3_exploration(root, state, cfg, outdir):
    """
    Full and permutation-reduced sampling graph.

    Physical-window markers use the same saved-sname marker/color convention
    as the other GAPS benchmark systems.  No legend is drawn; marker meanings
    are intended to be explained in the figure caption.
    """
    nrep = int(cfg.get("replicas_per_window", 10))

    samples = collect_display_samples(root, state, nrep)
    done = completed_windows(state)

    centers = (
        np.asarray([w["center"] for w in done], float)
        if done else np.empty((0, 3), float)
    )

    full_samples = symmetry_expand(samples) if len(samples) else samples
    reduced_samples = canonical(samples) if len(samples) else samples

    # Large, clean two-panel layout.
    PANEL_W = 6.90
    PANEL_H = 6.05
    COL_GAP = 0.42
    LEFT = 0.48
    RIGHT = 0.52
    BOTTOM = 0.34
    TOP_MARGIN = 0.72

    FIG_W = LEFT + 2 * PANEL_W + COL_GAP + RIGHT
    FIG_H = BOTTOM + PANEL_H + TOP_MARGIN

    # Figure 3 keeps the larger typography requested previously.
    label_fz = 26.25
    tick_fz = 20.25
    panel_fz = 30
    label_fz = 32.0
    tick_fz = 22.0
    panel_fz = 36.0
    print(
        f"Figure 3 geometry (no legend): canvas={FIG_W:.2f} in wide; "
        f"each panel={PANEL_W:.2f} x {PANEL_H:.2f} in"
    )

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    ax1 = _add_axes_inches(
        fig, FIG_W, FIG_H,
        LEFT, BOTTOM, PANEL_W, PANEL_H,
        projection="3d",
    )
    ax2 = _add_axes_inches(
        fig, FIG_W, FIG_H,
        LEFT + PANEL_W + COL_GAP, BOTTOM, PANEL_W, PANEL_H,
        projection="3d",
    )

    draw_cube(
        ax1, QUERY_MIN, QUERY_MAX,
        alpha=0.36, lw=0.82, ls="--",
    )
    draw_reduced_wedge(
        ax2, QUERY_MIN, QUERY_MAX,
        alpha=0.48, lw=0.92,
    )

    sample_color = "0.55"
    target_color = "gold"

    # --------------------------------------------------------------
    # Sample cloud
    # --------------------------------------------------------------
    if len(full_samples):
        ax1.scatter(
            full_samples[:, 0],
            full_samples[:, 1],
            full_samples[:, 2],
            s=2.8,
            alpha=0.038,
            color=sample_color,
            edgecolors="none",
            depthshade=False,
            zorder=1,
        )

        ax2.scatter(
            reduced_samples[:, 0],
            reduced_samples[:, 1],
            reduced_samples[:, 2],
            s=3.2,
            alpha=0.055,
            color=sample_color,
            edgecolors="none",
            depthshade=False,
            zorder=1,
        )

    # --------------------------------------------------------------
    # Physical nodes by saved sname
    # --------------------------------------------------------------
    observed_types = []
    for w in done:
        sname = _window_sname(w)
        if sname not in observed_types:
            observed_types.append(sname)

    preferred_order = [
        "bootstrap_seed",
        "bootstrap_bridge",
        "connectivity_bridge",
        "information_target",
    ]

    ordered_types = (
        [s for s in preferred_order if s in observed_types]
        + [s for s in observed_types if s not in preferred_order]
    )

    for sname in ordered_types:
        inds = [
            i for i, w in enumerate(done)
            if _window_sname(w) == sname
        ]
        if not inds:
            continue

        pts = centers[np.asarray(inds, dtype=int)]
        full_pts = symmetry_expand(pts)
        reduced_pts = canonical(pts)

        sty = _style_for_sname(sname)

        ax1.scatter(
            full_pts[:, 0],
            full_pts[:, 1],
            full_pts[:, 2],
            s=50,
            marker=sty["marker"],
            color=sty["color"],
            alpha=0.96,
            edgecolors="none",
            linewidths=0.0,
            depthshade=False,
            zorder=10,
        )

        ax2.scatter(
            reduced_pts[:, 0],
            reduced_pts[:, 1],
            reduced_pts[:, 2],
            s=62,
            marker=sty["marker"],
            color=sty["color"],
            alpha=0.98,
            edgecolors="none",
            linewidths=0.0,
            depthshade=False,
            zorder=10,
        )

    # --------------------------------------------------------------
    # Measured-overlap graph
    # --------------------------------------------------------------
    adj = measured_graph(root, state)

    draw_graph_3d(
        ax1, state, adj,
        canonical_space=False,
        visual_scale=1.05,
    )
    draw_graph_3d(
        ax2, state, adj,
        canonical_space=True,
        visual_scale=1.05,
    )

    # --------------------------------------------------------------
    # Current information target, if present in this historical state
    # --------------------------------------------------------------
    cur = state.get("current_query")
    if isinstance(cur, dict) and cur.get("center") is not None:
        q = np.asarray(cur["center"], float)[None, :]
        qr = canonical(q)[0]

        for perm in _PERMUTATIONS:
            qq = q[:, perm][0]
            ax1.scatter(
                [qq[0]], [qq[1]], [qq[2]],
                s=175,
                marker="*",
                color=target_color,
                edgecolors="black",
                linewidths=0.70,
                depthshade=False,
                zorder=20,
            )

        ax2.scatter(
            [qr[0]], [qr[1]], [qr[2]],
            s=205,
            marker="*",
            color=target_color,
            edgecolors="black",
            linewidths=0.75,
            depthshade=False,
            zorder=20,
        )

    # --------------------------------------------------------------
    # Axes
    # --------------------------------------------------------------
    data_lim = (QUERY_MIN, QUERY_MAX)

    z1 = _style_fullspace_axis(
        ax1,
        xlabel=r"$\mathit{d}_{\mathrm{12}}$",
        ylabel=r"$\mathit{d}_{\mathrm{13}}$",
        zlabel=r"$\mathit{d}_{\mathrm{23}}$",
        data_xlim=data_lim,
        data_ylim=data_lim,
        data_zlim=data_lim,
        show_zlabel=True,
        show_zticks=True,
        label_fz=label_fz,
        tick_fz=tick_fz,
        labelpad_scale=1.15,
        tick_step=0.50,
    )

    z2 = _style_fullspace_axis(
        ax2,
        xlabel=r"$\mathit{d}_{\mathrm{(1)}}$",
        ylabel=r"$\mathit{d}_{\mathrm{(2)}}$",
        zlabel=r"$\mathit{d}_{\mathrm{(3)}}$",
        data_xlim=data_lim,
        data_ylim=data_lim,
        data_zlim=data_lim,
        show_zlabel=True,
        show_zticks=True,
        label_fz=label_fz,
        tick_fz=tick_fz,
        labelpad_scale=1.15,
        tick_step=0.50,
    )

    # ax1.set_title("Full configuration space", fontproperties=_arial_fontprops(26.25), pad=8)
    # ax2.set_title("Permutation-reduced space", fontproperties=_arial_fontprops(26.25), pad=8)

    extra_artists = [
        artist for artist in (z1, z2)
        if artist is not None
    ]

    # --------------------------------------------------------------
    # Panel labels; no legend
    # --------------------------------------------------------------
    tag_a = ax1.text2D(
        -0.035, 0.875, "A",
        transform=ax1.transAxes,
        ha="left", va="top",
        fontproperties=_arial_fontprops(panel_fz, bold=True),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.3),
        clip_on=False,
    )

    tag_b = ax2.text2D(
        -0.035, 0.875, "B",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontproperties=_arial_fontprops(panel_fz, bold=True),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.3),
        clip_on=False,
    )

    extra_artists.extend([tag_a, tag_b])

    save_pair(
        fig, outdir,
        "Figure3_Full_vs_PermutationReduced_Graph",
        extra_artists=extra_artists,
    )
    display_figure_in_spyder(fig)



# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default=None,
        help="Explicit trimer campaign root. Default: campaign containing the newest no-penalty trimer online_analysis.npz.",
    )
    ap.add_argument(
        "--outdir", default=".",
        help="Output directory for the three publication figures.",
    )
    ap.add_argument(
        "--figure2-rows", type=int, choices=(2, 3, 4), default=FIGURE2_ROWS,
        help=(
            "Number of rows in Figure 2. Default comes from FIGURE2_ROWS at the "
            "top of the script. Choose 2, 3, or 4."
        ),
    )
    ap.add_argument(
        "--max-al-step", type=int, default=MAX_AL_STEP,
        help=(
            "Use results through this AL/query step. Default comes from "
            "MAX_AL_STEP near the top of the script; None uses latest/all."
        ),
    )
    args = ap.parse_args()

    root = resolve_root(args.root)
    state = load_state(root)
    cfg = campaign_config(root)
    apply_campaign_config(cfg)
    if abs(float(TRIANGLE_PENALTY_LAMBDA)) > 1.0e-15:
        raise RuntimeError(
            f"Selected campaign has triangle_penalty_lambda={TRIANGLE_PENALTY_LAMBDA}; "
            "this revised plotting script is intentionally restricted to the new no-penalty data."
        )
    if args.max_al_step is None:
        analysis = _latest_analysis_by_mtime(root)
    else:
        analysis = analysis_at_or_before(root, args.max_al_step)
    z = load_npz(analysis)
    plot_state = state_for_analysis(state, z)
    outdir = Path(args.outdir).expanduser().resolve()

    print("=" * 96)
    print("THREE PUBLICATION FIGURES — NEWEST NO-PENALTY TRIMER ANALYSIS")
    print("=" * 96)
    print("campaign:")
    print(" ", root)
    print("phase:", state.get("phase"))
    print("saved analysis used:")
    print(" requested AL cutoff:", "latest/all" if args.max_al_step is None else args.max_al_step)
    print(" ", analysis)
    print("analysis query:", query_number(analysis))
    from datetime import datetime
    print("analysis modified:", datetime.fromtimestamp(analysis.stat().st_mtime).isoformat(sep=" ", timespec="seconds"))
    print("runtime physics from campaign_config:")
    print("  D =", D_MIN, "..", D_MAX)
    print("  query =", QUERY_MIN, "..", QUERY_MAX)
    print("  ATM nu =", ATM_NU)
    print("  triangle penalty lambda =", TRIANGLE_PENALTY_LAMBDA, "(must be 0 for this run)")
    print("  Figure-2 colormap =", LANDSCAPE_CMAP)
    print("  Figure-2 color percentiles =", LANDSCAPE_LOW_PERCENTILE, "..", LANDSCAPE_HIGH_PERCENTILE)
    print("  Figure-2 PowerNorm gamma =", LANDSCAPE_POWER_GAMMA)
    print("  color tessellation factor =", COLOR_TESSELLATION_FACTOR, "(linear display only)")
    print("  max surface render points =", MAX_SURFACE_RENDER_POINTS, "per axis")
    print("  overlap threshold / planner =", MINIMUM_OVERLAP, PLANNER_OVERLAP)

    ncomplete = len(completed_windows(state))
    nplot = len(completed_windows(plot_state))
    nanalyzed = (
        len(np.asarray(z["connected_window_ids"]).ravel())
        if "connected_window_ids" in z else -1
    )
    print("completed physical windows in campaign state:", ncomplete)
    print("physical windows used for requested AL-step figure:", nplot)
    print("physical windows represented in saved WHAM/GP analysis:", nanalyzed)
    if args.max_al_step is not None and query_number(analysis) < int(args.max_al_step):
        print(
            f"WARNING: no exact query {args.max_al_step} snapshot was found; "
            f"using query {query_number(analysis)} instead."
        )

    print("\nMaking Figure 1 — four-panel diagnostics")
    figure1_diagnostics(root, plot_state, z, outdir, max_al_step=args.max_al_step)

    print(
        f"\nMaking Figure 2 — {args.figure2_rows} mixed slice rows x Ref/WHAM/GP"
    )
    figure2_mixed_slices(z, outdir, nrows=args.figure2_rows)

    print("\nMaking Figure 3 — full/reduced nodes + connectivity + distributions")
    figure3_exploration(root, plot_state, cfg, outdir)

    print("\nDONE")


if __name__ == "__main__":
    main()




