#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 — decomposition plot for the ACTUAL newest no-penalty trimer analysis.

This script is intentionally standalone.  It does NOT import an older trimer
plotting/controller module to decide which campaign to use.

IMPORTANT: the 20k/20k/sample-every-1/min-count-3 settings belonged to the
alanine campaign, not the trimer.  Therefore this script does NOT filter the
trimer campaign by those values.  Instead it scans trimer campaign directories
under /users/mfarshad/LJ_ADAPTIVE_US and chooses the no-penalty Cartesian/LAMMPS
campaign whose online_analysis.npz has the newest filesystem modification time.

The plot uses the no-penalty physical target

    W3 = sum_ij W2(d_ij) + W_ATM(d12,d13,d23)

and therefore the diagnostic decomposition is

    Delta W3^Ref = W_ATM
    W2,tot^Ref   = sum_ij W2(d_ij)
    W2,tot^GP    = W3^GP - Delta W3^Ref
    Delta W3^GP  = W3^GP - W2,tot^Ref

The GP component curves are inferred from the same learned total W3^GP; they are
not independently trained component GPs.  Consequently their component errors
are the same total-GP residual point-by-point.

Plotting style:
  * viridis, as in the alanine / Müller–Brown landscape figures
  * gentle PowerNorm display scaling to avoid two-band color compression
  * fixed energy-to-color mapping within each Ref/GP pair
  * shade=False so lighting does not change the energy color
  * linear display tessellation only (no spline/Gaussian smoothing)
  * right column: 1D equilateral three-body energy profiles, d12=d13=d23=d
  * 1st--99th percentile COLOR clipping only; z-heights unchanged
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, PowerNorm

# =============================================================================
# Campaign-selection controls
# =============================================================================
BASE = Path("/users/mfarshad/LJ_ADAPTIVE_US")

# Only third-system trimer campaigns are considered.  Selection is then based
# on the modification time of the newest saved online_analysis.npz.
TRIMER_NAME_TOKENS = ("TRIMER", "LJ_ATM", "LJATM")
PRINT_TOP_CANDIDATES = 12

# =============================================================================
# Figure controls
# =============================================================================
THREE_BODY_FMAX = 10.0
LANDSCAPE_CMAP = "viridis"
LANDSCAPE_LOW_PERCENTILE = 1.0
LANDSCAPE_HIGH_PERCENTILE = 99.0
LANDSCAPE_POWER_GAMMA = 0.68  # display-only: spreads viridis through the occupied energy range
COLOR_TESSELLATION_FACTOR = 12  # requested fine linear display tessellation; no landscape smoothing
MAX_SURFACE_DISPLAY_POINTS = 240  # cap display mesh only; physics/data are unchanged
PARITY_MAX_POINTS = 12000
PNG_DPI = 450
SHOW_FIGURES_IN_SPYDER = True

FONT_SCALE = 1.20
BASE_FONT_FZ = 25.0 * FONT_SCALE
LABEL_FZ = 36.0 * FONT_SCALE
TITLE_FZ = 36.0 * FONT_SCALE
TICK_FZ = 28.0 * FONT_SCALE
LEGEND_FZ = 32.0 * FONT_SCALE
PANEL_FZ = 36.0 * FONT_SCALE
SURFACE_LABEL_FZ = 26.25 * FONT_SCALE
SURFACE_TICK_FZ = 20.25 * FONT_SCALE
SURFACE_ZLABEL_FZ = SURFACE_LABEL_FZ
SMALL_LABEL_FZ = SURFACE_LABEL_FZ
SMALL_TICK_FZ = SURFACE_TICK_FZ
SMALL_LEGEND_FZ = 16.5 * FONT_SCALE

C_GP = "#3B82F6"
C_REF = "#111111"
C_DELTA = "#B2182B"
C_1D_REF = "#2FB47C"
C_1D_GP = "#2FB47C"

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": BASE_FONT_FZ,
    "axes.labelsize": LABEL_FZ,
    "axes.titlesize": TITLE_FZ,
    "xtick.labelsize": TICK_FZ,
    "ytick.labelsize": TICK_FZ,
    "legend.fontsize": LEGEND_FZ,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.cal": "Arial",
    "mathtext.tt": "Arial",
})


# =============================================================================
# Campaign / analysis selection
# =============================================================================
def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _query_number(path: Path):
    for part in path.parts[::-1]:
        m = re.search(r"query[_-]?(\d+)", part, flags=re.I)
        if m:
            return int(m.group(1))
    return -1


def _analysis_files(root: Path):
    fs = list((root / "analysis_history").rglob("online_analysis.npz"))
    if not fs:
        fs = list(root.rglob("online_analysis.npz"))
    return sorted(fs, key=lambda p: (_query_number(p), p.stat().st_mtime))


def _latest_analysis(root: Path, max_al_step=None):
    fs = _analysis_files(root)
    if max_al_step is not None:
        fs = [p for p in fs if _query_number(p) <= int(max_al_step)]
    if not fs:
        raise FileNotFoundError(f"No matching online_analysis.npz found under {root}")
    return max(fs, key=lambda p: (_query_number(p), p.stat().st_mtime))


def _config_values(cfg):
    wham = cfg.get("wham", {}) if isinstance(cfg.get("wham", {}), dict) else {}
    cal = cfg.get("calibration", {}) if isinstance(cfg.get("calibration", {}), dict) else {}
    return {
        "eq": cfg.get("equilibration_steps"),
        "prod": cfg.get("production_steps"),
        "sample": cfg.get("sample_every"),
        "min_count": wham.get("min_total_counts", cfg.get("min_total_counts")),
        "cal_eq": cal.get("equilibration_steps"),
        "cal_prod": cal.get("production_steps"),
        "cal_sample": cal.get("sample_every"),
    }


def _is_no_penalty_trimer(root: Path, cfg: dict):
    """True for the new physical trimer family: LAMMPS/Cartesian and no penalty."""
    name = root.name.upper()
    if not any(tok in name for tok in TRIMER_NAME_TOKENS):
        return False

    # Reject any campaign that explicitly carries a nonzero triangle penalty.
    penalty_lambda = cfg.get("triangle_penalty_lambda", cfg.get("TRIANGLE_PENALTY_LAMBDA", 0.0))
    try:
        if abs(float(penalty_lambda or 0.0)) > 1.0e-15:
            return False
    except Exception:
        pass

    # Strong positive evidence from campaign name/config for the new physical run.
    txt = " ".join([
        name,
        str(cfg.get("dynamics", "")),
        str(cfg.get("atm_implementation", "")),
        str(cfg.get("atm_force_implementation", "")),
        str(cfg.get("potential", "")),
    ]).upper()
    no_penalty_evidence = (
        "NOPENALTY" in txt or "NOPENTY" in txt or "NO PENALTY" in txt
        or "BUILT-IN PAIR_STYLE ATM" in txt
        or "PAIR_STYLE ATM" in txt
        or "LAMMPS" in txt
    )
    return bool(no_penalty_evidence)


def discover_latest_no_penalty_trimer(base=BASE, max_al_step=None):
    """Select by newest analysis file mtime across all no-penalty trimer campaigns."""
    if not base.exists():
        raise FileNotFoundError(base)

    candidates = []
    seen_roots = set()

    # campaign_config.json is the authoritative campaign marker.
    for cfg_path in base.rglob("campaign_config.json"):
        root = cfg_path.parent.resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        try:
            cfg = _read_json(cfg_path)
        except Exception:
            continue
        if not _is_no_penalty_trimer(root, cfg):
            continue
        try:
            analysis = _latest_analysis(root, max_al_step=max_al_step)
        except Exception:
            continue
        candidates.append((analysis.stat().st_mtime, root, cfg, analysis.resolve()))

    if not candidates:
        raise RuntimeError(
            "No no-penalty trimer campaign with online_analysis.npz was found under "
            f"{base}. Pass the exact campaign with --root /path/to/campaign."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)

    print("\nNewest no-penalty trimer analysis candidates:")
    print("-" * 96)
    from datetime import datetime
    for rank, (mtime, root, cfg, analysis) in enumerate(candidates[:PRINT_TOP_CANDIDATES], 1):
        v = _config_values(cfg)
        stamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{rank:2d}. {stamp}  {root.name}")
        print(f"    analysis: {analysis}")
        print(f"    EQ={v['eq']} PROD={v['prod']} SAMPLE={v['sample']} MIN_COUNT={v['min_count']}")
    print("-" * 96)

    _, root, cfg, analysis = candidates[0]
    return root, cfg, analysis


def select_campaign(user_root=None, max_al_step=None):
    if user_root:
        root = Path(user_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        cfg_path = root / "campaign_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        cfg = _read_json(cfg_path)
        analysis = _latest_analysis(root, max_al_step=max_al_step)
        return root, cfg, analysis

    return discover_latest_no_penalty_trimer(BASE, max_al_step=max_al_step)


# =============================================================================
# Physics reconstructed directly from campaign_config.json
# =============================================================================
class Physics:
    def __init__(self, cfg):
        domain = cfg.get("distance_domain", [0.95, 2.50])
        self.dmin = float(domain[0])
        self.dmax = float(domain[1])
        self.eps = float(cfg.get("lj_epsilon", 1.0))
        self.sigma = float(cfg.get("lj_sigma", 1.0))
        self.rc = float(cfg.get("lj_cutoff", 2.5))
        self.atm_nu = float(cfg.get("atm_nu", 4.0))
        self.contact_shift = float(cfg.get("contact_shift", cfg.get("CONTACT_SHIFT", 0.0)))

    def shifted_lj(self, r):
        """LJ evaluated in the reported/GAPS d coordinate, matching the campaign."""
        r = np.asarray(r, float)
        rc = self.rc
        rc_energy = 4.0 * self.eps * ((self.sigma / rc) ** 12 - (self.sigma / rc) ** 6)
        out = np.zeros_like(r, dtype=float)
        mask = np.isfinite(r) & (r < rc) & (r > 0.0)
        inv = self.sigma / r[mask]
        out[mask] = 4.0 * self.eps * (inv ** 12 - inv ** 6) - rc_energy
        return out

    def pair_energy(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return np.sum(self.shifted_lj(p), axis=1)

    def atm_energy(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        a, b, c = p[:, 0], p[:, 1], p[:, 2]
        eps = 1.0e-15
        cos1 = (a*a + b*b - c*c) / np.maximum(2.0*a*b, eps)
        cos2 = (a*a + c*c - b*b) / np.maximum(2.0*a*c, eps)
        cos3 = (b*b + c*c - a*a) / np.maximum(2.0*b*c, eps)
        denom = np.maximum((a*b*c) ** 3, 1.0e-300)
        return self.atm_nu * (1.0 + 3.0*cos1*cos2*cos3) / denom

    def exact_total(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return self.pair_energy(p) + self.atm_energy(p)

    def reported_cube_mask(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return (
            np.all(np.isfinite(p), axis=1)
            & np.all(p >= self.dmin - 1.0e-10, axis=1)
            & np.all(p <= self.dmax + 1.0e-10, axis=1)
        )


# =============================================================================
# Saved-analysis helpers
# =============================================================================
def require_keys(z, keys):
    missing = [k for k in keys if k not in z]
    if missing:
        raise KeyError(
            "The selected analysis does not contain the arrays required by Figure 7: "
            + ", ".join(missing)
        )


def load_npz(path):
    return dict(np.load(path, allow_pickle=False))


def exact_shift_from(z):
    if "exact_reference_shift" not in z:
        return 0.0
    return float(np.asarray(z["exact_reference_shift"]).ravel()[0])


def build_saved_isosceles_slice(z, phys):
    """Use the actual saved slice from THIS analysis; do not reconstruct an old run."""
    require_keys(z, ["slice_gp_mean"])

    if "slice_common_axis" in z:
        x = np.asarray(z["slice_common_axis"], float)
    elif "slice_axis" in z:
        x = np.asarray(z["slice_axis"], float)
    else:
        raise KeyError("Selected analysis has no slice_common_axis/slice_axis")

    if "slice_d12_axis" in z:
        y = np.asarray(z["slice_d12_axis"], float)
    else:
        y = x.copy()

    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([Y.ravel(), X.ravel(), X.ravel()])

    gp_total_aligned = np.asarray(z["slice_gp_mean"], float)
    if gp_total_aligned.shape != X.shape:
        gp_total_aligned = gp_total_aligned.reshape(X.shape)

    shift = exact_shift_from(z)
    gp_total = gp_total_aligned - shift

    # Reference decomposition: W3^Ref = W2,tot^Ref + DeltaW3^Ref.
    pair_ref = phys.pair_energy(pts).reshape(X.shape)
    delta_ref = phys.atm_energy(pts).reshape(X.shape)

    # Inferred GP decomposition from the SAME learned total W3^GP:
    #   W2,tot^GP  = W3^GP - DeltaW3^Ref
    #   DeltaW3^GP = W3^GP - W2,tot^Ref
    # These are complementary diagnostic projections of the total-GP error;
    # neither component is an independently trained component GP here.
    pair_gp = gp_total - delta_ref
    delta_gp = gp_total - pair_ref

    return X, Y, pair_ref, delta_ref, pair_gp, delta_gp


def expand_canonical_grid(z, phys):
    """Expand symmetry-reduced saved GP grid to the labeled full reported cube."""
    require_keys(z, ["gp_base_points", "gp_base_mean"])
    bp = np.asarray(z["gp_base_points"], float)
    bm = np.asarray(z["gp_base_mean"], float).ravel()
    if len(bp) != len(bm):
        raise ValueError("gp_base_points and gp_base_mean have inconsistent lengths")

    keep = phys.reported_cube_mask(bp) & np.isfinite(bm)
    bp = bp[keep]
    bm = bm[keep]

    all_pts = []
    all_gp = []
    for p, g in zip(bp, bm):
        perms = np.asarray(list(itertools.permutations(p.tolist(), 3)), float)
        # Remove duplicate permutations for coordinates such as (a,a,b).
        perms = np.unique(np.round(perms, 12), axis=0)
        all_pts.append(perms)
        all_gp.append(np.full(len(perms), float(g)))

    if not all_pts:
        raise RuntimeError("No saved GP base points lie inside the reported domain")

    return np.vstack(all_pts), np.concatenate(all_gp)


def full_space_quantities(z, phys):
    pts, gp_total_aligned = expand_canonical_grid(z, phys)
    shift = exact_shift_from(z)

    pair_ref = phys.pair_energy(pts)
    delta_ref = phys.atm_energy(pts)
    exact_total = pair_ref + delta_ref
    gp_total = gp_total_aligned - shift

    # Same inferred decomposition used in the plotted slice.
    pair_gp = gp_total - delta_ref
    delta_gp = gp_total - pair_ref

    mask = (
        phys.reported_cube_mask(pts)
        & np.isfinite(pair_ref)
        & np.isfinite(pair_gp)
        & np.isfinite(delta_ref)
        & np.isfinite(delta_gp)
        & np.isfinite(exact_total)
        & np.isfinite(gp_total)
    )

    # Same physical-energy diagnostic window as the previous Figure 7.
    rel = np.full_like(exact_total, np.nan)
    if np.any(mask):
        emin = float(np.nanmin(exact_total[mask]))
        rel[mask] = exact_total[mask] - emin
        mask &= np.isfinite(rel) & (rel <= THREE_BODY_FMAX)

    if not np.any(mask):
        raise RuntimeError("No full-space points survived the Figure-7 diagnostic mask")

    return {
        "pts": pts,
        "pair_ref": pair_ref,
        "pair_gp": pair_gp,
        "delta_ref": delta_ref,
        "delta_gp": delta_gp,
        "mask": mask,
    }


# =============================================================================
# Display-only linear tessellation: smoother COLOR changes, no data smoothing
# =============================================================================
def refine_surface_for_display(X, Y, Z, factor=COLOR_TESSELLATION_FACTOR):
    if factor <= 1:
        return X, Y, Z

    x = np.asarray(X[0, :], float)
    y = np.asarray(Y[:, 0], float)
    z = np.asarray(Z, float)

    nx = min((len(x) - 1) * factor + 1, MAX_SURFACE_DISPLAY_POINTS)
    ny = min((len(y) - 1) * factor + 1, MAX_SURFACE_DISPLAY_POINTS)
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


# =============================================================================
# Plot styling
# =============================================================================
def panel(ax, letter):
    if hasattr(ax, "text2D"):
        return ax.text2D(
            -0.025, 0.92, letter, transform=ax.transAxes,
            ha="left", va="top", clip_on=False,
            fontsize=PANEL_FZ, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.3),
        )
    return ax.text(
        0.018, 0.982, letter, transform=ax.transAxes,
        ha="left", va="top", fontsize=PANEL_FZ, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.3),
        zorder=100,
    )


def add_axes_inches(fig, fig_w, fig_h, x_in, y_in, w_in, h_in, projection=None):
    return fig.add_axes(
        [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h],
        projection=projection,
    )


def set_surface_axes(ax, xlabel, ylabel, zlim, zlabel):
    """Publication-style 3D axes matching the cleaner original Figure 7.

    Keep the large 3D panels, but reduce visual crowding by using an orthographic
    projection, clean 0.5-spaced distance ticks, fewer z ticks, and a z label
    pinned in 2D axes coordinates rather than attached to the rotating 3D axis.
    """
    ax.set_xlabel(xlabel, labelpad=20.0, fontsize=SURFACE_LABEL_FZ)
    ax.set_ylabel(ylabel, labelpad=20.0, fontsize=SURFACE_LABEL_FZ)
    ax.set_zlabel("")
    ax.set_zlim(*zlim)

    # Clean coordinate ticks, as in the older successful Figure 7.
    def _distance_ticks(lo, hi, step=0.5):
        start = np.ceil((float(lo) - 1.0e-12) / step) * step
        vals = np.arange(start, float(hi) + 0.5 * step, step)
        return vals[(vals >= float(lo) - 1.0e-10) & (vals <= float(hi) + 1.0e-10)]

    xticks = _distance_ticks(*ax.get_xlim(), step=0.5)
    yticks = _distance_ticks(*ax.get_ylim(), step=0.5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # Keep every coordinate tick label visible, including the lowest value (1).
    ax.set_xticklabels([f"{v:g}" for v in xticks])
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))

    ax.tick_params(axis="x", labelsize=SURFACE_TICK_FZ, pad=0)
    ax.tick_params(axis="y", labelsize=SURFACE_TICK_FZ, pad=0)
    ax.tick_params(axis="z", labelsize=SURFACE_TICK_FZ, pad=4)

    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=28, azim=-140)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.70), zoom=1.06)
    except TypeError:
        ax.set_box_aspect((1.0, 1.0, 0.70))
    except Exception:
        pass

    # Match the light panes/grid of the earlier successful Figure 7.
    ax.set_facecolor("white")
    for _axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            _axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            _axis.pane.set_edgecolor((0.80, 0.80, 0.80, 1.0))
            _axis._axinfo["grid"]["color"] = (0.80, 0.80, 0.80, 0.48)
            _axis._axinfo["grid"]["linewidth"] = 0.55
        except Exception:
            pass

    # Pin the z label so it never collides with the 3D spine/tick labels.
    ztxt = ax.text2D(
        -0.0525, 0.50, zlabel,
        transform=ax.transAxes,
        rotation=90,
        va="center", ha="right",
        fontsize=SURFACE_ZLABEL_FZ,
        clip_on=False,
    )
    return ztxt


def show_or_close(fig):
    if SHOW_FIGURES_IN_SPYDER:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        plt.show(block=False)
    else:
        plt.close(fig)


# =============================================================================
# Main Figure 7
# =============================================================================

def sample_diagonal_from_slice(X, Y, Z, dvals):
    """Bilinear interpolation of Z(d12, d13=d23) along the equilateral line d12=d13=d23=d."""
    x = np.asarray(X[0, :], float)
    y = np.asarray(Y[:, 0], float)
    z = np.asarray(Z, float)
    out = np.full(len(dvals), np.nan, float)

    for i, d in enumerate(np.asarray(dvals, float)):
        if d < x[0] or d > x[-1] or d < y[0] or d > y[-1]:
            continue
        # Interpolate each row at x=d.
        row_vals = np.full(len(y), np.nan, float)
        for j in range(len(y)):
            good = np.isfinite(z[j, :])
            if np.count_nonzero(good) >= 2:
                row_vals[j] = np.interp(d, x[good], z[j, good])
        good = np.isfinite(row_vals)
        if np.count_nonzero(good) >= 2:
            out[i] = np.interp(d, y[good], row_vals[good])
    return out



def continuous_surface_facecolors(Z, vmin, vmax, cmap_name=LANDSCAPE_CMAP):
    """Continuous display colors from cell-centered Z values; no landscape smoothing."""
    z = np.asarray(Z, float)
    # One RGBA color per surface quadrilateral, using the mean of its four vertices.
    zcell = 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])
    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    cmap = plt.get_cmap(cmap_name)
    fc = cmap(norm(zcell))
    bad = ~np.isfinite(zcell)
    if np.any(bad):
        fc[bad, 3] = 0.0
    return fc


def make_figure(z, phys, outdir):
    X, Y, pair_ref, delta_ref, pair_gp, delta_gp = build_saved_isosceles_slice(z, phys)

    # Display-only linear tessellation for smooth color variation.
    Xp, Yp, pair_ref_plot = refine_surface_for_display(X, Y, pair_ref)
    _, _, pair_gp_plot = refine_surface_for_display(X, Y, pair_gp)
    Xd, Yd, delta_ref_plot = refine_surface_for_display(X, Y, delta_ref)
    _, _, delta_gp_plot = refine_surface_for_display(X, Y, delta_gp)
    print(f"Rendered 3D display mesh: {pair_ref_plot.shape[0]} x {pair_ref_plot.shape[1]} points")

    # Shared z/color ranges for Ref/GP pairs.
    pair_union = np.concatenate([
        pair_ref[np.isfinite(pair_ref)], pair_gp[np.isfinite(pair_gp)]
    ])
    delta_union = np.concatenate([
        delta_ref[np.isfinite(delta_ref)], delta_gp[np.isfinite(delta_gp)]
    ])

    pair_zmin = float(np.nanmin(pair_union))
    pair_zmax = float(np.nanmax(pair_union))
    pair_pad = 0.04 * max(pair_zmax - pair_zmin, 1e-6)
    pair_zlim = (pair_zmin - pair_pad, pair_zmax + pair_pad)
    # Display colors only: trim both extreme tails and use a gentle monotonic
    # power normalization.  This prevents steep LJ/ATM regions from collapsing
    # most of viridis into two visually dominant bands.  Surface heights/data
    # are unchanged, and each Ref/GP pair still shares exactly one normalization.
    pair_cmin = float(np.nanpercentile(pair_union, LANDSCAPE_LOW_PERCENTILE))
    pair_cmax = float(np.nanpercentile(pair_union, LANDSCAPE_HIGH_PERCENTILE))
    if not np.isfinite(pair_cmin):
        pair_cmin = float(np.nanmin(pair_union))
    if not np.isfinite(pair_cmax) or pair_cmax <= pair_cmin + 1.0e-12:
        pair_cmax = float(np.nanmax(pair_union))

    delta_zmin = float(np.nanmin(delta_union))
    delta_zmax = float(np.nanmax(delta_union))
    delta_pad = 0.04 * max(delta_zmax - delta_zmin, 1e-6)
    delta_zlim = (delta_zmin - delta_pad, delta_zmax + delta_pad)
    delta_cmin = float(np.nanpercentile(delta_union, LANDSCAPE_LOW_PERCENTILE))
    delta_cmax = float(np.nanpercentile(delta_union, LANDSCAPE_HIGH_PERCENTILE))
    if not np.isfinite(delta_cmin):
        delta_cmin = float(np.nanmin(delta_union))
    if not np.isfinite(delta_cmax) or delta_cmax <= delta_cmin + 1.0e-12:
        delta_cmax = float(np.nanmax(delta_union))

    pair_norm = PowerNorm(gamma=LANDSCAPE_POWER_GAMMA, vmin=pair_cmin, vmax=pair_cmax, clip=True)
    delta_norm = PowerNorm(gamma=LANDSCAPE_POWER_GAMMA, vmin=delta_cmin, vmax=delta_cmax, clip=True)

    print(
        f"Landscape color scales: W2 [{pair_cmin:.4g}, {pair_cmax:.4g}] kBT; "
        f"DeltaW3 [{delta_cmin:.4g}, {delta_cmax:.4g}] kBT; "
        f"percentiles={LANDSCAPE_LOW_PERCENTILE:g}-{LANDSCAPE_HIGH_PERCENTILE:g}, "
        f"PowerNorm gamma={LANDSCAPE_POWER_GAMMA:g}"
    )

    # Full-space parity/error data from THIS analysis snapshot.
    qall = full_space_quantities(z, phys)
    m = qall["mask"]
    x_pair = qall["pair_ref"][m]
    y_pair = qall["pair_gp"][m]
    x_delta = qall["delta_ref"][m]
    y_delta = qall["delta_gp"][m]

    rmse_pair = float(np.sqrt(np.mean((y_pair - x_pair) ** 2)))
    rmse_delta = float(np.sqrt(np.mean((y_delta - x_delta) ** 2)))

    # Rendered scatter only may be downsampled; RMSE uses all points above.
    def downsample(x, y):
        if len(x) <= PARITY_MAX_POINTS:
            return x, y
        ind = np.linspace(0, len(x) - 1, PARITY_MAX_POINTS).astype(int)
        return x[ind], y[ind]

    x_pair_show, y_pair_show = downsample(x_pair, y_pair)
    x_delta_show, y_delta_show = downsample(x_delta, y_delta)

    parity_vals = np.concatenate([x_pair, y_pair, x_delta, y_delta])
    parity_lo = float(np.nanmin(parity_vals))
    parity_hi = float(np.nanmax(parity_vals))
    ppad = 0.04 * max(parity_hi - parity_lo, 1e-6)
    parity_lo -= ppad
    parity_hi += ppad

    err_pair = y_pair - x_pair
    err_delta = y_delta - x_delta

    # Algebraically, both inferred component errors equal the same total-GP
    # residual: (W3^GP - W3^Ref).  Verify this rather than showing a fake
    # zero-width W2 distribution.
    max_component_error_mismatch = float(np.nanmax(np.abs(err_pair - err_delta)))
    print(f"max |W2 error - DeltaW3 error| = {max_component_error_mismatch:.3e} kBT")

    err_shared = 0.5 * (err_pair + err_delta)
    err_lim = float(np.nanpercentile(np.abs(err_shared), 99.0))
    err_lim = max(err_lim, 0.05)
    bins = np.linspace(-err_lim, err_lim, 61)
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist_shared, _ = np.histogram(err_shared, bins=bins, density=True)

    # 1D pair curves.  The reference is the analytic LJ pair.  For the GP row,
    # infer a one-pair curve on the equilateral line: W2,tot^GP(d,d,d)/3.
    pair_1d_min = max(0.93, phys.dmin - 0.02)
    d1 = np.linspace(pair_1d_min, phys.dmax, 500)
    w2_ref_1d = phys.shifted_lj(d1)
    w2_gp_1d = sample_diagonal_from_slice(X, Y, pair_gp, d1) / 3.0

    # 1D three-body ENERGY profiles along the equilateral line
    # d12 = d13 = d23 = d.  These are direct 1D cross-sections of panels C/G,
    # so the plotted quantity and units are exactly the same as the 3D z axes.
    d3 = np.linspace(phys.dmin, phys.dmax, 500)
    equilateral_pts = np.column_stack([d3, d3, d3])
    delta_ref_1d = phys.atm_energy(equilateral_pts)
    delta_gp_1d = sample_diagonal_from_slice(X, Y, delta_gp, d3)

    # Geometry: original Figure-7 style, but with an added first column.
    LARGE_W = 7.25
    LARGE_H = 5.65
    SMALL = 3.85
    GAP12 = 1.05
    GAP23 = 0.70
    GAP34 = 1.40
    ROW_GAP = 0.55
    LEFT = 1.35
    RIGHT = 2.75
    BOTTOM = 1.10
    TOP_MARGIN = 0.70

    FIG_W = LEFT + SMALL + GAP12 + LARGE_W + GAP23 + LARGE_W + GAP34 + SMALL + RIGHT
    FIG_H = BOTTOM + LARGE_H + ROW_GAP + LARGE_H + TOP_MARGIN
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    x1 = LEFT
    x2 = x1 + SMALL + GAP12
    x3 = x2 + LARGE_W + GAP23
    x4 = x3 + LARGE_W + GAP34
    y_bot = BOTTOM
    y_top = y_bot + LARGE_H + ROW_GAP
    y_small_top = y_top + 0.5 * (LARGE_H - SMALL)
    y_small_bot = y_bot + 0.5 * (LARGE_H - SMALL)

    xlim = (float(np.nanmin(X)), float(np.nanmax(X)))
    ylim = (float(np.nanmin(Y)), float(np.nanmax(Y)))
    ypad = 0.11 * (ylim[1] - ylim[0])

    xlabel = r"$d_{\mathrm{13}}=d_{\mathrm{23}}$"
    ylabel = r"$d_{\mathrm{12}}$"

    def finish_surface(ax, zlim, zlabel, letter):
        ax.set_xlim(*xlim)
        ax.set_ylim(ylim[0], ylim[1] + ypad)
        set_surface_axes(ax, xlabel, ylabel, zlim, zlabel)
        ax.set_title("")
        panel(ax, letter)

    def finish_line(ax, xlabel, ylabel, letter):
        ax.set_xlabel(xlabel, labelpad=4, fontsize=SMALL_LABEL_FZ)
        ax.set_ylabel(ylabel, labelpad=4, fontsize=SMALL_LABEL_FZ)
        ax.tick_params(labelsize=SMALL_TICK_FZ, pad=1)
        panel(ax, letter)

    print("Building 2D/3D panels...")

    # A — 1D reference single-pair W2(d12)
    axA = add_axes_inches(fig, FIG_W, FIG_H, x1, y_small_top, SMALL, SMALL)
    axA.set_box_aspect(1)
    axA.plot(d1, w2_ref_1d, lw=2.6, color=C_1D_REF)
    axA.set_xlim(0.80, phys.dmax)
    finish_line(axA, r"$d_{\mathrm{12}}$", r"$W_2^{\mathrm{Ref}}(d_{12})\ (k_{\mathrm{B}}T)$", "A")

    # B — reference total pairwise contribution
    axB = add_axes_inches(fig, FIG_W, FIG_H, x2, y_top, LARGE_W, LARGE_H, projection="3d")
    axB.plot_surface(
        Xp, Yp, pair_ref_plot,
        cmap=LANDSCAPE_CMAP, norm=pair_norm,
        linewidth=0, edgecolor="none", antialiased=True, shade=False,
        rcount=pair_ref_plot.shape[0],
        ccount=pair_ref_plot.shape[1], rasterized=True,
    )
    finish_surface(axB, pair_zlim, r"$W_{2,\mathrm{tot}}^{\mathrm{Ref}}\ (k_{\mathrm{B}}T)$", "B")

    # C — reference Delta W3
    axC = add_axes_inches(fig, FIG_W, FIG_H, x3, y_top, LARGE_W, LARGE_H, projection="3d")
    axC.plot_surface(
        Xd, Yd, delta_ref_plot,
        cmap=LANDSCAPE_CMAP, norm=delta_norm,
        linewidth=0, edgecolor="none", antialiased=True, shade=False,
        rcount=delta_ref_plot.shape[0],
        ccount=delta_ref_plot.shape[1], rasterized=True,
    )
    finish_surface(axC, delta_zlim, r"$\Delta W_3^{\mathrm{Ref}}\ (k_{\mathrm{B}}T)$", "C")

    # D — 1D reference Delta W3 cross-section along d12=d13=d23=d
    axD = add_axes_inches(fig, FIG_W, FIG_H, x4, y_small_top, SMALL, SMALL)
    axD.set_box_aspect(1)
    axD.plot(d3, delta_ref_1d, lw=2.6, color=C_DELTA)
    axD.set_xlim(0.80, phys.dmax)
    axD.set_ylim(-1.0, delta_zlim[1])
    finish_line(
        axD,
        r"$d_{\mathrm{12}}=d_{\mathrm{13}}=d_{\mathrm{23}}$",
        r"$\Delta W_3^{\mathrm{Ref}}\ (k_{\mathrm{B}}T)$",
        "D",
    )

    # E — 1D GP single-pair W2(d12)
    axE = add_axes_inches(fig, FIG_W, FIG_H, x1, y_small_bot, SMALL, SMALL)
    axE.set_box_aspect(1)
    axE.plot(d1, w2_gp_1d, lw=2.6, color=C_1D_GP)
    axE.set_xlim(0.80, phys.dmax)
    finish_line(axE, r"$d_{\mathrm{12}}$", r"$W_2^{\mathrm{GP}}(d)\ (k_{\mathrm{B}}T)$", "E")

    # F — GP total pairwise contribution
    axF = add_axes_inches(fig, FIG_W, FIG_H, x2, y_bot, LARGE_W, LARGE_H, projection="3d")
    axF.plot_surface(
        Xp, Yp, pair_gp_plot,
        cmap=LANDSCAPE_CMAP, norm=pair_norm,
        linewidth=0, edgecolor="none", antialiased=True, shade=False,
        rcount=pair_gp_plot.shape[0],
        ccount=pair_gp_plot.shape[1], rasterized=True,
    )
    finish_surface(axF, pair_zlim, r"$W_{2,\mathrm{tot}}^{\mathrm{GP}}\ (k_{\mathrm{B}}T)$", "F")

    # G — GP Delta W3
    axG = add_axes_inches(fig, FIG_W, FIG_H, x3, y_bot, LARGE_W, LARGE_H, projection="3d")
    axG.plot_surface(
        Xd, Yd, delta_gp_plot,
        cmap=LANDSCAPE_CMAP, norm=delta_norm,
        linewidth=0, edgecolor="none", antialiased=True, shade=False,
        rcount=delta_gp_plot.shape[0],
        ccount=delta_gp_plot.shape[1], rasterized=True,
    )
    finish_surface(axG, delta_zlim, r"$\Delta W_3^{\mathrm{GP}}\ (k_{\mathrm{B}}T)$", "G")

    # H — 1D GP Delta W3 cross-section along d12=d13=d23=d
    axH = add_axes_inches(fig, FIG_W, FIG_H, x4, y_small_bot, SMALL, SMALL)
    axH.set_box_aspect(1)
    axH.plot(d3, delta_gp_1d, lw=2.6, color=C_DELTA)
    axH.set_xlim(0.80, phys.dmax)
    axH.set_ylim(-1.0, delta_zlim[1])
    finish_line(
        axH,
        r"$d_{\mathrm{12}}=d_{\mathrm{13}}=d_{\mathrm{23}}$",
        r"$\Delta W_3^{\mathrm{GP}}\ (k_{\mathrm{B}}T)$",
        "H",
    )

    # Final rendered-content centering with explicit edge clearance.
    # Manual nominal gaps above are tuned for the mixed 2D/3D panel geometry;
    # here we only shift the complete composition as one rigid block.
    all_axes = [axA, axB, axC, axD, axE, axF, axG, axH]
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbs = [ax.get_tightbbox(renderer) for ax in all_axes]
        left_px = min(bb.x0 for bb in bbs)
        right_px = max(bb.x1 for bb in bbs)
        fig_w_px = fig.bbox.width
        clearance_px = 0.45 * fig.dpi

        # Center first.
        content_center = 0.5 * (left_px + right_px)
        frame_center = 0.5 * fig_w_px
        shift_px = frame_center - content_center

        # Then enforce minimum left/right clearance.
        new_left = left_px + shift_px
        new_right = right_px + shift_px
        if new_left < clearance_px:
            shift_px += clearance_px - new_left
        if new_right > fig_w_px - clearance_px:
            shift_px -= new_right - (fig_w_px - clearance_px)

        if abs(shift_px) < 0.5:
            break
        shift_frac = shift_px / fig_w_px
        for ax in all_axes:
            pos = ax.get_position()
            ax.set_position([pos.x0 + shift_frac, pos.y0, pos.width, pos.height])

    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "Figure_7_Trimer_ACTUAL_LATEST_no_penalty_2x4"
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    print("Saving PNG...")
    fig.savefig(png, dpi=PNG_DPI, bbox_inches=None, facecolor="white")
    print("Saving PDF...")
    fig.savefig(pdf, bbox_inches=None, facecolor="white")
    print("Save complete.")
    print("saved:", png)
    print("saved:", pdf)
    print(f"full-space inferred W2 RMSE    = {rmse_pair:.6f} kBT")
    print(f"full-space inferred DeltaW3 RMSE = {rmse_delta:.6f} kBT")
    print("NOTE: the two component RMSEs/errors are identical by construction because")
    print("      both are complementary decompositions of the same total W3^GP residual.")

    show_or_close(fig)


# =============================================================================
# Entry point
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="Explicit trimer campaign root. Omit to auto-select the trimer campaign with the newest no-penalty online_analysis.npz.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory. Default: <selected campaign>/diagnostics")
    ap.add_argument("--max-al-step", type=int, default=None,
                    help="Optional maximum saved query/AL step; default uses newest analysis in selected campaign.")
    args = ap.parse_args()

    root, cfg, analysis = select_campaign(args.root, max_al_step=args.max_al_step)
    values = _config_values(cfg)
    phys = Physics(cfg)
    z = load_npz(analysis)

    print("=" * 96)
    print("FIGURE 7 — ACTUAL NEWEST TRIMER ANALYSIS SELECTION")
    print("=" * 96)
    print("SELECTED CAMPAIGN:")
    print(" ", root)
    print("SELECTED ANALYSIS:")
    print(" ", analysis)
    print("analysis query:", _query_number(analysis))
    from datetime import datetime
    print("analysis modified:", datetime.fromtimestamp(analysis.stat().st_mtime).isoformat(sep=" ", timespec="seconds"))
    print("\nCONFIGURATION READ FROM THIS CAMPAIGN:")
    print("  equilibration_steps =", values["eq"])
    print("  production_steps    =", values["prod"])
    print("  sample_every        =", values["sample"])
    print("  min_total_counts    =", values["min_count"])
    print("  calibration eq/prod/sample =",
          values["cal_eq"], values["cal_prod"], values["cal_sample"])
    print("  distance domain     =", phys.dmin, "..", phys.dmax)
    print("  ATM nu              =", phys.atm_nu)
    print("  contact_shift       =", phys.contact_shift)
    print("  target              = pair LJ + ATM only; NO penalty")
    print("  colormap            =", LANDSCAPE_CMAP)
    print("  color percentiles   =", LANDSCAPE_LOW_PERCENTILE, "..", LANDSCAPE_HIGH_PERCENTILE)
    print("  color PowerNorm gamma =", LANDSCAPE_POWER_GAMMA)
    print("  color tessellation  =", COLOR_TESSELLATION_FACTOR, "(linear display only)")
    print("=" * 96)

    outdir = args.outdir if args.outdir is not None else str(root / "diagnostics")
    make_figure(z, phys, outdir)


if __name__ == "__main__":
    main()




