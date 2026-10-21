#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-style Figure 4: 6-panel alanine-dipeptide GAPS-US figure.

Panel order
-----------
A  Reference FES + final adaptive sampling network.
B  Final direct periodic WHAM FES.
C  Fixed-domain convergence:
     GP RMSE on one fixed F_ref validation domain + WHAM coverage.
D  Reference FES.
E  Final periodic GP FES.
F  Horizontal, vertical, and wrapped-diagonal 1D profiles.

Previous order note removed.


The molecular schematic from the previous draft is intentionally removed from
the main-text figure.  Panels E and F are retained from the previous version.

The default root is hard-locked to the audited completed 2026-09-19 R4
campaign. Pass --root to override it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
if not os.environ.get("DISPLAY") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse
from scipy.interpolate import RegularGridInterpolator


# Show generated figures in Spyder's Plots pane / active Matplotlib backend.
# Set to False for batch/HPC runs where you only want saved PNG/PDF files.
SHOW_FIGURES_IN_SPYDER = True

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
# USER / PROJECT SETTINGS
# =============================================================================

BASE = Path("/users/mfarshad/LJ_ADAPTIVE_US")

FINAL_DEFAULT_ROOT = BASE / (
    "ALANINE2D_GAPS_SPYDER_R4_SAFE_PREBRIDGE_GRID50_ALLCELL_PERIODICGP_"
    "EQ100PS_PROD400PS_DT001_REF_OVL010_FRONTIER010_COVER020_"
    "PARREP_FASTWHAM_20260919_141529"
)

LATEST = BASE / "ALANINE2D_GAPS_SPYDER_GRID50_ALLCELL_PERIODICGP_LATEST"

FALLBACK_REFERENCE = BASE / (
    "ALANINE2D_GAPS_V6_BUQCOMPARABLE_K200_OVL010_20PS_FRESH_20260821_135336/"
    "reference_only/BUQ_reference_fes.dat"
)

KBT_KCAL_PER_MOL = 0.00198720425864083 * 300.0

PI = np.pi
TWO_PI = 2.0*np.pi

RAMA_LIM = (-180.0, 180.0)
RAMA_TICKS = [-180, -90, 0, 90, 180]
RAMA_TICK_LABELS = [
    r"$-\pi$",
    r"$-\pi/2$",
    r"$0$",
    r"$\pi/2$",
    r"$\pi$",
]

# Same publication-style energy normalization used by the successful
# alanine controller diagnostic.
ENERGY_PERCENTILE = 97.0
ENERGY_FLOOR_KCAL = 5.0
ENERGY_CEILING_KCAL = 12.0
ENERGY_FLOOR_KBT = ENERGY_FLOOR_KCAL / KBT_KCAL_PER_MOL
ENERGY_CEILING_KBT = ENERGY_CEILING_KCAL / KBT_KCAL_PER_MOL

# Fixed validation domain used in panel C at every AL snapshot.
VALIDATION_FMAX_KBT = 10.0

# Number of points along each 1D cut.
CUT_N = 361

# Vertical separation of the three stacked cuts in panel E.
CUT_OFFSET_KBT = 5.0

PNG_DPI = 450

# Optional Spyder-friendly cutoff for the analysis shown in Figure 4.
# Example: MAX_AL_STEP = 50 uses the newest saved analysis at or before AL step 50,
# restricts the adaptive network to that snapshot, and truncates convergence history.
# Set to None to use the full/latest campaign.
MAX_AL_STEP = None

# Plot colors: identical WHAM/GP convention used in Figures 2 and 3.
C_WHAM = "#ff7f0e"   # WHAM: orange
C_GP = "#3B82F6"     # GP: richer blue
C_REF = "#111111"
C_REF_PROFILE = "#B2182B"  # contrasting Ref color used only in panel F
C_COVERAGE = "#1B9E77"
C_FOOT = "#FFFFFF"   # light sampling-distribution ellipse
ELLIPSE_CHI2 = 5.991 # 95% contour for a 2D Gaussian

TYPE_STYLE = {
    # High-contrast overlay colors chosen to remain visible on viridis.
    "bootstrap_seed":      dict(marker="o", color="#D81B60", label="Bootstrap"),
    "bootstrap_bridge":    dict(marker="D", color="#8E44AD", label="Bootstrap bridge"),
    "connectivity_bridge": dict(marker="s", color="#E31A1C", label="Connectivity bridge"),
    "information_target":  dict(marker="^", color="#00D4FF", label="Information target"),
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 25.0,
    "axes.labelsize": 32.0,
    "axes.titlesize": 32.0,
    "xtick.labelsize": 25.0,
    "ytick.labelsize": 25.0,
    "legend.fontsize": 26.0,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


# =============================================================================
# BASIC HELPERS
# =============================================================================

def read_json(path):
    return json.loads(Path(path).read_text())


def wrap_angle(x):
    x = np.asarray(x, float)
    return (x + PI) % TWO_PI - PI


def periodic_delta(a, b):
    return wrap_angle(np.asarray(a, float) - np.asarray(b, float))


def query_number(path):
    m = re.search(r"query_(\d+)", str(path))
    return int(m.group(1)) if m else None


def panel_label(ax, letter):
    ax.text(
        0.018, 0.982, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=36, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.3),
        zorder=100,
    )


def setup_rama(ax):
    ax.set_xlim(*RAMA_LIM)
    ax.set_ylim(*RAMA_LIM)
    ax.set_xticks(RAMA_TICKS)
    ax.set_yticks(RAMA_TICKS)
    ax.set_xticklabels(RAMA_TICK_LABELS)
    ax.set_yticklabels(RAMA_TICK_LABELS)
    ax.set_xlabel(r"$\mathit{\phi}$")
    ax.set_ylabel(r"$\mathit{\psi}$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)


def set_square(ax):
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass


def align_surface(reference, surface, mask):
    ref = np.asarray(reference, float)
    surf = np.asarray(surface, float)
    good = np.asarray(mask, bool) & np.isfinite(ref) & np.isfinite(surf)
    if not np.any(good):
        return surf.copy(), 0.0
    shift = float(np.median(ref[good] - surf[good]))
    return surf + shift, shift


def save_figure(fig, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / "Figure_Alanine_Dipeptide_GAPS_US.png"
    pdf = outdir / "Figure_Alanine_Dipeptide_GAPS_US.pdf"

    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")

    print("saved:", png)
    print("saved:", pdf)


# =============================================================================
# CAMPAIGN / ANALYSIS RESOLUTION
# =============================================================================

def analysis_files(root):
    fs = list((root / "analysis_history").rglob("online_analysis.npz"))
    if not fs:
        fs = list(root.rglob("online_analysis.npz"))
    return sorted(
        fs,
        key=lambda p: (
            query_number(p) if query_number(p) is not None else 10**9,
            p.stat().st_mtime,
        ),
    )


def resolve_root(user_root=None):
    if user_root:
        p = Path(user_root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    env = os.environ.get("ALANINE_CAMPAIGN_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    # Default: automatically use the alanine campaign whose saved
    # online_analysis.npz is newest on disk.
    candidates = []
    for p in BASE.glob("ALANINE2D_GAPS*"):
        if not p.is_dir():
            continue
        if not (p / "campaign_state.json").exists():
            continue
        fs = analysis_files(p)
        if fs:
            newest_analysis = max(fs, key=lambda f: f.stat().st_mtime)
            candidates.append((newest_analysis.stat().st_mtime, p, newest_analysis))

    if candidates:
        _, p, newest_analysis = max(candidates, key=lambda t: t[0])
        print("Auto-selected newest alanine campaign:")
        print(" ", p.resolve())
        print("Newest saved analysis in that campaign:")
        print(" ", newest_analysis.resolve())
        return p.resolve()

    # Conservative fallback to the previously audited campaign.
    if FINAL_DEFAULT_ROOT.exists() and analysis_files(FINAL_DEFAULT_ROOT):
        return FINAL_DEFAULT_ROOT.resolve()

    raise FileNotFoundError(
        "No alanine GAPS campaign containing campaign_state.json and online_analysis.npz was found."
    )

def load_state(root):
    p = root / "campaign_state.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return read_json(p)


def latest_analysis_file(root, state, max_al_step=None):
    if max_al_step is None:
        p = state.get("latest_full_analysis")
        if p and Path(p).exists():
            return Path(p)

        fs = analysis_files(root)
        if not fs:
            raise FileNotFoundError(f"No online_analysis.npz under {root}")
        return max(fs, key=lambda p: p.stat().st_mtime)

    fs = [
        p for p in analysis_files(root)
        if query_number(p) is not None and query_number(p) <= int(max_al_step)
    ]
    if not fs:
        raise FileNotFoundError(
            f"No online_analysis.npz with AL step <= {int(max_al_step)} under {root}"
        )
    return max(fs, key=lambda p: (query_number(p), p.stat().st_mtime))


def state_for_analysis(state, z):
    """Restrict the campaign state to windows represented by the selected analysis."""
    if "connected_window_ids" not in z:
        return state
    keep = set(int(x) for x in np.asarray(z["connected_window_ids"]).ravel())
    out = dict(state)
    out["windows"] = [
        w for w in state.get("windows", [])
        if w.get("status") == "complete" and int(w.get("id", -1)) in keep
    ]
    out["current_query"] = None
    return out


# =============================================================================
# PUBLISHED / REFERENCE FES
# =============================================================================

def reference_path(root):
    local = root / "simulations_essentials" / "BUQ_reference_fes.dat"
    if local.exists():
        return local
    if FALLBACK_REFERENCE.exists():
        return FALLBACK_REFERENCE
    raise FileNotFoundError(
        "Could not find BUQ_reference_fes.dat in campaign or fallback location."
    )


def load_reference_grid(root):
    """
    Return phi radians, psi radians, and reference F [psi,phi] in kBT.
    """
    p = reference_path(root)
    data = np.genfromtxt(p, comments="#")

    if data.ndim != 2 or data.shape[1] < 3:
        raise RuntimeError(f"Could not parse reference FES: {p}")

    ph = np.asarray(data[:, 0], float)
    ps = np.asarray(data[:, 1], float)
    ff = np.asarray(data[:, 2], float)

    good = np.isfinite(ph) & np.isfinite(ps) & np.isfinite(ff)
    ph = wrap_angle(ph[good])
    ps = wrap_angle(ps[good])
    ff = ff[good]

    phu = np.unique(np.round(ph, 12))
    psu = np.unique(np.round(ps, 12))
    phu.sort()
    psu.sort()

    R = np.full((len(psu), len(phu)), np.nan, float)

    ip = np.searchsorted(phu, np.round(ph, 12))
    jp = np.searchsorted(psu, np.round(ps, 12))
    R[jp, ip] = ff

    if np.any(~np.isfinite(R)):
        raise RuntimeError(
            f"Reference FES is not a complete rectangular grid: {p}"
        )

    R -= np.nanmin(R)
    R /= KBT_KCAL_PER_MOL

    return phu, psu, R, p


def periodic_interpolator(x, y, Z):
    """
    x, y are 1D radian coordinates.
    Z is [y,x].
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Z = np.asarray(Z, float)

    xe = np.r_[x - TWO_PI, x, x + TWO_PI]
    ye = np.r_[y - TWO_PI, y, y + TWO_PI]
    Ze = np.tile(Z, (3, 3))

    return RegularGridInterpolator(
        (ye, xe),
        Ze,
        bounds_error=False,
        fill_value=np.nan,
    )


def eval_periodic(interp, phi, psi):
    ph = wrap_angle(np.asarray(phi, float))
    ps = wrap_angle(np.asarray(psi, float))
    pts = np.column_stack([ps.ravel(), ph.ravel()])
    out = np.asarray(interp(pts), float)
    return out.reshape(np.asarray(ph).shape)


# =============================================================================
# FINAL RECONSTRUCTION
# =============================================================================

def load_final(root, state, max_al_step=None):
    p = latest_analysis_file(root, state, max_al_step=max_al_step)
    z = dict(np.load(p, allow_pickle=False))

    required = [
        "x_edges", "y_edges", "x_centers", "y_centers",
        "wham_f", "support", "gp_base_mean_grid",
        "gp_recon_x_edges", "gp_recon_y_edges",
        "gp_recon_x_centers", "gp_recon_y_centers",
        "gp_recon_mean_grid",
    ]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"Final analysis is missing arrays: {missing}")

    x = np.asarray(z["x_centers"], float)
    y = np.asarray(z["y_centers"], float)
    xe = np.asarray(z["x_edges"], float)
    ye = np.asarray(z["y_edges"], float)

    wham = np.asarray(z["wham_f"], float)
    support = np.asarray(z["support"], bool)
    gp_base = np.asarray(z["gp_base_mean_grid"], float)

    gxc = np.asarray(z["gp_recon_x_centers"], float)
    gyc = np.asarray(z["gp_recon_y_centers"], float)
    gxe = np.asarray(z["gp_recon_x_edges"], float)
    gye = np.asarray(z["gp_recon_y_edges"], float)
    gp_hi = np.asarray(z["gp_recon_mean_grid"], float)

    # Match the production diagnostic's reference interpolation.
    ph_ref, ps_ref, Rref, refpath = load_reference_grid(root)
    ref_interp = periodic_interpolator(ph_ref, ps_ref, Rref)

    PH0, PS0 = np.meshgrid(x, y, indexing="xy")
    ref_base = eval_periodic(ref_interp, PH0, PS0)

    PHH, PSH = np.meshgrid(gxc, gyc, indexing="xy")
    ref_hi = eval_periodic(ref_interp, PHH, PSH)

    # Put WHAM/GP into the reference gauge using the same supported base-grid cells.
    w0 = wham.copy()
    g0 = gp_base.copy()

    if np.any(support):
        w0 -= np.nanmin(w0[support])
        g0 -= np.nanmin(g0[support])

    refmask = (
        support
        & np.isfinite(ref_base)
        & np.isfinite(w0)
        & np.isfinite(g0)
    )

    if not np.any(refmask):
        raise RuntimeError("No common supported reference/WHAM/GP cells.")

    wham_ref, wshift = align_surface(ref_base, w0, refmask)
    gp_ref, gshift = align_surface(ref_base, g0, refmask)
    gp_hi_ref = gp_hi + gshift

    wrmse = float(np.sqrt(np.mean((wham_ref[refmask] - ref_base[refmask])**2)))
    grmse = float(np.sqrt(np.mean((gp_ref[refmask] - ref_base[refmask])**2)))

    # Shared publication energy scale.
    pieces = [
        wham_ref[refmask],
        gp_hi_ref[np.isfinite(gp_hi_ref)],
        ref_hi[np.isfinite(ref_hi)],
    ]
    vals = np.concatenate([v.ravel() for v in pieces if v.size])
    vmax = float(np.nanpercentile(vals, ENERGY_PERCENTILE))
    vmax = float(min(ENERGY_CEILING_KBT, max(ENERGY_FLOOR_KBT, vmax)))

    return dict(
        path=p,
        raw=z,
        ref_path=refpath,
        x=x, y=y, xe=xe, ye=ye,
        gxc=gxc, gyc=gyc, gxe=gxe, gye=gye,
        reference_base=ref_base,
        reference_hi=ref_hi,
        wham=wham_ref,
        gp_base=gp_ref,
        gp_hi=gp_hi_ref,
        support=support,
        refmask=refmask,
        wrmse=wrmse,
        grmse=grmse,
        vmax=vmax,
        reference_interp=ref_interp,
    )


# =============================================================================
# NETWORK / WINDOW MARKERS
# =============================================================================

def completed_windows(state):
    return [
        w for w in state.get("windows", [])
        if w.get("status") == "complete"
    ]


def window_dir(root, wid):
    return Path(root) / "windows" / f"window_{int(wid):06d}"


def load_window_phi_psi_samples(root, wid):
    """Load all replica samples for one alanine umbrella window."""
    chunks = []

    # This mirrors the sampling-file layout used by the other GAPS-US systems.
    for p in sorted(window_dir(root, wid).glob("rep_*/samples.dat")):
        try:
            a = np.loadtxt(p, comments="#", ndmin=2)
            if a.ndim != 2 or a.shape[1] < 3:
                continue

            # samples.dat convention: first column is time/index, next two are phi, psi.
            xy = np.asarray(a[:, 1:3], float)

            if len(xy):
                chunks.append(xy)
        except Exception:
            pass

    if not chunks:
        return None

    xy = np.vstack(chunks)

    # Alanine controller normally stores angles in radians.  If a file instead
    # contains degrees, convert automatically.
    finite = np.isfinite(xy)
    if np.any(finite) and np.nanmax(np.abs(xy[finite])) > (2.0*np.pi + 0.5):
        xy = np.deg2rad(xy)

    return xy


def periodic_distribution_ellipse(center, xy):
    """
    95% covariance ellipse for a periodic (phi, psi) distribution.

    Samples are unwrapped relative to the umbrella center before the mean and
    covariance are computed, avoiding artificial broadening across +/-pi.
    """
    if xy is None or len(xy) < 5:
        return None

    center = np.asarray(center, float)
    xy = np.asarray(xy, float)

    good = np.all(np.isfinite(xy), axis=1)
    xy = xy[good]
    if len(xy) < 5:
        return None

    dphi = periodic_delta(xy[:, 0], center[0])
    dpsi = periodic_delta(xy[:, 1], center[1])

    unwrapped = np.column_stack([
        center[0] + dphi,
        center[1] + dpsi,
    ])

    mean = wrap_angle(np.mean(unwrapped, axis=0))

    cov = np.cov(unwrapped.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1.0e-14)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width_rad  = 2.0*np.sqrt(ELLIPSE_CHI2*vals[0])
    height_rad = 2.0*np.sqrt(ELLIPSE_CHI2*vals[1])
    angle_deg  = float(np.degrees(np.arctan2(vecs[1,0], vecs[0,0])))

    return (
        float(np.degrees(width_rad)),
        float(np.degrees(height_rad)),
        angle_deg,
        mean,
    )


def draw_periodic_distribution_ellipse(ax, center, ellipse):
    """Draw the ellipse plus periodic copies needed near +/-180 degree edges."""
    if ellipse is None:
        return

    width, height, angle, mean = ellipse
    cdeg = np.rad2deg(np.asarray(mean, float))

    # Draw periodic copies.  Matplotlib clipping keeps only the visible pieces,
    # so windows near the Ramachandran boundaries look continuous.
    for sx in (-360.0, 0.0, 360.0):
        for sy in (-360.0, 0.0, 360.0):
            ax.add_patch(
                Ellipse(
                    (cdeg[0] + sx, cdeg[1] + sy),
                    width,
                    height,
                    angle=angle,
                    fill=False,
                    edgecolor=C_FOOT,
                    lw=1.15,
                    alpha=0.52,
                    zorder=3.5,
                )
            )


def overlap_cache_path(root):
    p = root / "analysis_current" / "measured_overlap_cache.json"
    if p.exists():
        return p

    fs = list(root.rglob("measured_overlap_cache.json"))
    if not fs:
        return None

    return max(fs, key=lambda q: q.stat().st_mtime)


def measured_edges(root, state, connected_ids):
    p = overlap_cache_path(root)
    if p is None:
        return []

    threshold = float(state.get("minimum_overlap", 0.10))
    ids = set(int(i) for i in connected_ids)

    cache = read_json(p)
    out = []

    for key, rec in cache.items():
        try:
            i, j = map(int, key.split(":"))
            ov = float(rec.get("overlap", 0.0))
        except Exception:
            continue

        if i in ids and j in ids and ov >= threshold:
            out.append((i, j, ov))

    return out


def periodic_segment(ax, a, b, **kwargs):
    """
    Plot shortest toroidal connection on [-180,180]^2.
    """
    a = np.asarray(a, float)
    d = periodic_delta(np.asarray(b, float), a)

    aa = np.rad2deg(a)
    bb = np.rad2deg(a + d)

    for sx in (-360.0, 0.0, 360.0):
        for sy in (-360.0, 0.0, 360.0):
            ax.plot(
                [aa[0] + sx, bb[0] + sx],
                [aa[1] + sy, bb[1] + sy],
                **kwargs,
            )


def plot_network(ax, root, state, final):
    connected_ids = np.asarray(
        final["raw"].get("connected_window_ids", []), int
    ).ravel().tolist()

    if not connected_ids:
        connected_ids = [
            int(w["id"]) for w in completed_windows(state)
        ]

    idset = set(connected_ids)
    byid = {
        int(w["id"]): w
        for w in completed_windows(state)
        if int(w["id"]) in idset
    }

    # GP background.
    bg = ax.pcolormesh(
        np.rad2deg(final["gxe"]),
        np.rad2deg(final["gye"]),
        np.clip(final["gp_hi"], 0, final["vmax"]),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=final["vmax"],
        rasterized=True,
    )

    # Sampling distributions: same visual idea as Müller–Brown Figure 2A.
    n_distributions = 0
    for w in byid.values():
        xy = load_window_phi_psi_samples(root, int(w["id"]))
        ell = periodic_distribution_ellipse(w["center"], xy)
        if ell is not None:
            draw_periodic_distribution_ellipse(ax, w["center"], ell)
            n_distributions += 1

    print(
        f"Panel A: drew sampling-distribution ellipses for "
        f"{n_distributions}/{len(byid)} connected windows"
    )

    # Measured-overlap graph.
    for i, j, ov in measured_edges(root, state, connected_ids):
        if i not in byid or j not in byid:
            continue
        periodic_segment(
            ax,
            byid[i]["center"],
            byid[j]["center"],
            color="black",
            lw=0.62,
            alpha=0.30,
            zorder=4,
        )

    # Adaptive windows.
    used = set()
    for w in byid.values():
        kind = w.get("kind", "window")
        st = TYPE_STYLE.get(
            kind,
            dict(marker="o", color="0.6", label=kind),
        )

        c = np.rad2deg(np.asarray(w["center"], float))

        ax.scatter(
            [c[0]], [c[1]],
            s=50,
            marker=st["marker"],
            c=st["color"],
            edgecolors="none",
            linewidths=0.0,
            zorder=8,
        )
        used.add(st["label"])

    setup_rama(ax)
    panel_label(ax, "A")

    return bg


def network_legend_handles(state):
    present = {w.get("kind") for w in completed_windows(state)}

    order = [
        "bootstrap_seed",
        "bootstrap_bridge",
        "information_target",
        "connectivity_bridge",
    ]

    handles = []
    labels = []

    for k in order:
        if k not in present:
            continue

        st = TYPE_STYLE[k]
        handles.append(
            Line2D(
                [0], [0],
                marker=st["marker"],
                linestyle="None",
                markerfacecolor=st["color"],
                markeredgecolor="none",
                markeredgewidth=0.0,
                markersize=10,
            )
        )
        labels.append(st["label"])

    return handles, labels


# =============================================================================
# PANEL A: CV SCHEMATIC
# =============================================================================

def plot_cv_schematic(ax):
    """
    Chemically meaningful 2D schematic of ACE-Ala-NME.

    The two backbone torsions are highlighted on the bonds around C_alpha:
        phi : C(ACE)-N-C_alpha-C'
        psi : N-C_alpha-C'-N(NME)

    Atom-index definitions used in the simulations are kept for the caption /
    methods rather than cluttering the molecular drawing.
    """
    # Backbone / terminal-group anchor positions in axes coordinates.
    CH3L = np.array([0.06, 0.56])
    C0   = np.array([0.21, 0.62])   # ACE carbonyl C
    O0   = np.array([0.21, 0.84])
    N1   = np.array([0.38, 0.52])
    CA   = np.array([0.55, 0.64])
    CB   = np.array([0.55, 0.88])   # alanine methyl substituent
    C1   = np.array([0.72, 0.53])   # alanine carbonyl C
    O1   = np.array([0.72, 0.29])
    N2   = np.array([0.88, 0.64])
    CH3R = np.array([0.98, 0.54])

    def bond(a, b, lw=2.7, color="0.25", z=2):
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=color, lw=lw, solid_capstyle="round", zorder=z)

    def double_bond(a, b):
        d = b - a
        n = np.array([-d[1], d[0]], float)
        n /= max(np.linalg.norm(n), 1e-12)
        off = 0.008*n
        bond(a+off, b+off, lw=1.9)
        bond(a-off, b-off, lw=1.9)

    # Molecular skeleton.
    bond(CH3L, C0)
    double_bond(C0, O0)
    bond(C0, N1)
    bond(N1, CA)
    bond(CA, C1)
    double_bond(C1, O1)
    bond(C1, N2)
    bond(N2, CH3R)
    bond(CA, CB, lw=2.3)

    # Highlight the torsional central bonds.
    bond(N1, CA, lw=5.2, color="#6A3D9A", z=3)
    bond(CA, C1, lw=5.2, color="#1B9E77", z=3)

    # Atom/group labels.
    txt = dict(ha="center", va="center", fontsize=15, zorder=6)
    ax.text(*CH3L, r"$\mathrm{CH_3}$", **txt)
    ax.text(*C0, r"$\mathrm{C}$", **txt)
    ax.text(*O0, r"$\mathrm{O}$", **txt)
    ax.text(*N1, r"$\mathrm{N}$", color="#2C7FB8", **txt)
    ax.text(*CA, r"$\mathrm{C_\alpha}$", **txt)
    ax.text(*CB, r"$\mathrm{CH_3}$", **txt)
    ax.text(*C1, r"$\mathrm{C}$", **txt)
    ax.text(*O1, r"$\mathrm{O}$", **txt)
    ax.text(*N2, r"$\mathrm{N}$", color="#2C7FB8", **txt)
    ax.text(*CH3R, r"$\mathrm{CH_3}$", **txt)

    # Small backbone hydrogens on the two amide nitrogens.
    H1 = np.array([0.36, 0.31])
    H2 = np.array([0.89, 0.84])
    bond(N1, H1, lw=1.5, color="0.45")
    bond(N2, H2, lw=1.5, color="0.45")
    ax.text(*H1, r"$\mathrm{H}$", fontsize=12, ha="center", va="center")
    ax.text(*H2, r"$\mathrm{H}$", fontsize=12, ha="center", va="center")

    # Curved torsion arrows.
    phi_mid = 0.5*(N1 + CA)
    psi_mid = 0.5*(CA + C1)

    ax.annotate(
        "",
        xy=(phi_mid[0]+0.055, phi_mid[1]-0.11),
        xytext=(phi_mid[0]-0.075, phi_mid[1]+0.10),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.0,
            color="#6A3D9A",
            connectionstyle="arc3,rad=-0.55",
        ),
        zorder=7,
    )
    ax.text(phi_mid[0]-0.09, phi_mid[1]-0.12,
            r"$\mathit{\phi}$", color="#6A3D9A", fontsize=22,
            ha="center", va="center")

    ax.annotate(
        "",
        xy=(psi_mid[0]+0.07, psi_mid[1]+0.10),
        xytext=(psi_mid[0]-0.05, psi_mid[1]-0.11),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.0,
            color="#1B9E77",
            connectionstyle="arc3,rad=-0.55",
        ),
        zorder=7,
    )
    ax.text(psi_mid[0]+0.10, psi_mid[1]-0.10,
            r"$\mathit{\psi}$", color="#1B9E77", fontsize=22,
            ha="center", va="center")

    # Fragment labels.
    ax.text(0.17, 0.08, "ACE", fontsize=13, ha="center", va="center")
    ax.text(0.56, 0.08, "Ala", fontsize=13, ha="center", va="center")
    ax.text(0.89, 0.08, "NME", fontsize=13, ha="center", va="center")

    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0.02, 0.98)
    ax.set_box_aspect(1)
    ax.axis("off")
    ax.set_title("Alanine dipeptide CVs", pad=10)
    panel_label(ax, "A")


# =============================================================================
# PANEL E: WRAPPED 1D CUTS
# =============================================================================

def build_surface_interpolators(final):
    # Reference interpolator already exists.
    refi = final["reference_interp"]

    whami = periodic_interpolator(
        final["x"],
        final["y"],
        final["wham"],
    )

    gpi = periodic_interpolator(
        final["gxc"],
        final["gyc"],
        final["gp_hi"],
    )

    return refi, whami, gpi


def reference_minimum_location(final):
    R = np.asarray(final["reference_hi"], float)
    iy, ix = np.unravel_index(np.nanargmin(R), R.shape)
    phi0 = float(final["gxc"][ix])
    psi0 = float(final["gyc"][iy])
    return phi0, psi0


def plot_cuts(ax, final):
    refi, _, gpi = build_surface_interpolators(final)
    phi0, psi0 = reference_minimum_location(final)

    s = np.linspace(-PI, PI, CUT_N)
    sdeg = np.rad2deg(s)

    paths = [
        (
            "Horizontal",
            wrap_angle(phi0 + s),
            np.full_like(s, psi0),
            "-",
            0.0,
        ),
        (
            "Vertical",
            np.full_like(s, phi0),
            wrap_angle(psi0 + s),
            "--",
            CUT_OFFSET_KBT,
        ),
        (
            "Diagonal",
            wrap_angle(phi0 + s),
            wrap_angle(psi0 + s),
            ":",
            2.0*CUT_OFFSET_KBT,
        ),
    ]

    for name, ph, ps, ls, offset in paths:
        ref = eval_periodic(refi, ph, ps)
        gp = eval_periodic(gpi, ph, ps)

        goodref = np.isfinite(ref)
        zero = float(np.nanmin(ref[goodref])) if np.any(goodref) else 0.0

        refp = ref - zero + offset
        gpp = gp - zero + offset

        ax.plot(sdeg, refp, color=C_REF_PROFILE, lw=2.2, ls=ls, alpha=0.98)
        ax.plot(sdeg, gpp, color=C_GP, lw=2.2, ls=ls, alpha=0.98)

    # Only identify Ref and GP. The line styles are explained in the caption.
    method_handles = [
        Line2D([0],[0], color=C_REF_PROFILE, lw=2.5, ls="-",
               label=r"$\mathrm{Ref}$"),
        Line2D([0],[0], color=C_GP, lw=2.5, ls="-",
               label=r"$\mathrm{GP}$"),
    ]
    ax.legend(
        handles=method_handles,
        loc="upper center",
        frameon=False,
        ncol=1,
        handlelength=2.4,
    )

    ax.set_xlim(-180, 180)
    ax.set_xticks(RAMA_TICKS)
    ax.set_xticklabels(RAMA_TICK_LABELS)
    ax.set_xlabel("Wrapped coordinate")
    ax.set_ylabel(r"$\mathit{F}$ ($\mathit{k}_{\mathrm{B}}\mathit{T}$)")
    set_square(ax)
    panel_label(ax, "F")

    print(
        "Cut reference point (reference minimum): "
        f"phi={np.rad2deg(phi0):.2f} deg, "
        f"psi={np.rad2deg(psi0):.2f} deg"
    )


# =============================================================================
# PANEL F: RMSE HISTORY
# =============================================================================

def snapshot_convergence(root, path, validation_fmax):
    """
    Fixed-domain convergence diagnostic.

    The validation mask is defined ONLY from F_ref:
        F_ref <= validation_fmax

    GP gauge alignment uses only the CURRENT physically supported WHAM cells,
    then GP error is evaluated over the entire fixed validation mask.  Thus an
    early GP is penalized for regions it has not yet learned.

    WHAM itself is not assigned an RMSE outside support; instead we report the
    fraction of the fixed validation domain that WHAM physically supports.
    """
    z = dict(np.load(path, allow_pickle=False))

    needed = [
        "x_centers", "y_centers",
        "wham_f", "support",
        "gp_base_mean_grid",
    ]
    if any(k not in z for k in needed):
        return None

    x = np.asarray(z["x_centers"], float)
    y = np.asarray(z["y_centers"], float)
    gp = np.asarray(z["gp_base_mean_grid"], float)
    sup = np.asarray(z["support"], bool)

    ph_ref, ps_ref, Rref, _ = load_reference_grid(root)
    refi = periodic_interpolator(ph_ref, ps_ref, Rref)

    PH, PS = np.meshgrid(x, y, indexing="xy")
    ref = eval_periodic(refi, PH, PS)

    validation = np.isfinite(ref) & (ref <= float(validation_fmax))
    if not np.any(validation):
        return None

    # Gauge is learned only where the current calculation has physical support.
    gauge = sup & np.isfinite(ref) & np.isfinite(gp)
    if not np.any(gauge):
        return None

    g0 = gp.copy()
    if np.any(sup & np.isfinite(g0)):
        g0 -= np.nanmin(g0[sup & np.isfinite(g0)])

    gp_aligned, _ = align_surface(ref, g0, gauge)

    # A fixed-domain RMSE is meaningful only if the GP predicts the whole
    # validation set.  The production periodic GP normally does.
    if not np.all(np.isfinite(gp_aligned[validation])):
        gp_rmse = np.nan
    else:
        gp_rmse = float(
            np.sqrt(np.mean(
                (gp_aligned[validation] - ref[validation])**2
            ))
        )

    coverage = float(
        np.sum(sup & validation) / np.sum(validation)
    )

    ids = np.asarray(
        z.get("connected_window_ids", []), int
    ).ravel()

    if len(ids):
        nwin = int(len(ids))
    else:
        nwin = int(
            len(np.asarray(z.get("umbrella_centers", [])))
        )

    return dict(
        query=query_number(path),
        windows=nwin,
        gp_rmse=gp_rmse,
        coverage=coverage,
        validation_cells=int(np.sum(validation)),
        supported_validation_cells=int(np.sum(sup & validation)),
        path=path,
    )


def collect_history(root, validation_fmax, max_al_step=None):
    rows = []

    for p in analysis_files(root):
        q = query_number(p)
        if max_al_step is not None and q is not None and q > int(max_al_step):
            continue
        try:
            r = snapshot_convergence(root, p, validation_fmax)
            if r is not None:
                rows.append(r)
        except Exception as exc:
            print("WARNING history:", p, exc)

    # One snapshot per selected query.
    dedup = {}
    for r in rows:
        key = r["query"] if r["query"] is not None else str(r["path"])
        dedup[key] = r

    rows = list(dedup.values())
    rows.sort(
        key=lambda r:
        r["query"] if r["query"] is not None else 10**9
    )

    return rows

def load_uniform_csv(path):
    if path is None:
        return []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    rows = []

    with p.open(newline="") as fh:
        rd = csv.DictReader(fh)

        if rd.fieldnames is None:
            raise RuntimeError("Uniform CSV has no header.")

        names = {n.lower(): n for n in rd.fieldnames}

        if "windows" not in names or "rmse" not in names:
            raise RuntimeError(
                "Uniform CSV must contain columns 'windows' and 'rmse'."
            )

        for row in rd:
            rows.append(
                dict(
                    windows=float(row[names["windows"]]),
                    rmse=float(row[names["rmse"]]),
                    label=row.get(names.get("label", ""), "") if "label" in names else "",
                )
            )

    return rows


def plot_convergence(ax, history, uniform_rows, validation_fmax):
    if not history:
        ax.text(
            0.5, 0.5,
            "No analysis history found",
            transform=ax.transAxes,
            ha="center", va="center",
        )
        set_square(ax)
        panel_label(ax, "C")
        return

    n = np.asarray([r["windows"] for r in history], float)
    gr = np.asarray([r["gp_rmse"] for r in history], float)
    cov = 100.0*np.asarray([r["coverage"] for r in history], float)

    good = np.isfinite(n) & np.isfinite(gr)

    l1, = ax.plot(
        n[good], gr[good],
        marker="s",
        ms=9.6,
        lw=2.3,
        color=C_GP,
        markeredgecolor="none",
        markeredgewidth=0,
        label="GP RMSE",
    )

    # Optional predetermined-grid points must already have been evaluated with
    # the same fixed validation-domain definition.
    extra_handles = [l1]
    extra_labels = [l1.get_label()]

    if uniform_rows:
        uw = np.asarray([r["windows"] for r in uniform_rows], float)
        ur = np.asarray([r["rmse"] for r in uniform_rows], float)
        order = np.argsort(uw)

        lu, = ax.plot(
            uw[order],
            ur[order],
            marker="D",
            ms=9.6,
            lw=1.5,
            ls="--",
            color="0.38",
            label="Uniform US",
        )
        extra_handles.append(lu)
        extra_labels.append(lu.get_label())

    ax.set_xlabel("Number of windows")
    ax.set_ylabel(r"GP RMSE ($\mathit{k}_{\mathrm{B}}\mathit{T}$)", color=C_GP)
    ax.tick_params(axis="y", labelcolor=C_GP)

    ax2 = ax.twinx()
    l2, = ax2.plot(
        n, cov,
        marker="o",
        ms=9.6,
        lw=2.2,
        color=C_COVERAGE,
        markeredgecolor="none",
        markeredgewidth=0,
        label="WHAM coverage",
    )
    ax2.set_ylabel("WHAM coverage (%)", color=C_COVERAGE)
    ax2.tick_params(axis="y", labelcolor=C_COVERAGE, pad=10)
    ax2.set_ylim(0, 102)

    extra_handles.append(l2)
    extra_labels.append(l2.get_label())

    # GP RMSE and WHAM coverage are already identified by their colored axes.
    # Only the optional uniform-US comparison needs a legend entry.
    if uniform_rows:
        ax.legend(
            [lu],
            [lu.get_label()],
            loc="upper right",
            frameon=False,
        )

    set_square(ax)
    panel_label(ax, "C")


# =============================================================================
# MAIN FIGURE
# =============================================================================

def make_figure(root, state, final, history, uniform_rows, validation_fmax, outdir):
    """
    Final main-text 2x3 alanine layout:

      A  F_ref + adaptive network
      B  F_WHAM
      C  fixed-domain convergence
      D  F_ref
      E  F_GP
      F  wrapped 1D profiles

    The adaptive-network marker convention matches Figures 2 and 3; marker
    meanings are left to the main-text/caption description rather than a panel legend.
    A horizontal shared landscape colorbar is placed below D/E so it cannot
    overlap panel F.
    """
    fig, ax = plt.subplots(2, 3, figsize=(18.2, 13.2))

    # Extra horizontal spacing prevents the y-axis labels of neighboring panels
    # from running into one another.
    fig.subplots_adjust(
        left=0.060,
        right=0.955,
        bottom=0.145,
        top=0.900,
        wspace=0.40,
        hspace=0.325,
    )

    # A. Reference FES + adaptive network
    im = plot_network(ax[0,0], root, state, final)
    ax[0,0].set_title(r"$\mathit{F}^{\mathrm{GP}}$ + adaptive network", pad=10)

    # B. Direct periodic WHAM
    whplot = np.where(
        final["support"],
        np.clip(final["wham"], 0, final["vmax"]),
        np.nan,
    )
    ax[0,1].pcolormesh(
        np.rad2deg(final["xe"]),
        np.rad2deg(final["ye"]),
        whplot,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=final["vmax"],
        rasterized=True,
    )
    setup_rama(ax[0,1])
    ax[0,1].set_ylabel("")
    ax[0,1].tick_params(axis="y", labelleft=False)
    ax[0,1].set_title(r"$\mathit{F}^{\mathrm{WHAM}}$", pad=10)
    panel_label(ax[0,1], "B")
    # A
    im = plot_network(ax[0,0], root, state, final)
    ax[0,0].set_title(r"$\mathit{F}^{\mathrm{GP}}$ + adaptive network", pad=10)
    
    # B
    ...
    setup_rama(ax[0,1])
    ax[0,1].set_ylabel("")
    ax[0,1].tick_params(axis="y", labelleft=False)
    ax[0,1].set_title(r"$\mathit{F}^{\mathrm{WHAM}}$", pad=10)
    panel_label(ax[0,1], "B")
    
    # Remove x labels/tick labels from A and B
    for a in (ax[0,0], ax[0,1]):
        a.set_xlabel("")
        a.tick_params(axis="x", labelbottom=False)
    
    # C. Fixed-domain convergence
    plot_convergence(
        ax[0,2],
        history,
        uniform_rows,
        validation_fmax,
    )

    # D. Reference FES
    ax[1,0].pcolormesh(
        np.rad2deg(final["gxe"]),
        np.rad2deg(final["gye"]),
        np.clip(final["reference_hi"], 0, final["vmax"]),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=final["vmax"],
        rasterized=True,
    )
    setup_rama(ax[1,0])
    ax[1,0].set_title(r"$\mathit{F}^{\mathrm{Ref}}$", pad=10)
    panel_label(ax[1,0], "D")

    # E. Periodic GP
    ax[1,1].pcolormesh(
        np.rad2deg(final["gxe"]),
        np.rad2deg(final["gye"]),
        np.clip(final["gp_hi"], 0, final["vmax"]),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=final["vmax"],
        rasterized=True,
    )
    setup_rama(ax[1,1])
    ax[1,1].set_ylabel("")
    ax[1,1].tick_params(axis="y", labelleft=False)
    ax[1,1].set_title(r"$\mathit{F}^{\mathrm{GP}}$", pad=10)
    panel_label(ax[1,1], "E")

    # F. Wrapped 1D profiles
    plot_cuts(ax[1,2], final)

    # Compact vertical colorbar centered in the gap between the first two columns.
    fig.canvas.draw()
    bbox_a = ax[0,0].get_position()
    bbox_b = ax[0,1].get_position()
    bbox_d = ax[1,0].get_position()
    bbox_e = ax[1,1].get_position()

    gap_center_x = 0.46 * (bbox_a.x1 + bbox_b.x0)
    mid_y = 0.5 * (bbox_d.y0 + bbox_a.y1)
    cbar_h = 0.3
    cbar_w = 0.016

    cax = fig.add_axes([
        gap_center_x - 0.5*cbar_w,
        mid_y - 0.5*cbar_h,
        cbar_w,
        cbar_h,
    ])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label(r"$\mathit{F}$ ($\mathit{k}_{\mathrm{B}}\mathit{T}$)", labelpad=8)
    cb.ax.tick_params(labelsize=25.0)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "Figure_4_Alanine_Dipeptide_GAPS_US.png"
    pdf = outdir / "Figure_4_Alanine_Dipeptide_GAPS_US.pdf"

    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")

    print("saved:", png)
    print("saved:", pdf)

    show_or_close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default=None,
        help="Exact alanine adaptive campaign root.",
    )

    ap.add_argument(
        "--outdir",
        default=".",
        help="Output directory.",
    )

    ap.add_argument(
        "--max-al-step",
        type=int,
        default=MAX_AL_STEP,
        help=(
            "Use results through this AL step. Default is MAX_AL_STEP near "
            "the top of the script; None uses the full/latest campaign."
        ),
    )

    ap.add_argument(
        "--validation-fmax",
        type=float,
        default=VALIDATION_FMAX_KBT,
        help=(
            "Fixed F_ref threshold in kBT used for panel-C validation "
            f"(default {VALIDATION_FMAX_KBT:g})."
        ),
    )

    ap.add_argument(
        "--uniform-csv",
        default=None,
        help=(
            "Optional uniform-US comparison CSV with columns windows,rmse[,label]. "
            "Those RMSE values should use the same fixed validation domain."
        ),
    )

    args = ap.parse_args()

    root = resolve_root(args.root)
    outdir = Path(args.outdir).expanduser().resolve()
    validation_fmax = float(args.validation_fmax)

    print("="*90)
    print("FIGURE 4 — ALANINE-DIPEPTIDE 6-PANEL MAIN-TEXT FIGURE")
    print("="*90)
    print("campaign root:")
    print(" ", root)
    print("max AL step:", "latest/all" if args.max_al_step is None else args.max_al_step)

    state = load_state(root)
    final = load_final(root, state, max_al_step=args.max_al_step)
    plot_state = state_for_analysis(state, final["raw"])

    print("state phase:")
    print(" ", state.get("phase"))

    ncomplete = len(completed_windows(plot_state))
    analyzed_ids = np.asarray(
        final["raw"].get("connected_window_ids", []), int
    ).ravel()
    nanalyzed = int(len(analyzed_ids)) if len(analyzed_ids) else int(
        len(np.asarray(final["raw"].get("umbrella_centers", [])))
    )

    print("completed physical windows in selected snapshot:", ncomplete)
    print("physical windows in selected saved analysis:", nanalyzed)

    if ncomplete != nanalyzed:
        print(
            "NOTE: the final campaign state and latest saved analysis contain "
            "different window counts. The figure uses the saved analysis data; "
            "it does not silently recompute WHAM/GP."
        )

    print("selected analysis:")
    print(" ", final["path"])
    print("selected AL step:", query_number(final["path"]))

    print("reference FES:")
    print(" ", final["ref_path"])

    print(
        f"final WHAM RMSE on its supported reference domain = "
        f"{final['wrmse']:.6f} kBT"
    )

    print(
        f"final GP RMSE on its supported reference domain   = "
        f"{final['grmse']:.6f} kBT"
    )

    print(
        f"fixed panel-C validation domain: "
        f"F^Ref <= {validation_fmax:g} kBT"
    )

    print(
        f"shared landscape color scale: "
        f"0 to {final['vmax']:.3f} kBT"
    )

    history = collect_history(root, validation_fmax, max_al_step=args.max_al_step)
    print("analysis-history snapshots:", len(history))

    if history:
        print(
            "panel-C final analyzed point: "
            f"N={history[-1]['windows']} windows, "
            f"GP RMSE={history[-1]['gp_rmse']:.6f} kBT, "
            f"WHAM coverage={100.0*history[-1]['coverage']:.2f}%"
        )

    uniform_rows = load_uniform_csv(args.uniform_csv)

    if args.uniform_csv is None:
        print(
            "Uniform-grid CSV not supplied: panel C shows fixed-domain "
            "GP RMSE and WHAM coverage only."
        )
    else:
        print("uniform comparison rows:", len(uniform_rows))

    make_figure(
        root,
        plot_state,
        final,
        history,
        uniform_rows,
        validation_fmax,
        outdir,
    )

    print("\nDONE")


if __name__ == "__main__":
    main()



