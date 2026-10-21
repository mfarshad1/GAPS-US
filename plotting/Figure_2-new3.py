#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication figures for the Müller–Brown GAPS-US benchmark.

Main text:
  Figure 2 — adaptive network, WHAM/GP reconstruction, error, and convergence.
Supporting Information:
  full reconstruction errors, efficiency/parity diagnostics, adaptive-K geometry,
  and overlap/connectivity diagnostics.

Latest changes requested by user:
- restore RMSE-vs-nodes to the earlier "visible / physically relevant" mask:
      common support AND exact F <= FMAX
- improve square landscape rendering and avoid poor-looking empty framing
- move Figure 3 panel-A legend outside the panel (top, row-wise)
- Figure 4 panel A uses number of windows (not cumulative propagated steps)
- Figure 4 y-axis uses just K
- RMSE legends use only "WHAM" and "GP"
- replace curvature panel by a more intuitive "K vs exact F"
- make the K-distribution panel prettier
"""

from __future__ import annotations
import argparse, json, math, os, re, sys
from pathlib import Path

import numpy as np
import matplotlib
if not os.environ.get("DISPLAY") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

BASE = Path("/users/mfarshad/LJ_ADAPTIVE_US")
DEFAULT_LATEST = BASE / "MULLER_BROWN_2D_GAPS_SPYDER_GRID55_ALLCELL_EXACTGP_LATEST"

MB_SCALE = 0.05
FMAX = 10.0
ERROR_LIMIT = None
ERROR_PERCENTILE = 99.0
ELLIPSE_CHI2 = 5.991
PNG_DPI = 450
DEFAULT_MINIMUM_OVERLAP = 0.10

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 24.0,
    "axes.labelsize": 26.25,
    "axes.titlesize": 26.25,
    "xtick.labelsize": 20.25,
    "ytick.labelsize": 20.25,
    "legend.fontsize": 18.0,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

C_WHAM = "#ff7f0e"   # WHAM: orange, matching the shared 3D diagnostics code
C_GP   = "#3B82F6"   # GP: richer blue, matching the shared 3D diagnostics code
C_SIG  = "#6A3D9A"
C_SUP  = "#1B9E77"
C_HIST = "#4878A8"
C_FOOT = "#FFFFFF"   # very light sampling-distribution ellipses
C_LINE = "#777777"

TYPE_STYLE = {
    # High-contrast overlay colors chosen to remain visible on the viridis landscape.
    "bootstrap_seed":      dict(marker="o", color="#D81B60", label="Bootstrap"),
    "bootstrap_bridge":    dict(marker="D", color="#8E44AD", label="Bootstrap bridge"),
    "connectivity_bridge": dict(marker="s", color="#E31A1C", label="Connectivity bridge"),
    "information_target":  dict(marker="^", color="#00D4FF", label="Information target"),
}

def read_json(path):
    return json.loads(Path(path).read_text())

def query_number(path):
    m = re.search(r"query_(\d+)", str(path))
    return int(m.group(1)) if m else None

def exact_mb_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    shp = x.shape
    A  = np.array([-200.0, -100.0, -170.0, 15.0])[:, None]
    a  = np.array([-1.0, -1.0, -6.5, 0.7])[:, None]
    b  = np.array([ 0.0,  0.0, 11.0, 0.6])[:, None]
    c  = np.array([-10.0,-10.0, -6.5, 0.7])[:, None]
    x0 = np.array([ 1.0,  0.0, -0.5,-1.0])[:, None]
    y0 = np.array([ 0.0,  0.5,  1.5, 1.0])[:, None]
    xx = x.ravel()[None, :]
    yy = y.ravel()[None, :]
    q = a*(xx-x0)**2 + b*(xx-x0)*(yy-y0) + c*(yy-y0)**2
    return (MB_SCALE*np.sum(A*np.exp(q), axis=0)).reshape(shp)

def global_exact_min():
    x = np.linspace(-2.0, 1.0, 700)
    y = np.linspace(-0.5, 2.0, 600)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return float(np.nanmin(exact_mb_xy(xx, yy)))

EXACT_MIN = global_exact_min()


def exact_rel(x, y):
    return exact_mb_xy(x, y) - EXACT_MIN

def exact_mb_local_geometry(x, y):
    """
    Analytic local geometry of the Müller–Brown surface.
    Returns gradient norm, Hessian Frobenius norm, Laplacian, and
    destabilizing curvature max(0, -lambda_min(H)).
    """
    x = float(x); y = float(y)

    A  = np.array([-200.0, -100.0, -170.0, 15.0], dtype=float)
    a  = np.array([-1.0, -1.0, -6.5, 0.7], dtype=float)
    b  = np.array([ 0.0,  0.0, 11.0, 0.6], dtype=float)
    c  = np.array([-10.0,-10.0, -6.5, 0.7], dtype=float)
    x0 = np.array([ 1.0,  0.0, -0.5,-1.0], dtype=float)
    y0 = np.array([ 0.0,  0.5,  1.5, 1.0], dtype=float)

    dx = x - x0
    dy = y - y0
    q = a*dx*dx + b*dx*dy + c*dy*dy
    e = np.exp(q)
    t = MB_SCALE * A * e

    qx = 2.0*a*dx + b*dy
    qy = b*dx + 2.0*c*dy

    fx  = np.sum(t * qx)
    fy  = np.sum(t * qy)
    fxx = np.sum(t * (qx*qx + 2.0*a))
    fyy = np.sum(t * (qy*qy + 2.0*c))
    fxy = np.sum(t * (qx*qy + b))

    H = np.array([[fxx, fxy],
                  [fxy, fyy]], dtype=float)

    grad_norm = float(np.hypot(fx, fy))
    hess_frob = float(np.sqrt(np.sum(H*H)))
    lap = float(np.trace(H))
    evals = np.linalg.eigvalsh(H)
    kappa_destab = float(max(0.0, -evals.min()))
    return grad_norm, hess_frob, lap, kappa_destab

def align_to_ref(ref, est, mask):
    ref = np.asarray(ref, float)
    est = np.asarray(est, float)
    good = np.asarray(mask, bool) & np.isfinite(ref) & np.isfinite(est)
    if not np.any(good):
        return est.copy(), 0.0
    shift = float(np.median(ref[good] - est[good]))
    return est + shift, shift

def panel(ax, s):
    ax.text(
        0.02, 0.98, s, transform=ax.transAxes,
        ha="left", va="top", fontsize=30, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.4),
        zorder=100
    )

def xylabels(ax):
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

def savefig(fig, stem, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("saved:", png)
    print("saved:", pdf)

def set_square_axes(ax):
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

def square_data_window_from_finite(x_edges, y_edges, z):
    Z = np.asarray(z, float)
    finite = np.isfinite(Z)
    xmin_all = float(np.min(x_edges))
    xmax_all = float(np.max(x_edges))
    ymin_all = float(np.min(y_edges))
    ymax_all = float(np.max(y_edges))
    if finite.ndim != 2 or not np.any(finite):
        xc = 0.5*(xmin_all + xmax_all)
        yc = 0.5*(ymin_all + ymax_all)
        side = max(xmax_all - xmin_all, ymax_all - ymin_all)
        return xc - 0.5*side, xc + 0.5*side, yc - 0.5*side, yc + 0.5*side
    row_has = np.any(finite, axis=1)
    col_has = np.any(finite, axis=0)
    iy = np.where(row_has)[0]
    ix = np.where(col_has)[0]
    xmin = float(x_edges[int(ix.min())])
    xmax = float(x_edges[int(ix.max()) + 1])
    ymin = float(y_edges[int(iy.min())])
    ymax = float(y_edges[int(iy.max()) + 1])
    xc = 0.5*(xmin + xmax)
    yc = 0.5*(ymin + ymax)
    side = max(xmax - xmin, ymax - ymin) * 1.01
    return xc - 0.5*side, xc + 0.5*side, yc - 0.5*side, yc + 0.5*side

def apply_square_data_window(ax, x_edges, y_edges, z):
    """
    Fill a square AXES FRAME with the finite landscape and remove external blank bands.

    Important: using aspect="equal" together with a square axes box forces extra
    whitespace whenever the x- and y-ranges are not exactly identical.  The older
    plotting style looked better because the landscape filled the axes.  Here we
    therefore keep the frame square (box_aspect=1) but use aspect="auto" so the
    finite data footprint fills that square frame.

    Any NaN/unsupported cells INSIDE the WHAM domain remain blank, as desired.
    """
    Z = np.asarray(z, float)
    finite = np.isfinite(Z)

    if finite.ndim == 2 and np.any(finite):
        iy = np.where(np.any(finite, axis=1))[0]
        ix = np.where(np.any(finite, axis=0))[0]

        xmin = float(x_edges[int(ix.min())])
        xmax = float(x_edges[int(ix.max()) + 1])
        ymin = float(y_edges[int(iy.min())])
        ymax = float(y_edges[int(iy.max()) + 1])
    else:
        xmin = float(np.min(x_edges))
        xmax = float(np.max(x_edges))
        ymin = float(np.min(y_edges))
        ymax = float(np.max(y_edges))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Square frame, landscape fills it completely.
    ax.set_aspect("auto")
    ax.set_box_aspect(1)
    ax.margins(x=0, y=0)

def resolve_root(user_root=None):
    if user_root:
        p = Path(user_root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    env = os.environ.get("MB_CAMPAIGN_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    tmp = Path("/tmp/mb_best_campaign.txt")
    if tmp.exists():
        t = tmp.read_text().strip()
        if t:
            p = Path(t).expanduser().resolve()
            if p.exists():
                return p
    if DEFAULT_LATEST.exists() or DEFAULT_LATEST.is_symlink():
        p = DEFAULT_LATEST.resolve()
        if p.exists() and list(p.rglob("online_analysis.npz")):
            return p
    cands = []
    for p in BASE.glob("MULLER_BROWN_2D_GAPS_SPYDER_*"):
        if p.is_dir():
            npzs = list(p.rglob("online_analysis.npz"))
            if npzs:
                cands.append((max(q.stat().st_mtime for q in npzs), p))
    if not cands:
        raise FileNotFoundError("No Müller–Brown campaign with online_analysis.npz found.")
    return max(cands, key=lambda t: t[0])[1]

def analysis_files(root):
    fs = list((root / "analysis_history").rglob("online_analysis.npz"))
    if not fs:
        fs = list(root.rglob("online_analysis.npz"))
    return sorted(fs, key=lambda p: ((query_number(p) if query_number(p) is not None else 10**9), p.stat().st_mtime))

def latest_analysis_file(root, state):
    p = state.get("latest_full_analysis")
    if p and Path(p).exists():
        return Path(p)
    fs = analysis_files(root)
    if not fs:
        raise FileNotFoundError(f"No online_analysis.npz under {root}")
    return max(fs, key=lambda p: p.stat().st_mtime)

def load_final(root, state):
    p = latest_analysis_file(root, state)
    z = dict(np.load(p, allow_pickle=False))
    req = ["x_centers", "y_centers", "x_edges", "y_edges", "wham_f", "support", "gp_base_mean_grid"]
    miss = [k for k in req if k not in z]
    if miss:
        raise KeyError(f"Final npz missing arrays: {miss}")
    x = np.asarray(z["x_centers"], float)
    y = np.asarray(z["y_centers"], float)
    xe = np.asarray(z["x_edges"], float)
    ye = np.asarray(z["y_edges"], float)
    wh = np.asarray(z["wham_f"], float)
    sup = np.asarray(z["support"], bool)
    gp0 = np.asarray(z["gp_base_mean_grid"], float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ex0 = exact_rel(xx, yy)
    common = sup & np.isfinite(wh) & np.isfinite(gp0) & np.isfinite(ex0)
    if not np.any(common):
        raise RuntimeError("No common support among exact / WHAM / GP.")
    wha, _ = align_to_ref(ex0, wh, common)
    gpa, gp_shift = align_to_ref(ex0, gp0, common)
    if all(k in z for k in ["gp_recon_x_edges", "gp_recon_y_edges", "gp_recon_x_centers", "gp_recon_y_centers", "gp_recon_mean_grid"]):
        gxe = np.asarray(z["gp_recon_x_edges"], float)
        gye = np.asarray(z["gp_recon_y_edges"], float)
        gxc = np.asarray(z["gp_recon_x_centers"], float)
        gyc = np.asarray(z["gp_recon_y_centers"], float)
        gp_hi = np.asarray(z["gp_recon_mean_grid"], float) + gp_shift
        gxx, gyy = np.meshgrid(gxc, gyc, indexing="xy")
        ex_hi = exact_rel(gxx, gyy)
    else:
        gxe, gye = xe, ye
        gp_hi = gpa.copy()
        ex_hi = ex0.copy()
    d_we = np.full_like(wha, np.nan)
    d_ge = np.full_like(gpa, np.nan)
    gp_to_wham, _ = align_to_ref(wha, gp0, common)
    d_gw = np.full_like(gpa, np.nan)
    d_we[common] = wha[common] - ex0[common]
    d_ge[common] = gpa[common] - ex0[common]
    d_gw[common] = gp_to_wham[common] - wha[common]
    def rmse(err, mask):
        return float(np.sqrt(np.mean(np.asarray(err)[mask]**2)))
    return dict(path=p, raw=z, x=x, y=y, xe=xe, ye=ye, gxe=gxe, gye=gye,
                exact_base=ex0, exact_hi=ex_hi, wham=wha, gp_base=gpa, gp_hi=gp_hi,
                support=sup, common=common, d_we=d_we, d_ge=d_ge, d_gw=d_gw,
                wrmse=rmse(d_we, common), grmse=rmse(d_ge, common))

def error_limit(*arrs):
    if ERROR_LIMIT is not None:
        return float(ERROR_LIMIT)
    vals = []
    for a in arrs:
        a = np.asarray(a, float)
        vals.extend(np.abs(a[np.isfinite(a)]).tolist())
    if not vals:
        return 0.5
    lim = max(float(np.percentile(vals, ERROR_PERCENTILE)), 0.05)
    p = 10**math.floor(math.log10(lim))
    s = lim / p
    nice = 1 if s <= 1 else 2 if s <= 2 else 5 if s <= 5 else 10
    return nice * p

def make_si_reconstruction_errors(d, outdir):
    lim = error_limit(d["d_we"], d["d_ge"], d["d_gw"])

    # Manual spacing gives tighter row separation than constrained_layout.
    fig, ax = plt.subplots(2, 3, figsize=(17.6, 11.8))
    fig.subplots_adjust(left=0.070, right=0.900, bottom=0.090, top=0.93,
                        wspace=0.25, hspace=0.36)

    ekw = dict(shading="auto", vmin=0, vmax=FMAX, cmap="viridis", rasterized=True)

    # Row 1: free-energy landscapes
    im = ax[0,0].pcolormesh(d["gxe"], d["gye"], np.clip(d["exact_hi"], 0, FMAX), **ekw)
    ax[0,0].set_title(r"$F^{\mathrm{Ref}}$", pad=12)
    xylabels(ax[0,0]); ax[0,0].xaxis.labelpad = 1; apply_square_data_window(ax[0,0], d["gxe"], d["gye"], d["exact_hi"]); panel(ax[0,0], "A")

    wp = np.where(d["support"], np.clip(d["wham"], 0, FMAX), np.nan)
    ax[0,1].pcolormesh(d["xe"], d["ye"], wp, **ekw)
    ax[0,1].set_title(r"$F^{\mathrm{WHAM}}$", pad=12)
    xylabels(ax[0,1]); ax[0,1].xaxis.labelpad = 1; apply_square_data_window(ax[0,1], d["xe"], d["ye"], wp); panel(ax[0,1], "B")

    ax[0,2].pcolormesh(d["gxe"], d["gye"], np.clip(d["gp_hi"], 0, FMAX), **ekw)
    ax[0,2].set_title(r"$F^{\mathrm{GP}}$", pad=12)
    xylabels(ax[0,2]); ax[0,2].xaxis.labelpad = 1; apply_square_data_window(ax[0,2], d["gxe"], d["gye"], d["gp_hi"]); panel(ax[0,2], "C")

    # Row 2: difference maps
    dkw = dict(shading="auto", cmap="coolwarm", vmin=-lim, vmax=lim, rasterized=True)

    imd = ax[1,0].pcolormesh(d["xe"], d["ye"], d["d_we"], **dkw)
    ax[1,0].set_title(r"$F^{\mathrm{WHAM}} - F^{\mathrm{Ref}}$", pad=12)
    xylabels(ax[1,0]); apply_square_data_window(ax[1,0], d["xe"], d["ye"], d["d_we"]); panel(ax[1,0], "D")

    ax[1,1].pcolormesh(d["xe"], d["ye"], d["d_ge"], **dkw)
    ax[1,1].set_title(r"$F^{\mathrm{GP}} - F^{\mathrm{Ref}}$", pad=12)
    xylabels(ax[1,1]); apply_square_data_window(ax[1,1], d["xe"], d["ye"], d["d_ge"]); panel(ax[1,1], "E")

    ax[1,2].pcolormesh(d["xe"], d["ye"], d["d_gw"], **dkw)
    ax[1,2].set_title(r"$F^{\mathrm{GP}} - F^{\mathrm{WHAM}}$", pad=12)
    xylabels(ax[1,2]); apply_square_data_window(ax[1,2], d["xe"], d["ye"], d["d_gw"]); panel(ax[1,2], "F")

    # Draw once so that panel positions are finalized, then add colorbars with
    # EXACTLY the same vertical span as the panel data area.
    fig.canvas.draw()

    # Show y-axis labels only on the first column
    for r in range(2):
        ax[r, 1].set_ylabel("")
        ax[r, 2].set_ylabel("")
        
    bbox_top = ax[0,2].get_position()
    cax_top = fig.add_axes([bbox_top.x1 + 0.012, bbox_top.y0, 0.015, bbox_top.height])
    cb = fig.colorbar(im, cax=cax_top)
    cb.set_label(r"$F$ ($k_{\mathrm{B}}T$)")

    bbox_bot = ax[1,2].get_position()
    cax_bot = fig.add_axes([bbox_bot.x1 + 0.012, bbox_bot.y0, 0.015, bbox_bot.height])
    cb = fig.colorbar(imd, cax=cax_bot)
    cb.set_label(r"$\Delta F$ ($k_{\mathrm{B}}T$)")

    savefig(fig, "Figure_3_Muller_Brown_Reconstruction_Differences", outdir)
    show_or_close(fig)

def wdir(root, wid):
    return root / "windows" / f"window_{int(wid):06d}"

def read_xy(path):
    a = np.loadtxt(path, comments="#", ndmin=2)
    if a.shape[1] < 3:
        raise ValueError(path)
    return np.asarray(a[:,1:3], float)

def prod_samples(root, wid):
    chunks = []
    for p in sorted(wdir(root, wid).glob("rep_*/samples.dat")):
        try:
            a = read_xy(p)
            if len(a):
                chunks.append(a)
        except Exception:
            pass
    return np.vstack(chunks) if chunks else None

def ellipse_stats(xy):
    if xy is None or len(xy) < 5:
        return None
    cov = np.cov(np.asarray(xy).T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1.0e-14)
    o = np.argsort(vals)[::-1]
    vals = vals[o]
    vecs = vecs[:,o]
    width  = 2*np.sqrt(ELLIPSE_CHI2*vals[0])
    height = 2*np.sqrt(ELLIPSE_CHI2*vals[1])
    angle  = float(np.degrees(np.arctan2(vecs[1,0], vecs[0,0])))
    area   = float(np.pi*ELLIPSE_CHI2*np.sqrt(vals[0]*vals[1]))
    return width, height, angle, area

def spring_info(root, w):
    ad = w.get("adaptive_spring")
    if not isinstance(ad, dict):
        p = wdir(root, w["id"]) / "adaptive_spring.json"
        ad = read_json(p) if p.exists() else {}
    k = w.get("spring_k", ad.get("chosen_K", np.nan))
    try:
        k = float(k)
    except Exception:
        k = np.nan
    attempts = ad.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    return k, attempts

def exact_f_at_center(center):
    return float(exact_mb_xy(np.array([center[0]]), np.array([center[1]]))[0] - EXACT_MIN)

def collect_windows(root, state):
    rows = []
    ws = sorted([w for w in state.get("windows", []) if w.get("status") == "complete"], key=lambda w: int(w["id"]))
    for idx, w in enumerate(ws, start=1):
        wid = int(w["id"])
        center = np.asarray(w["center"], float)
        k, attempts = spring_info(root, w)
        xy = prod_samples(root, wid)
        ell = ellipse_stats(xy)
        area = np.nan if ell is None else ell[3]
        grad_norm, hess_frob, lap, kappa_destab = exact_mb_local_geometry(center[0], center[1])
        rows.append(dict(index=idx, id=wid, kind=w.get("kind", "window"), center=center,
                         k=k, attempts=len(attempts), samples=xy, ellipse=ell,
                         footprint=area, exact_f=exact_f_at_center(center),
                         grad_norm=grad_norm, hess_frob=hess_frob,
                         laplacian=lap, kappa_destab=kappa_destab))
    return rows

def overlap_cache(root):
    p = root / "analysis_current" / "measured_overlap_cache.json"
    if p.exists():
        return p
    fs = list(root.rglob("measured_overlap_cache.json"))
    return max(fs, key=lambda q: q.stat().st_mtime) if fs else None

def edges(root, state, ids):
    ids = set(map(int, ids))
    p = overlap_cache(root)
    if p is None:
        return []
    thr = float(state.get("minimum_overlap", DEFAULT_MINIMUM_OVERLAP))
    out = []
    for key, rec in read_json(p).items():
        try:
            i, j = map(int, key.split(":"))
            ov = float(rec.get("overlap", 0))
        except Exception:
            continue
        if i in ids and j in ids and ov >= thr:
            out.append((i, j, ov))
    return out

def snapshot(path):
    z = dict(np.load(path, allow_pickle=False))
    need = ["x_centers", "y_centers", "wham_f", "support", "gp_base_mean_grid"]
    if any(k not in z for k in need):
        return None
    x = np.asarray(z["x_centers"], float)
    y = np.asarray(z["y_centers"], float)
    wh = np.asarray(z["wham_f"], float)
    gp = np.asarray(z["gp_base_mean_grid"], float)
    sup = np.asarray(z["support"], bool)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ex = exact_rel(xx, yy)
    com = sup & np.isfinite(wh) & np.isfinite(gp) & np.isfinite(ex)
    if not np.any(com):
        return None
    wa, _ = align_to_ref(ex, wh, com)
    ga, _ = align_to_ref(ex, gp, com)
    visible = com & (ex <= FMAX)
    mask = visible if np.any(visible) else com
    we = wa[mask] - ex[mask]
    ge = ga[mask] - ex[mask]
    ids = np.asarray(z.get("connected_window_ids", []), int).ravel()
    nwin = len(ids) if len(ids) else len(np.asarray(z.get("umbrella_centers", [])))
    sigma = np.nan
    s = path.parent / "analysis_summary.json"
    if s.exists():
        try:
            sigma = float((read_json(s).get("query") or {}).get("sigma_max", np.nan))
        except Exception:
            pass
    return dict(query=query_number(path), nwin=int(nwin),
                wrmse=float(np.sqrt(np.mean(we**2))), grmse=float(np.sqrt(np.mean(ge**2))),
                sigma=sigma, support_fraction=float(np.mean(sup)), path=path)

def collect_history(root):
    rows = []
    for p in analysis_files(root):
        try:
            r = snapshot(p)
            if r:
                rows.append(r)
        except Exception as e:
            print("WARNING history", p, e)
    d = {}
    for r in rows:
        d[r["query"] if r["query"] is not None else str(r["path"])] = r
    return sorted(d.values(), key=lambda r: r["query"] if r["query"] is not None else 10**9)

def network_handles():
    handles = []
    labels = []
    for key in ["bootstrap_seed", "information_target", "connectivity_bridge"]:
        st = TYPE_STYLE[key]
        handles.append(Line2D([0],[0], marker=st["marker"], linestyle="None",
                              markerfacecolor=st["color"], markeredgecolor="black",
                              markeredgewidth=0.6, markersize=8.5))
        labels.append(st["label"])
    return handles, labels

def draw_network(ax, root, state, d, wrows, background="exact", add_panel=True, panel_tag="A"):
    ids = np.asarray(d["raw"].get("connected_window_ids", []), int).ravel().tolist()
    if not ids:
        ids = [r["id"] for r in wrows]
    idset = set(ids)
    byid = {r["id"]: r for r in wrows}

    if str(background).lower() == "gp":
        bgZ = np.clip(d["gp_hi"], 0, FMAX)
    else:
        bgZ = np.clip(d["exact_hi"], 0, FMAX)

    bg = ax.pcolormesh(d["gxe"], d["gye"], bgZ,
                       shading="auto", cmap="viridis", vmin=0, vmax=FMAX, rasterized=True)
    for r in wrows:
        if r["id"] in idset and r["ellipse"] is not None:
            W, H, A, _ = r["ellipse"]
            ax.add_patch(Ellipse(r["center"], W, H, angle=A, fill=False,
                                 edgecolor=C_FOOT, lw=1.00, alpha=0.48, zorder=4))
    for i, j, ov in edges(root, state, ids):
        if i in byid and j in byid:
            ci = byid[i]["center"]; cj = byid[j]["center"]
            ax.plot([ci[0], cj[0]], [ci[1], cj[1]], color="black", lw=0.72, alpha=0.33, zorder=5)
    for r in wrows:
        if r["id"] not in idset:
            continue
        st = TYPE_STYLE.get(r["kind"], dict(marker="o", color="0.65", label=r["kind"]))
        ax.scatter([r["center"][0]], [r["center"][1]], s=50, marker=st["marker"],
                   c=st["color"], edgecolors="none", linewidths=0.0, zorder=8)
    xylabels(ax)
    apply_square_data_window(ax, d["gxe"], d["gye"], bgZ)
    if add_panel:
        panel(ax, panel_tag)
    return bg

def make_main_figure2(root, state, d, hist, wrows, outdir):
    if not hist:
        raise RuntimeError("No history snapshots found.")

    q = np.asarray([np.nan if r["query"] is None else r["query"] for r in hist], float)
    n = np.asarray([r["nwin"] for r in hist], float)
    wr = np.asarray([r["wrmse"] for r in hist], float)
    gr = np.asarray([r["grmse"] for r in hist], float)
    sig = np.asarray([r["sigma"] for r in hist], float)
    sup = 100*np.asarray([r["support_fraction"] for r in hist], float)

    fig, ax = plt.subplots(2, 2, figsize=(13.9, 12.4))
    fig.subplots_adjust(left=0.085, right=0.955, bottom=0.090, top=0.91,
                        wspace=0.30, hspace=0.24)

    bg = draw_network(ax[0,0], root, state, d, wrows, background="gp", add_panel=True, panel_tag="A")

    # Do not let the colorbar shrink panel A.  Instead, place a dedicated
    # colorbar axis in the gap between columns so A aligns with C below.
    fig.canvas.draw()
    bbox_a = ax[0,0].get_position()
    cax_a = fig.add_axes([
        bbox_a.x1 + 0.007,
        bbox_a.y0,
        0.011,
        bbox_a.height
    ])
    cb = fig.colorbar(bg, cax=cax_a)
    cb.set_label(r"$F^{\mathrm{GP}}$ ($k_{\mathrm{B}}T$)")

    good = np.isfinite(q) & np.isfinite(sig)
    l1, = ax[0,1].plot(q[good], sig[good], marker="o", ms=9.2, lw=2.4, color=C_SIG, markeredgecolor="none", markeredgewidth=0, label="Max GP uncertainty")
    ax[0,1].set_xlabel("AL step")
    ax[0,1].set_ylabel(r"Max GP uncertainty ($k_{\mathrm{B}}T$)", color=C_SIG)
    ax[0,1].tick_params(axis="y", labelcolor=C_SIG)

    a2 = ax[0,1].twinx()
    l2, = a2.plot(q, sup, marker="s", ms=8.8, lw=2.3, color=C_SUP, markeredgecolor="none", markeredgewidth=0, label="WHAM-supported domain")
    a2.set_ylabel("WHAM-supported grid (%)", color=C_SUP)
    a2.tick_params(axis="y", labelcolor=C_SUP)
    a2.set_ylim(0, 102)

    set_square_axes(ax[0,1]); panel(ax[0,1], "B")

    common = d["common"]
    ex = d["exact_base"][common]
    wa = d["wham"][common]
    ga = d["gp_base"][common]
    if len(ex) > 3000:
        sel = np.linspace(0, len(ex)-1, 3000).astype(int)
        ex, wa, ga = ex[sel], wa[sel], ga[sel]

    ax[1,0].scatter(ex, wa, s=28, alpha=0.28, color=C_WHAM, edgecolors="none")
    ax[1,0].scatter(ex, ga, s=28, alpha=0.28, color=C_GP, edgecolors="none")
    hi = min(FMAX, max(np.nanmax(ex), np.nanmax(wa), np.nanmax(ga)))
    ax[1,0].plot([0, hi], [0, hi], ls="--", lw=1.4, color="black")
    ax[1,0].set_xlim(0, hi); ax[1,0].set_ylim(0, hi)
    ax[1,0].set_aspect("equal", adjustable="box"); ax[1,0].set_box_aspect(1)
    ax[1,0].set_xlabel(r"$F^{\mathrm{Ref}}$ ($k_{\mathrm{B}}T$)")
    ax[1,0].set_ylabel(r"$F^{\mathrm{GP}}, F^{\mathrm{WHAM}}$ ($k_{\mathrm{B}}T$)")
    leg_handles = [
        Line2D([0], [0], marker='o', linestyle='None', color='none',
               markerfacecolor=C_WHAM, markeredgecolor='none', markersize=17.0, label='WHAM'),
        Line2D([0], [0], marker='o', linestyle='None', color='none',
               markerfacecolor=C_GP, markeredgecolor='none', markersize=17.0, label='GP'),
    ]
    ax[1,0].legend(handles=leg_handles, loc="lower right", frameon=False)
    panel(ax[1,0], "C")

    ax[1,1].plot(n, wr, marker="o", ms=9.6, lw=2.5, color=C_WHAM, markeredgecolor="none", markeredgewidth=0, label="WHAM")
    ax[1,1].plot(n, gr, marker="s", ms=9.6, lw=2.5, color=C_GP, markeredgecolor="none", markeredgewidth=0, label="GP")
    ax[1,1].set_xlabel("Number of windows")
    ax[1,1].set_ylabel(r"RMSE ($k_{\mathrm{B}}T$)")
    ax[1,1].legend(loc="upper right", frameon=False)
    set_square_axes(ax[1,1]); panel(ax[1,1], "D")

    savefig(fig, "Figure_2_Muller_Brown_GAPS_US", outdir)
    show_or_close(fig)

def rho_text(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 4 or spearmanr is None:
        return r"$\rho_s$ = n/a"
    return rf"$\rho_s={spearmanr(x[good], y[good]).statistic:.2f}$"

def scatter_by_kind(ax, x, y, kinds):
    x = np.asarray(x, float); y = np.asarray(y, float); kinds = np.asarray(kinds, object)
    seen = set()
    for kind in np.unique(kinds):
        m = (kinds == kind) & np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue
        st = TYPE_STYLE.get(kind, dict(marker="o", color="0.55", label=str(kind)))
        lab = st["label"] if st["label"] not in seen else None
        seen.add(st["label"])
        ax.scatter(x[m], y[m], s=42, marker=st["marker"], c=st["color"], edgecolors="black",
                   linewidths=0.35, alpha=0.85, label=lab)

def make_si_adaptive_k_geometry(rows, outdir):
    v = [r for r in rows if np.isfinite(r["k"])]
    if not v:
        print("WARNING: no adaptive K values found; skipping Figure 4")
        return

    kinds  = np.asarray([r["kind"] for r in v], object)
    K      = np.asarray([r["k"] for r in v], float)
    idx    = np.asarray([r["index"] for r in v], float)
    area   = np.asarray([r["footprint"] for r in v], float)
    grad   = np.asarray([r.get("grad_norm", np.nan) for r in v], float)

    fig, ax = plt.subplots(2, 2, figsize=(12.4, 10.9))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.08, top=0.92,
                        wspace=0.18, hspace=0.25)

    ax[0,0].plot(idx, K, color=C_LINE, lw=1.0, alpha=0.72, zorder=1)
    scatter_by_kind(ax[0,0], idx, K, kinds)
    ax[0,0].set_xlabel("Number of windows")
    ax[0,0].set_ylabel(r"$K$")
    n_total = len(rows)
    x_hi = max(20, int(np.ceil(n_total / 20.0) * 20))
    ax[0,0].set_xlim(0, x_hi)
    ax[0,0].set_xticks(np.arange(0, x_hi + 1, 50 if x_hi >= 150 else 20))
    set_square_axes(ax[0,0]); panel(ax[0,0], "A")

    lev, cnt = np.unique(K[np.isfinite(K)], return_counts=True)
    ax[0,1].vlines(lev, 0, cnt, color=C_HIST, lw=3.0, alpha=0.85)
    ax[0,1].scatter(lev, cnt, s=105, color=C_HIST,
                    edgecolors="white", linewidths=1.1, zorder=3)
    ax[0,1].set_xscale("log", base=2)
    ax[0,1].set_xticks(lev)
    ax[0,1].set_xticklabels(
        [f"{k:g}" for k in lev],
        rotation=30,
        ha="right",
        rotation_mode="anchor"
    )
    ax[0,1].tick_params(axis="x", pad=4)
    ax[0,1].set_xlim(lev.min()/np.sqrt(2.0), lev.max()*np.sqrt(2.0))
    ax[0,1].set_xlabel(r"$K$")
    ax[0,1].set_ylabel("Number of windows")
    set_square_axes(ax[0,1]); panel(ax[0,1], "B")

    scatter_by_kind(ax[1,0], grad, K, kinds)
    ax[1,0].set_xlabel(r"$|\nabla F|$")
    ax[1,0].set_ylabel(r"$K$")
    set_square_axes(ax[1,0]); panel(ax[1,0], "C")

    scatter_by_kind(ax[1,1], K, area, kinds)
    ax[1,1].set_xscale("log", base=2)
    ax[1,1].set_xticks(lev)
    ax[1,1].set_xticklabels(
        [f"{k:g}" for k in lev],
        rotation=30,
        ha="right",
        rotation_mode="anchor"
    )
    ax[1,1].tick_params(axis="x", pad=4)
    ax[1,1].set_xlim(lev.min()/np.sqrt(2.0), lev.max()*np.sqrt(2.0))
    ax[1,1].set_xlabel(r"$K$")
    ax[1,1].set_ylabel("Footprint area")
    set_square_axes(ax[1,1]); panel(ax[1,1], "D")

    handles, labels = network_handles()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 0.985))
    savefig(fig, "Figure-SI-MullerBrown-Adaptive-K-Geometry", outdir)
    show_or_close(fig)


def make_si_main_story_2x3(root, state, d, hist, wrows, outdir):
    """Alternate 2x3 Müller–Brown story retained as a supporting-figure option."""
    if not hist:
        raise RuntimeError("No history snapshots found.")

    q = np.asarray([np.nan if r["query"] is None else r["query"] for r in hist], float)
    n = np.asarray([r["nwin"] for r in hist], float)
    wr = np.asarray([r["wrmse"] for r in hist], float)
    gr = np.asarray([r["grmse"] for r in hist], float)
    sig = np.asarray([r["sigma"] for r in hist], float)
    sup = 100.0*np.asarray([r["support_fraction"] for r in hist], float)

    fig, ax = plt.subplots(2, 3, figsize=(17.8, 11.8))
    fig.subplots_adjust(left=0.070, right=0.910, bottom=0.090, top=0.91,
                        wspace=0.33, hspace=0.30)

    # A — exact landscape + final adaptive sampling graph.
    bg = draw_network(ax[0,0], root, state, d, wrows)
    ax[0,0].set_title(r"$F_{\mathrm{exact}}$ + adaptive network", pad=10)

    # B — direct WHAM reconstruction.
    wp = np.where(d["support"], np.clip(d["wham"], 0, FMAX), np.nan)
    ax[0,1].pcolormesh(d["xe"], d["ye"], wp, shading="auto", cmap="viridis",
                       vmin=0, vmax=FMAX, rasterized=True)
    ax[0,1].set_title(r"$F^{\mathrm{WHAM}}$", pad=10)
    xylabels(ax[0,1]); apply_square_data_window(ax[0,1], d["xe"], d["ye"], wp)
    panel(ax[0,1], "B")

    # C — GP reconstruction.
    ax[0,2].pcolormesh(d["gxe"], d["gye"], np.clip(d["gp_hi"], 0, FMAX),
                       shading="auto", cmap="viridis", vmin=0, vmax=FMAX,
                       rasterized=True)
    ax[0,2].set_title(r"$F^{\mathrm{GP}}$", pad=10)
    xylabels(ax[0,2]); apply_square_data_window(ax[0,2], d["gxe"], d["gye"], d["gp_hi"])
    panel(ax[0,2], "C")

    # D — the most useful single reconstruction-error map.
    lim = error_limit(d["d_ge"])
    imerr = ax[1,0].pcolormesh(d["xe"], d["ye"], d["d_ge"], shading="auto",
                               cmap="coolwarm", vmin=-lim, vmax=lim,
                               rasterized=True)
    ax[1,0].set_title(r"$F_{\mathrm{GP}}-F_{\mathrm{exact}}$", pad=10)
    xylabels(ax[1,0]); apply_square_data_window(ax[1,0], d["xe"], d["ye"], d["d_ge"])
    panel(ax[1,0], "D")

    # E — quantitative convergence with physical-window cost.
    ax[1,1].plot(n, wr, marker="o", ms=4.8, lw=2.3, color=C_WHAM, label="WHAM")
    ax[1,1].plot(n, gr, marker="s", ms=4.8, lw=2.3, color=C_GP, label="GP")
    ax[1,1].set_xlabel("Number of windows")
    ax[1,1].set_ylabel(r"RMSE ($k_{\mathrm{B}}T$)")
    ax[1,1].legend(loc="upper right", frameon=False)
    set_square_axes(ax[1,1]); panel(ax[1,1], "E")

    # F — acquisition uncertainty decreases while the physically connected
    # WHAM-supported domain grows.
    good = np.isfinite(q) & np.isfinite(sig)
    l1, = ax[1,2].plot(q[good], sig[good], marker="o", ms=4.6, lw=2.2,
                       color=C_SIG, label="Max GP uncertainty")
    ax[1,2].set_xlabel("AL step")
    ax[1,2].set_ylabel(r"Max GP uncertainty / $k_{\mathrm{B}}T$", color=C_SIG)
    ax[1,2].tick_params(axis="y", labelcolor=C_SIG)
    a2 = ax[1,2].twinx()
    l2, = a2.plot(q, sup, marker="s", ms=4.4, lw=2.1,
                  color=C_SUP, label="WHAM-supported domain")
    a2.set_ylabel("WHAM-supported grid / %", color=C_SUP)
    a2.tick_params(axis="y", labelcolor=C_SUP)
    a2.set_ylim(0, 102)
    ax[1,2].legend([l1, l2], [l1.get_label(), l2.get_label()],
                   loc="center right", frameon=False)
    set_square_axes(ax[1,2]); panel(ax[1,2], "F")

    handles, labels = network_handles()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.065, 0.988),
               ncol=3, frameon=False, columnspacing=1.1, handletextpad=0.45)

    # Shared energy and error colorbars without changing panel sizes.
    fig.canvas.draw()
    bc = ax[0,2].get_position()
    cax = fig.add_axes([bc.x1 + 0.010, bc.y0, 0.012, bc.height])
    cb = fig.colorbar(bg, cax=cax)
    cb.set_label(r"$F$ ($k_{\mathrm{B}}T$)")

    bd = ax[1,0].get_position()
    cax2 = fig.add_axes([bd.x1 + 0.008, bd.y0, 0.011, bd.height])
    cb2 = fig.colorbar(imerr, cax=cax2)
    cb2.set_label(r"$\Delta F$ ($k_{\mathrm{B}}T$)")

    savefig(fig, "Figure-SI-MullerBrown-Story-2x3", outdir)
    show_or_close(fig)


def parent_overlap_records(root, state, wrows):
    """Return child-window K and its measured overlap with its parent window."""
    p = overlap_cache(root)
    if p is None:
        return []
    cache = read_json(p)
    row_by_id = {int(r["id"]): r for r in wrows}
    out = []
    for w in state.get("windows", []):
        if w.get("status") != "complete" or w.get("parent_window") is None:
            continue
        wid = int(w["id"]); parent = int(w["parent_window"])
        key = f"{min(wid,parent)}:{max(wid,parent)}"
        rec = cache.get(key)
        if not isinstance(rec, dict):
            continue
        try:
            ov = float(rec.get("overlap", np.nan))
        except Exception:
            continue
        r = row_by_id.get(wid)
        if r is None or not np.isfinite(ov):
            continue
        out.append(dict(index=r["index"], id=wid, parent=parent,
                        kind=r["kind"], k=r["k"], overlap=ov))
    return out


def make_si_overlap_connectivity(root, state, hist, wrows, outdir):
    """SI diagnostic linking adaptive K, measured overlap, and graph-building cost."""
    recs = parent_overlap_records(root, state, wrows)
    if not recs:
        print("WARNING: no measured parent overlaps; skipping overlap/connectivity SI figure")
        return

    kinds = np.asarray([r["kind"] for r in recs], object)
    idx = np.asarray([r["index"] for r in recs], float)
    K = np.asarray([r["k"] for r in recs], float)
    ov = np.asarray([r["overlap"] for r in recs], float)
    thr = float(state.get("minimum_overlap", DEFAULT_MINIMUM_OVERLAP))

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.8))
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.085, top=0.90,
                        wspace=0.27, hspace=0.27)

    # A — how the adaptive spring constant changes across physical windows.
    goodk = np.isfinite(K) & (K > 0)
    if np.any(goodk):
        ax[0,0].plot(idx[goodk], K[goodk], color=C_LINE, lw=1.0, alpha=0.65)
        scatter_by_kind(ax[0,0], idx, K, kinds)
        ax[0,0].set_yscale("log", base=2)
    ax[0,0].set_xlabel("Number of windows")
    ax[0,0].set_ylabel(r"$K$")
    set_square_axes(ax[0,0]); panel(ax[0,0], "A")

    # B — measured parent/child overlap and the graph-connectivity threshold.
    scatter_by_kind(ax[0,1], idx, ov, kinds)
    ax[0,1].axhline(thr, ls="--", lw=1.5, color="black", label="Connectivity threshold")
    frac = 100.0*np.mean(ov >= thr) if len(ov) else np.nan
    ax[0,1].text(0.04, 0.95, f"{frac:.0f}% ≥ threshold", transform=ax[0,1].transAxes,
                 ha="left", va="top")
    ax[0,1].set_xlabel("Number of windows")
    ax[0,1].set_ylabel("Measured parent overlap")
    ax[0,1].set_ylim(bottom=0)
    ax[0,1].legend(loc="lower right", frameon=False)
    set_square_axes(ax[0,1]); panel(ax[0,1], "B")

    # C — direct test that adaptive K keeps overlap in a connected regime.
    scatter_by_kind(ax[1,0], K, ov, kinds)
    ax[1,0].axhline(thr, ls="--", lw=1.5, color="black")
    if np.any(goodk):
        ax[1,0].set_xscale("log", base=2)
    ax[1,0].set_xlabel(r"$K$")
    ax[1,0].set_ylabel("Measured parent overlap")
    ax[1,0].set_ylim(bottom=0)
    set_square_axes(ax[1,0]); panel(ax[1,0], "C")

    # D — graph-building overhead: some AL targets require bridge windows.
    q = np.asarray([np.nan if r["query"] is None else r["query"] for r in hist], float)
    n = np.asarray([r["nwin"] for r in hist], float)
    good = np.isfinite(q) & np.isfinite(n)
    q = q[good]; n = n[good]
    order = np.argsort(q); q = q[order]; n = n[order]
    if len(q) >= 2:
        dq = np.diff(q)
        dn = np.diff(n)
        per_step = np.divide(dn, dq, out=np.full_like(dn, np.nan), where=dq > 0)
        qmid = q[1:]
        ax[1,1].bar(qmid, per_step, width=0.72, alpha=0.78, color=C_HIST)
        ax[1,1].axhline(1.0, ls="--", lw=1.5, color="black", label="One window / target")
        ax[1,1].legend(frameon=False, loc="upper right")
    ax[1,1].set_xlabel("AL step")
    ax[1,1].set_ylabel("Windows added per AL step")
    ax[1,1].set_ylim(bottom=0)
    set_square_axes(ax[1,1]); panel(ax[1,1], "D")

    handles, labels = network_handles()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    savefig(fig, "Figure-SI-MullerBrown-Overlap-Connectivity", outdir)
    show_or_close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Exact production campaign root")
    ap.add_argument("--outdir", default=".", help="Output directory")
    a = ap.parse_args()
    root = resolve_root(a.root)
    out  = Path(a.outdir).expanduser().resolve()
    sp = root / "campaign_state.json"
    if not sp.exists():
        raise FileNotFoundError(sp)
    state = read_json(sp)
    print("=" * 90)
    print("MÜLLER–BROWN PUBLICATION FIGURES 2–3 + SI (REVISED V4)")
    print("=" * 90)
    print("campaign root:", root)
    print("output dir   :", out)
    d = load_final(root, state)
    hist = collect_history(root)
    rows = collect_windows(root, state)
    print("final analysis :", d["path"])
    print(f"final WHAM full-support RMSE = {d['wrmse']:.6f} kBT")
    print(f"final GP   full-support RMSE = {d['grmse']:.6f} kBT")
    print("history snapshots:", len(hist))
    print("completed windows:", len(rows))
    print("\nMain-text Figure 2 ...")
    make_main_figure2(root, state, d, hist, rows, out)

    print("\nMain-text Figure 3 ...")
    make_si_reconstruction_errors(d, out)
    print("\nSupporting Information — alternate 2x3 story figure ...")
    make_si_main_story_2x3(root, state, d, hist, rows, out)
    print("\nSupporting Information — overlap/connectivity diagnostics ...")
    make_si_overlap_connectivity(root, state, hist, rows, out)
    print("\nSupporting Information — adaptive-K geometry ...")
    make_si_adaptive_k_geometry(rows, out)
    print("\nDONE")

if __name__ == "__main__":
    main()

