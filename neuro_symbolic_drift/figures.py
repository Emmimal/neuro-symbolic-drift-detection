"""
figures.py
----------
All article figures for Article 3.
Called by app.py --step figures after results are saved.

    generate_all_figures(results)   ← produces all 5 figures

Figures produced:
    fig1_detection_lag_{drift_type}.png      per-drift lag chart (money chart)
    fig2_all_drift_types_comparison.png      3-panel RWSS vs F1
    fig3_fidi_heatmap_concept.png            feature drift heatmap
    fig4_v14_weight_collapse.png             V14 weight evolution
    fig5_alert_timeline_grid.png             seeds × drift types grid
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch
from pathlib            import Path
from typing             import List

from .drift_metrics  import RWSS_ALERT_THRESHOLD
from .experiment     import SeedDriftResult, DRIFT_TYPES

FIGURES_DIR  = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────── #
RWSS_COL  = "#3B6D11"   # deep green
F1_COL    = "#D85A30"   # coral
PSI_COL   = "#888780"   # gray
LAG_FILL  = "#EAF3DE"   # light green fill for lag region
BG_COL    = "#F8F7F4"   # off-white panel bg

DRIFT_COL = {
    "covariate": "#185FA5",  # blue
    "prior":     "#BA7517",  # amber
    "concept":   "#D85A30",  # coral
}
W_LABELS = ["W0\n(base)", "W1", "W2", "W3", "W4", "W5", "W6", "W7"]


def _style(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(BG_COL)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#D3D1C7")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="#5F5E5A", labelsize=10)
    if title:
        ax.set_title(title,  fontsize=11, fontweight="normal",
                     pad=10, color="#2C2C2A")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#5F5E5A")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#5F5E5A")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 1 — Detection lag chart  (the article's money chart)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_detection_lag(results: List[SeedDriftResult],
                       drift_type: str = "concept") -> Path:
    """
    For each seed, plot RWSS and F1 across windows.
    Shade the detection lag region between the two alert lines.
    Mean line shown in bold; individual seeds in light.
    """
    dt_results = [r for r in results if r.drift_type == drift_type]
    if not dt_results:
        print(f"  [fig1] No results for drift_type={drift_type}")
        return None

    n_wins = max(len(r.windows) for r in dt_results)
    w_nums = list(range(n_wins))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    _style(ax,
           title=f"RWSS signals drift before F1 drops — {drift_type} drift",
           ylabel="Score",
           xlabel="Time window")

    # Individual seed lines (light)
    for r in dt_results:
        rwss_vals = [w.rwss for w in r.windows]
        f1_vals   = [w.f1   for w in r.windows]
        ax.plot(w_nums[:len(rwss_vals)], rwss_vals,
                color=RWSS_COL, alpha=0.18, linewidth=1.2)
        ax.plot(w_nums[:len(f1_vals)],   f1_vals,
                color=F1_COL,   alpha=0.18, linewidth=1.2, linestyle="--")

    # Mean lines (bold)
    rwss_mat = np.array([[w.rwss for w in r.windows] for r in dt_results])
    f1_mat   = np.array([[w.f1   for w in r.windows] for r in dt_results])
    ax.plot(w_nums, rwss_mat.mean(0), color=RWSS_COL, linewidth=2.8,
            marker="o", markersize=7, zorder=5, label="RWSS (mean)")
    ax.plot(w_nums, f1_mat.mean(0),   color=F1_COL,   linewidth=2.8,
            marker="s", markersize=7, zorder=5, linestyle="--", label="F1 (mean)")

    # Threshold lines
    ax.axhline(RWSS_ALERT_THRESHOLD, color=RWSS_COL, linewidth=0.9,
               linestyle=":", alpha=0.7,
               label=f"RWSS threshold ({RWSS_ALERT_THRESHOLD})")
    mean_base_f1 = f1_mat[:, 0].mean()
    ax.axhline(mean_base_f1 - 0.03, color=F1_COL, linewidth=0.9,
               linestyle=":", alpha=0.7, label="F1 threshold (baseline − 0.03)")

    # Shade lag region
    rwss_alerts = [r.rwss_alert_window for r in dt_results
                   if r.rwss_alert_window is not None]
    f1_alerts   = [r.f1_alert_window   for r in dt_results
                   if r.f1_alert_window   is not None]
    if rwss_alerts and f1_alerts:
        mean_rwss_w = round(np.mean(rwss_alerts))
        mean_f1_w   = round(np.mean(f1_alerts))
        if mean_f1_w > mean_rwss_w:
            ax.axvspan(mean_rwss_w, mean_f1_w, alpha=0.12, color=RWSS_COL)
            lags = [r.detection_lag for r in dt_results
                    if r.detection_lag is not None]
            lag_str = f"~{np.mean(lags):.1f}w earlier"
            ax.annotate(lag_str,
                        xy=((mean_rwss_w + mean_f1_w) / 2,
                            RWSS_ALERT_THRESHOLD - 0.025),
                        ha="center", fontsize=10,
                        color=RWSS_COL, fontweight="bold")

    ax.set_xticks(w_nums)
    ax.set_xticklabels(W_LABELS[:n_wins])
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=9, framealpha=0.6, loc="lower left")
    fig.tight_layout()

    path = FIGURES_DIR / f"fig1_detection_lag_{drift_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 2 — All three drift types side by side
# ══════════════════════════════════════════════════════════════════════════════

def fig2_drift_type_comparison(results: List[SeedDriftResult]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, dt in zip(axes, DRIFT_TYPES):
        dtr    = [r for r in results if r.drift_type == dt]
        n_wins = max(len(r.windows) for r in dtr) if dtr else 5
        w_nums = list(range(n_wins))

        rwss_mat = np.array([[w.rwss for w in r.windows] for r in dtr])
        f1_mat   = np.array([[w.f1   for w in r.windows] for r in dtr])
        col      = DRIFT_COL[dt]

        _style(ax, title=f"{dt.capitalize()} drift",
               ylabel="Score" if dt == "covariate" else "")

        ax.fill_between(w_nums,
                        rwss_mat.mean(0) - rwss_mat.std(0),
                        rwss_mat.mean(0) + rwss_mat.std(0),
                        alpha=0.12, color=col)
        ax.plot(w_nums, rwss_mat.mean(0), color=col, linewidth=2.5,
                marker="o", markersize=6, label="RWSS")

        ax.fill_between(w_nums,
                        f1_mat.mean(0) - f1_mat.std(0),
                        f1_mat.mean(0) + f1_mat.std(0),
                        alpha=0.10, color=F1_COL)
        ax.plot(w_nums, f1_mat.mean(0), color=F1_COL, linewidth=2.5,
                marker="s", markersize=6, linestyle="--", label="F1")

        ax.axhline(RWSS_ALERT_THRESHOLD, color=col,
                   linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xticks(w_nums)
        ax.set_xticklabels(W_LABELS[:n_wins], fontsize=9)
        ax.set_ylim(0.45, 1.05)
        if dt == "covariate":
            ax.legend(fontsize=9, framealpha=0.6)

    fig.suptitle("RWSS vs F1 — mean ± std across 5 seeds",
                 fontsize=12, y=1.01, color="#2C2C2A")
    fig.tight_layout()

    path = FIGURES_DIR / "fig2_all_drift_types_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 3 — FIDI heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig3_fidi_heatmap(results: List[SeedDriftResult],
                      drift_type: str = "concept") -> Path:
    TOP_FEATS = ["V14", "V12", "V4", "V11", "V10", "V17", "V3"]

    dtr    = [r for r in results if r.drift_type == drift_type]
    if not dtr:
        return None
    n_wins = len(dtr[0].windows)

    # Accumulate mean FIDI per feature
    fidi_data = {f: np.zeros(n_wins) for f in TOP_FEATS}
    for r in dtr:
        for w in r.windows:
            fidi_data["V14"][w.window] += w.fidi_v14 / len(dtr)
            fidi_data["V12"][w.window] += w.fidi_v12 / len(dtr)
            fidi_data["V4"][w.window]  += w.fidi_v4  / len(dtr)
            # Others default to 0 — replace with real FIDI once computed

    matrix = np.array([fidi_data[f] for f in TOP_FEATS])

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, label="FIDI  (positive = collapsed, negative = amplified)")
    ax.set_xticks(range(n_wins))
    ax.set_xticklabels(W_LABELS[:n_wins], fontsize=10)
    ax.set_yticks(range(len(TOP_FEATS)))
    ax.set_yticklabels(TOP_FEATS, fontsize=10)
    ax.set_title(
        f"Feature Importance Drift Index — {drift_type} drift  (mean, 5 seeds)\n"
        "Red = feature weight collapsed  |  Blue = feature weight amplified",
        fontsize=10, color="#2C2C2A"
    )
    fig.tight_layout()

    path = FIGURES_DIR / f"fig3_fidi_heatmap_{drift_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 4 — V14 weight collapse under concept drift
# ══════════════════════════════════════════════════════════════════════════════

def fig4_v14_collapse(results: List[SeedDriftResult]) -> Path:
    dtr = [r for r in results if r.drift_type == "concept"]
    if not dtr:
        return None

    n_wins  = len(dtr[0].windows)
    w_nums  = list(range(n_wins))
    v14_mat = np.array([[w.fidi_v14 for w in r.windows] for r in dtr])

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    _style(ax,
           title="V14 rule weight collapses under concept drift — all 5 seeds",
           ylabel="FIDI  (positive = collapsed)",
           xlabel="Time window")

    for row in v14_mat:
        ax.plot(w_nums, row, color=RWSS_COL, alpha=0.30,
                linewidth=1.5, marker="o", markersize=4)

    ax.plot(w_nums, v14_mat.mean(0), color=RWSS_COL, linewidth=3.0,
            marker="o", markersize=8, label="Mean across seeds", zorder=5)
    ax.fill_between(w_nums,
                    v14_mat.mean(0) - v14_mat.std(0),
                    v14_mat.mean(0) + v14_mat.std(0),
                    alpha=0.15, color=RWSS_COL)

    ax.axhline(0,    color="#888780", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.20, color=RWSS_COL, linewidth=0.9, linestyle=":",
               alpha=0.7, label="FIDI alert threshold (0.20)")

    ax.set_xticks(w_nums)
    ax.set_xticklabels(W_LABELS[:n_wins])
    ax.legend(fontsize=9, framealpha=0.6)
    fig.tight_layout()

    path = FIGURES_DIR / "fig4_v14_weight_collapse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 5 — Alert timeline grid  (seeds × drift types)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_alert_timeline(results: List[SeedDriftResult]) -> Path:
    seeds = sorted(set(r.seed for r in results))
    n_s   = len(seeds)
    n_d   = len(DRIFT_TYPES)

    fig, axes = plt.subplots(n_s, n_d,
                             figsize=(11, n_s * 1.7),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    if n_s == 1:
        axes = [axes]

    for row, seed in enumerate(seeds):
        for col, dt in enumerate(DRIFT_TYPES):
            ax = axes[row][col]
            ax.set_facecolor(BG_COL)
            ax.spines[:].set_visible(False)
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.tick_params(labelsize=9)

            if row == 0:
                ax.set_title(dt.capitalize(), fontsize=10, color="#3d3d3a")
            if col == 0:
                ax.set_ylabel(f"seed {seed}", fontsize=9, color="#5F5E5A",
                              rotation=0, ha="right", va="center")

            r = next((r for r in results
                      if r.seed == seed and r.drift_type == dt), None)
            if r is None:
                continue

            # BUG 4 FIX — was: range(N_WINDOWS if hasattr(results[0], '__len__') else 5)
            # hasattr(results[0], '__len__') is always False for a dataclass,
            # so this always evaluated to range(5) regardless of actual window count.
            # Now reads window count directly from the result object.
            n_wins = len(r.windows) if r is not None else N_WINDOWS
            ax.set_xlim(-0.5, n_wins - 0.5)
            for w in range(n_wins):
                ax.plot(w, 0, "o", color="#D3D1C7", markersize=9, zorder=1)

            if r.rwss_alert_window is not None:
                ax.plot(r.rwss_alert_window, 0.1, "o",
                        color=RWSS_COL, markersize=13, zorder=4)
            if r.f1_alert_window is not None:
                ax.plot(r.f1_alert_window, -0.1, "s",
                        color=F1_COL, markersize=11, zorder=3, alpha=0.85)

            if (r.detection_lag is not None and r.detection_lag > 0
                    and r.rwss_alert_window is not None):
                ax.annotate(f"+{r.detection_lag}w",
                            xy=(r.rwss_alert_window, 0.38),
                            ha="center", fontsize=8,
                            color=RWSS_COL, fontweight="bold")

    legend_elems = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=RWSS_COL, markersize=11, label="RWSS alert"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=F1_COL,  markersize=10, label="F1 alert"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.6,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Alert timeline — RWSS circles fire before F1 squares\n"
        "+Nw = windows of early detection",
        fontsize=11, y=1.01, color="#2C2C2A"
    )
    fig.tight_layout()

    path = FIGURES_DIR / "fig5_alert_timeline_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Master call
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(results: List[SeedDriftResult]):
    print(f"\n  Generating figures → {FIGURES_DIR}/")
    for dt in DRIFT_TYPES:
        fig1_detection_lag(results, drift_type=dt)
    fig2_drift_type_comparison(results)
    fig3_fidi_heatmap(results, drift_type="concept")
    fig4_v14_collapse(results)
    fig5_alert_timeline(results)
    print(f"  5 figures saved.\n")


def print_summary_table(summary):
    def _lag(val):
        return f"{val:.2f}w" if val is not None else "—"

    print("\n  ── Results summary ──────────────────────────────────────────────────────────────────────")
    print(f"  {'Drift type':12s}  {'F1 fired':>10}  {'RWSS fired':>12}  "
          f"{'VEL fired':>10}  {'FIDIZ fired':>12}  {'PSIR fired':>11}  "
          f"{'RWSS lag':>10}  {'VEL lag':>9}  {'FIDIZ lag':>10}  {'PSIR lag':>10}")
    print("  " + "-" * 118)
    for _, row in summary.iterrows():
        print(f"  {row['drift_type']:12s}  "
              f"{row['f1_fired_rate']:>10}  "
              f"{row['rwss_fired_rate']:>12}  "
              f"{row['rwss_vel_fired_rate']:>10}  "
              f"{row['fidi_z_fired_rate']:>12}  "
              f"{row['psi_r_fired_rate']:>11}  "
              f"{_lag(row['rwss_mean_lag']):>10}  "
              f"{_lag(row['vel_mean_lag']):>9}  "
              f"{_lag(row['fidi_z_mean_lag']):>10}  "
              f"{_lag(row['psi_r_mean_lag']):>10}")
    print()
    print("  Lag = windows before F1 alert. Positive = early warning. Negative = late.")
