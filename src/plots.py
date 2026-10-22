"""Static charts (matplotlib, headless). Styling follows a small fixed spec: one hue for one series, thin marks,
recessive solid hairline grid, values at the bar tip, text always in ink colours (never the series colour),
and a legend + distinct marker shapes whenever there is more than one series."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")  # headless scripts/CI; inside a notebook the inline backend must stay in charge
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"H": "#2a78d6", "D": "#eb6834", "A": "#1baf7a"}   # categorical slots 1-3 (validated together)
MARKERS = {"H": "o", "D": "s", "A": "^"}
CLASS_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def plot_champion_probabilities(table: pd.DataFrame, path: str | Path, *, title: str, banner: str, n_sims: int,
                                top_n: int = 16) -> Path:
    """Horizontal bars of championship probability with 95% Monte Carlo whiskers."""
    df = table.head(top_n).iloc[::-1]  # largest at the top
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(df) + 2.3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = np.arange(len(df))
    p = df["champion_probability"].to_numpy() * 100
    lo, hi = df["ci95_low"].to_numpy() * 100, df["ci95_high"].to_numpy() * 100
    ax.barh(y, p, height=0.52, color=SERIES["H"], zorder=3)
    ax.errorbar(p, y, xerr=[p - lo, hi - p], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=2.5, zorder=4)
    xmax = max(hi.max() * 1.22, 1.0)
    for yi, pi, hii in zip(y, p, hi):
        ax.text(hii + xmax * 0.012, yi, f"{pi:.1f}%", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(df["team_name"], fontsize=9.5, color=INK)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10], integer=True))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Probability of winning the Champions League", color=INK2, fontsize=9)
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.915, banner, fontsize=9.5, color="#a33a1c", ha="left", va="top", fontweight="bold")
    fig.text(0.02, 0.012, f"Bars: share of {n_sims:,} simulated tournaments won. Whiskers: 95% Wilson interval - "
             "Monte Carlo noise only, not model uncertainty.", fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.12)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_calibration(table: pd.DataFrame, path: str | Path, *, title: str, subtitle: str) -> Path:
    """Reliability curves (predicted vs observed frequency) for H/D/A, built from ``calibration_table``."""
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.2, zorder=1, label="Perfect calibration")
    for cls in ("H", "D", "A"):
        t = table[table["class"] == cls].sort_values("mean_predicted")
        ax.plot(t["mean_predicted"], t["observed_frequency"], color=SERIES[cls], linewidth=2, marker=MARKERS[cls],
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=CLASS_NAMES[cls], zorder=3)
    lim = float(max(table["mean_predicted"].max(), table["observed_frequency"].max()) * 1.08)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Predicted probability (bin average)", color=INK2, fontsize=9)
    ax.set_ylabel("Observed frequency", color=INK2, fontsize=9)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.text(0.04, 0.975, title, fontsize=12, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.04, 0.925, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.86, bottom=0.12)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_outcome_shares(matches: pd.DataFrame, path: str | Path | None = None):
    """Share of home wins / draws / away wins per season (90-minute results)."""
    done = matches[matches["is_completed"]].copy()
    done["result"] = np.where(done["home_goals_90"] > done["away_goals_90"], "H",
                              np.where(done["home_goals_90"] < done["away_goals_90"], "A", "D"))
    shares = done.groupby(["season", "result"]).size().unstack(fill_value=0)
    shares = shares.div(shares.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for cls in ("H", "D", "A"):
        ax.plot(shares.index, shares[cls], color=SERIES[cls], linewidth=2, marker=MARKERS[cls], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=CLASS_NAMES[cls])
    ax.set_xticks(shares.index)
    ax.set_xticklabels([f"{s}/{str(s + 1)[-2:]}" for s in shares.index], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, max(0.6, float(shares.to_numpy().max()) * 1.1))
    leg = ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.set_title("Match outcomes by season (90 minutes)", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_goal_histogram(matches: pd.DataFrame, path: str | Path | None = None):
    """Total goals per match vs a Poisson with the same mean (is the Poisson goals assumption reasonable?)."""
    from scipy.stats import poisson

    done = matches[matches["is_completed"]]
    total = (done["home_goals_90"] + done["away_goals_90"]).astype(int)
    counts = total.value_counts(normalize=True).sort_index()
    k = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.bar(counts.index, counts.values, width=0.5, color=SERIES["H"], label="Observed", zorder=3)
    ax.plot(k, poisson.pmf(k, total.mean()), color=INK2, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            label=f"Poisson (mean {total.mean():.2f})", zorder=4)
    ax.set_xlim(-0.6, 10.6)
    ax.set_xticks(k)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Total goals in 90 minutes", color=INK2, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.set_title("Goals per match", loc="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_model_comparison(pooled: pd.DataFrame, chosen: str, title: str, path: str | Path | None = None):
    """Pooled out-of-sample log loss per model (lower is better); the no-skill climatology baseline is the first bar."""
    df = pooled.sort_values("log_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(df) + 1.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = np.arange(len(df))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.scatter(df["log_loss"], y, s=70, color=SERIES["H"], edgecolor=SURFACE, linewidth=1.5, zorder=4)  # dots: a non-zero axis is fine
    for yi, (m, v) in enumerate(zip(df["model"], df["log_loss"])):
        ax.text(v + 0.003, yi, f"{v:.4f}" + ("  <- selected" if m == chosen else ""), va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=9.5, color=INK)
    lo = float(df["log_loss"].min()) - 0.02
    ax.set_xlim(lo, float(df["log_loss"].max()) + 0.045)
    ax.set_xlabel("Multiclass log loss, pooled out-of-sample folds (lower is better; the axis does not start at zero)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig
