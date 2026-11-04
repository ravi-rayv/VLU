"""Core figures (PNG + SVG) in the style used in the manuscript."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib as mpl  # noqa: E402

from .config import BANDS, GROUPS, LANDUSE  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, LIGHT_BLUE, NODATA = "#2a78d6", "#86b6ef", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
FLOOR_BANDS = [0, 1, 3, 6, 10, 15, 1000]
FLOOR_NAMES = ["1", "2–3", "4–6", "7–10", "11–15", "16+"]
FLOOR_COLS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 200, "savefig.bbox": "tight",
}


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, name)
    fig.savefig(base + ".png")
    fig.savefig(base + ".svg")
    plt.close(fig)
    return base + ".png"


def floors_distribution(bld, fig_dir, title):
    with mpl.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
        h = bld.loc[bld["h_median"] >= 2.5, "h_median"]
        if len(h):
            a1.hist(h, bins=np.arange(0, min(h.max(), 150) + 3, 3), color=BLUE, edgecolor=SURFACE, linewidth=1)
            a1.axvline(h.median(), color=INK2, lw=1, ls="--")
            a1.annotate(f"median {h.median():.1f} m", (h.median(), a1.get_ylim()[1] * 0.92),
                        xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=9)
        a1.set(title="Building height", xlabel="Median height (m)", ylabel="Buildings"); a1.grid(axis="x", visible=False)
        band = pd.cut(bld["floors_final"], FLOOR_BANDS, labels=FLOOR_NAMES)
        counts = band.value_counts().reindex(FLOOR_NAMES).fillna(0)
        bars = a2.bar(FLOOR_NAMES, counts.values, color=FLOOR_COLS, edgecolor=SURFACE, linewidth=2)
        a2.bar_label(bars, labels=[f"{v:,.0f}\n{v / max(len(bld), 1):.0%}" for v in counts.values],
                     fontsize=8.5, color=INK2, padding=2)
        a2.set(title="Floors per building", xlabel="Floors", ylabel="Buildings"); a2.margins(y=0.18)
        a2.grid(axis="x", visible=False)
        fig.suptitle(f"{title}: {len(bld):,} OSM buildings", x=0.01, ha="left", fontsize=13, fontweight="bold")
        return _save(fig, fig_dir, "fig_floors_distribution")


def calibration(cal_info, fig_dir):
    if "calibration" not in cal_info:
        return None
    c = cal_info["calibration"]; good = c[c["plausible"]]
    with mpl.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
        rng = np.random.default_rng(0)
        a1.scatter(good["building_levels"] + rng.uniform(-0.25, 0.25, len(good)), good["h_median"],
                   s=8, alpha=0.35, color=BLUE, lw=0, label="Tagged buildings")
        med = good.groupby("building_levels")["h_median"].median()
        a1.plot(med.index, med.values, "o", mfc="white", mec=INK, ms=5, label="Median per floor count")
        if cal_info.get("slope"):
            f = np.linspace(1, max(good["building_levels"].max(), 2), 100)
            a1.plot(f, (f - cal_info["intercept"]) / cal_info["slope"], color=INK2, lw=1.5,
                    label=f"Fitted linear model (≈ {cal_info['storey_m']:.1f} m/floor)")
        a1.set(title="Satellite height vs OSM floor tags", xlabel="Floors (OSM building:levels)",
               ylabel="Median satellite height (m)")
        a1.legend(frameon=False, fontsize=8.5)
        cv = cal_info["cv"]
        if len(cv):
            bars = a2.barh(cv.index, cv["within_1"], color=[LIGHT_BLUE, BLUE, LIGHT_BLUE], height=0.5)
            a2.bar_label(bars, labels=[f"{v:.0%}  MAE {m:.2f}" for v, m in zip(cv["within_1"], cv["MAE"])],
                         padding=4, fontsize=8.5, color=INK2)
            a2.set_xlim(0, 1.25); a2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
        a2.set(title="Held-out buildings within ±1 floor"); a2.grid(axis="y", visible=False)
        return _save(fig, fig_dir, "fig_calibration")


def vertical_profile(profile: pd.DataFrame, fig_dir):
    with mpl.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
        g = pd.DataFrame({k: profile[v].sum(1) for k, v in GROUPS.items()})
        left = np.zeros(len(g))
        for i, col in enumerate(g.columns):
            a1.barh(g.index, g[col], left=left, color=CAT[i], label=col, height=0.62)
            for y, (l, v) in enumerate(zip(left, g[col])):
                if v >= 0.07:
                    a1.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if i < 3 else INK)
            left += g[col].values
        a1.set_xlim(0, 1); a1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
        a1.set(title="Land-use mix on each floor band"); a1.grid(axis="y", visible=False)
        a1.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=8.5)
        a2.barh(profile.index, profile["expected_POIs"], color=BLUE, height=0.62)
        a2.set(title="POIs per band"); a2.set_yticklabels([]); a2.grid(axis="y", visible=False)
        return _save(fig, fig_dir, "fig_vertical_profile")


def grid_map(grid, aoi, fig_dir):
    if grid is None or not len(grid):
        return None
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 8))
        v = float(np.nanmax(np.abs(grid["VSI_excess"]))) or 0.1
        cmap = mpl.colors.LinearSegmentedColormap.from_list("div", ["#d03b3b", "#e98a86", "#f0efec", "#86b6ef", "#2a78d6"])
        aoi.boundary.plot(ax=ax, color=INK2, lw=0.8)
        grid.plot(ax=ax, column="VSI_excess", cmap=cmap, vmin=-v, vmax=v, edgecolor="white", lw=0.3,
                  legend=True, legend_kwds={"shrink": 0.6, "label": "VSI_excess"})
        ax.set_axis_off(); ax.set_title("Vertical stratification beyond chance (grid)")
        return _save(fig, fig_dir, "fig_grid_vsi_excess")


def buildings_mix(b_idx, fig_dir):
    if b_idx is None or not len(b_idx):
        return None
    d = b_idx[b_idx["H_use"] > 0]
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(d["H_use"], d["VSI_excess"], s=np.clip(d["n_poi"] * 4, 6, 200), alpha=0.6, color=BLUE, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        top = d.sort_values("n_poi", ascending=False).head(5)
        for _, r in top.iterrows():
            if isinstance(r.get("name"), str) and r["name"]:
                ax.annotate(r["name"], (r["H_use"], r["VSI_excess"]), xytext=(5, 5), textcoords="offset points",
                            fontsize=8, color=INK2, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set(title="Buildings: land-use mix vs stratification beyond chance",
               xlabel="Land-use mix (H_use / ln K)", ylabel="VSI_excess")
        return _save(fig, fig_dir, "fig_buildings_mix")


__all__ = ["floors_distribution", "calibration", "vertical_profile", "grid_map", "buildings_mix", "LANDUSE", "BANDS"]
