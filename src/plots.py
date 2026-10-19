"""Static figures for the README, drawn from the files the pipeline writes to reports/.

    python -m src.plots --reports reports --source "Inside Airbnb, Bogotá, scraped 2026-09"

Design rules applied (data-viz method): one color for single-series charts, emphasis (gray the
rest) for the headline comparison, one-hue sequential ramp for counts and magnitude, a two-pole
diverging scale for the price gap, thin marks, recessive solid hairline grid, titles that state
the finding (computed from the data, so they never claim more than the numbers show), values
labelled selectively, and every figure has a table twin in the CSV/JSON it was drawn from.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator

from .features import REFERENCE_POINTS
from .gap import tail_shape

# ---- palette (reference instance, light surface) -------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, RED = "#2a78d6", "#e34948"          # series slot 1 / diverging warm pole
DEEMPH = "#b5b4ac"                        # de-emphasised marks
# Sequential blue, steps 100/200/300/400/500/600/700
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging midpoint: darker than the palette's #f0efec so near-zero dots stay visible on a light surface
NEUTRAL = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
})

MODEL_LABELS = {
    "random_forest": "Random forest", "ridge": "Ridge regression",
    "lasso": "Lasso regression", "tree_depth4": "Decision tree (depth 4)",
}
REF_LABELS = {"zona_t": "Zona T", "la_candelaria": "La Candelaria", "usaquen": "Usaquén",
              "el_dorado_airport": "El Dorado airport"}


# ---- helpers ------------------------------------------------------------------------------
def _compact(v, pos=None):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}".rstrip("0").rstrip(".") + "M"
    if a >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


COMPACT = FuncFormatter(_compact)


def _header(fig, title, subtitle=None):
    # Wrap long titles instead of letting them run off the canvas.
    lines = textwrap.wrap(title, width=int(fig.get_figwidth() * 8.0)) or [title]
    fig.text(0.04, 0.96, "\n".join(lines), fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        y_sub = 0.96 - (len(lines) * 0.27 + 0.05) / fig.get_figheight()
        fig.text(0.04, y_sub, subtitle, fontsize=10.5, color=INK2, ha="left", va="top")


def _footer(fig, note):
    fig.text(0.04, 0.02, note, fontsize=8, color=MUTED, ha="left", va="bottom")


def _save(fig, out: Path, name: str):
    path = out / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def _pretty(name: str) -> str:
    for prefix, label in [("neighbourhood_cleansed_", "Locality: "), ("room_type_", "Room type: "),
                          ("property_type_", "Property: ")]:
        if name.startswith(prefix):
            return label + name[len(prefix):]
    special = {
        "km_to_zona_t": "Distance to Zona T (km)", "km_to_la_candelaria": "Distance to La Candelaria (km)",
        "km_to_usaquen": "Distance to Usaquén (km)", "km_to_el_dorado_airport": "Distance to airport (km)",
        "n_amenities": "Number of amenities", "accommodates": "Guests accommodated",
        "calculated_host_listings_count": "Host's listing count", "minimum_nights": "Minimum nights",
        "reviews_per_month": "Reviews per month", "number_of_reviews": "Number of reviews",
    }
    if name in special:
        return special[name]
    if name.startswith("has_"):
        return "Has " + name[4:].replace("_", " ")
    return name.replace("_", " ").capitalize()


def _short(name: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", _pretty(name))


def _read_series(path: Path) -> pd.Series:
    return pd.read_csv(path, index_col=0).iloc[:, 0]


def _luminance(hex_color: str) -> float:
    r, g, b = to_rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---- 1. model comparison ------------------------------------------------------------------
def fig_model_comparison(rep: Path, out: Path, note: str):
    df = pd.read_csv(rep / "price_model_comparison.csv").sort_values("MAPE_pct").reset_index(drop=True)
    vals = df["MAPE_pct"].to_numpy()
    best = int(vals.argmin())
    colors = [BLUE if i == best else DEEMPH for i in range(len(vals))]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.24, right=0.95)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.38, color=colors)
    for yi, v, mae in zip(y, vals, df["MAE_local_currency"]):
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=11.5, fontweight="bold", color=INK)
        ax.annotate(f"typical miss {mae / 1e3:.0f}K COP", (v, yi), xytext=(66, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in df["model"]], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, vals.max() * 1.45)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
    ax.grid(axis="x")
    ax.set_xlabel("Mean absolute percentage error on held-out listings (lower is better)", fontsize=9.5)

    if df["model"].iloc[best] == "random_forest" and len(vals) > 1:
        runner_up = np.delete(vals, best).min()
        cut = (runner_up - vals[best]) / runner_up * 100
        title = f"Random forest cuts pricing error by {cut:.0f}% vs. the best simpler model"
    else:
        title = "Held-out pricing error by model"
    _header(fig, title, "Nightly-price prediction on Bogotá listings the models never saw during training")
    _footer(fig, note)
    _save(fig, out, "01_model_comparison.png")


# ---- 2. actual vs fair price --------------------------------------------------------------
def fig_actual_vs_fair(rep: Path, out: Path, note: str):
    t = pd.read_csv(rep / "price_gap_listings.csv", usecols=["price", "fair_price"])
    t = t[(t["price"] > 0) & (t["fair_price"] > 0)]
    lo = min(t["price"].min(), t["fair_price"].min()) * 0.9
    hi = max(t["price"].max(), t["fair_price"].max()) * 1.1

    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.12, right=0.9)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    hb = ax.hexbin(t["fair_price"], t["price"], xscale="log", yscale="log", gridsize=55,
                   mincnt=1, cmap=cmap, norm=LogNorm(vmin=1), linewidths=0.0)
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=INK2, lw=1.2)
    ax.plot(xs, xs * 1.3, color=MUTED, lw=0.8)
    ax.plot(xs, xs * 0.7, color=MUTED, lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    nice = [20e3, 30e3, 50e3, 70e3, 100e3, 150e3, 200e3, 300e3, 500e3, 700e3, 1e6, 1.5e6, 2e6, 3e6,
            5e6, 7e6, 10e6, 20e6]
    ticks = [v for v in nice if lo <= v <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(COMPACT)
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    # Short labels at the end of each line, where the plot is normally empty (no boxes over data)
    fig.canvas.draw()
    p0 = ax.transData.transform((lo * 5, lo * 5))
    p1 = ax.transData.transform((lo * 6, lo * 6))
    angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    for k, text in ((1.0, "fair price"), (1.3, "+30%"), (0.7, "−30%")):
        x_end = min(hi, hi / k) * 0.985
        ax.text(x_end, k * x_end, text, rotation=angle, rotation_mode="anchor", fontsize=9.5,
                color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Fair price from out-of-fold random forest (COP per night, log scale)", fontsize=9.5)
    ax.set_ylabel("Listed price (COP per night, log scale)", fontsize=9.5)
    ax.grid(True)
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("Listings per cell (log scale)", fontsize=9, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    vmax = float(np.nanmax(hb.get_array()))
    cticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500) if v <= vmax]
    cb.set_ticks(cticks)
    cb.set_ticklabels([f"{v:,}" for v in cticks])
    cb.ax.minorticks_off()

    within = float((np.abs(t["price"] / t["fair_price"] - 1) <= 0.30).mean() * 100)
    _header(fig, f"{within:.0f}% of listings are priced within 30% of their fair price",
            "Lines mark the ±30% flag thresholds; fair prices come from a model that never saw each listing")
    _footer(fig, note)
    _save(fig, out, "02_actual_vs_fair_price.png")


# ---- 3. gap vs occupancy ------------------------------------------------------------------
def fig_gap_occupancy(rep: Path, out: Path, note: str):
    b = pd.read_csv(rep / "price_gap_buckets.csv")
    s = json.loads((rep / "price_gap_summary.json").read_text())
    s = {**s, **tail_shape(b)}  # derive shape/active diagnostics from the buckets file itself
    r = s["spearman_gap_vs_occupancy_within_locality"]
    use_active = {"share_active_ltm_pct", "median_occupancy_active_only"} <= set(b.columns)

    names = ["Very\nunderpriced", "Under-\npriced", "In line\n", "Over-\npriced", "Very\noverpriced"]
    ranges = ["< −30%", "−30 to −15%", "−15 to +15%", "+15 to +30%", "> +30%"]
    n = b["listings"].to_numpy()
    labels = [f"{a}\n{rg}\nn = {k:,}" for a, rg, k in zip(names, ranges, n)]
    x = np.arange(len(b))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(top=0.76, bottom=0.22, left=0.06, right=0.98, wspace=0.18)
    if use_active:
        # Raw occupancy mixes in listings that are open but dormant; separate the two questions.
        b["share_active_ltm"] = b["share_active_ltm_pct"] / 100
        panels = [
            ("share_active_ltm", "Share of listings active (a review in the last 12 months)",
             lambda v, d: f"{v * 100:.{d}f}%", False),
            ("median_occupancy_active_only", "Median occupancy among active listings",
             lambda v, d: f"{v * 100:.{d}f}%", False),
        ]
    else:
        panels = [
            ("median_occupancy", "Median occupancy (share of year booked)",
             lambda v, d: f"{v * 100:.{d}f}%", False),
            ("median_occ_vs_locality", "Occupancy vs. the listing's locality median (points)",
             lambda v, d: f"{v * 100:+.{d}f}".replace("-", "−"), True),
        ]
    for ax, (col, ttl, fmt, signed) in zip(axes, panels):
        v = b[col].to_numpy(dtype=float)
        ax.bar(x, np.nan_to_num(v), width=0.42, color=BLUE)
        span = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
        dec = 1 if span * 100 < 25 else 0  # small effects need a decimal to be readable
        for xi, vi in zip(x, v):
            if not np.isfinite(vi):
                continue
            off = span * 0.03
            ax.text(xi, vi + (off if vi >= 0 else -off), fmt(vi, dec), ha="center",
                    va="bottom" if vi >= 0 else "top", fontsize=10.5, fontweight="bold", color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.8)
        ax.tick_params(axis="x", length=0)
        ax.set_title(ttl, fontsize=10.5, color=INK2, loc="left", pad=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))  # whole-number percent ticks
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, p: f"{t * 100:.0f}".replace("-", "−")))
        ax.grid(axis="y")
        ax.axhline(0, color=AXIS if not signed else INK2, lw=0.9)
        if signed:
            ax.set_ylim(-span * 1.3, span * 1.3)
        else:
            ax.set_ylim(0, span * 1.18)
    axes[0].set_ylabel("Percent", fontsize=9.5)
    axes[1].set_ylabel("Percent" if use_active else "Percentage points", fontsize=9.5)

    if r <= -0.10:
        title = "Listings priced above their peers book less"
    elif use_active and s.get("tails_less_active"):
        lower = [name for name, ratio in (("underpriced", s["active_occ_vs_in_line_underpriced"]),
                                          ("overpriced", s["active_occ_vs_in_line_overpriced"])) if ratio < 0.9]
        if not lower:
            title = "Far-from-peer prices go with inactive listings; active ones book at similar rates"
        elif len(lower) == 1:
            title = f"Far-from-peer prices go with inactive listings; very {lower[0]} ones also book less when active"
        else:
            title = "Far-from-peer prices go with inactive listings, and both extremes book less even when active"
    elif s.get("inverted_u"):
        title = "Listings priced far from peers, in either direction, show less booking activity"
    elif r >= 0.05:
        title = "Price gap does not predict bookings: pricier-than-peers listings do not book less"
    else:
        title = "Price gap shows only a weak link to occupancy"
    r_txt = f"{r:+.2f}".replace("-", "−")
    if use_active and s.get("tails_less_active"):
        sub = "Active = at least one review in the last 12 months. Bars are medians by price-gap bucket; occupancy is a proxy."
    elif s.get("inverted_u") and r > -0.10:
        sub = f"The pattern is a hump, not a line, so the within-locality rank correlation ({r_txt}) understates it."
    else:
        sub = (f"Rank correlation between price gap and occupancy within locality: {r_txt}. "
               "Listings are grouped by how far they sit from their fair price.")
    _header(fig, title, sub)
    _footer(fig, note + " Correlation across different listings, not a causal estimate.")
    _save(fig, out, "03_price_gap_vs_occupancy.png")


# ---- 4. map -------------------------------------------------------------------------------
def fig_gap_map(rep: Path, out: Path, note: str):
    t = pd.read_csv(rep / "price_gap_listings.csv")
    if not {"latitude", "longitude"} <= set(t.columns):
        print("[plots] skipped map: price_gap_listings.csv has no latitude/longitude (rerun the pipeline).")
        return
    s = json.loads((rep / "price_gap_summary.json").read_text())
    t = t.dropna(subset=["latitude", "longitude", "gap_pct"])
    la0, la1 = t["latitude"].quantile([0.005, 0.995])
    lo0, lo1 = t["longitude"].quantile([0.005, 0.995])
    t = t[t["latitude"].between(la0, la1) & t["longitude"].between(lo0, lo1)]
    t = t.loc[t["gap_pct"].abs().sort_values().index]  # extremes drawn last, on top

    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)

    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    fig.subplots_adjust(top=0.84, bottom=0.07, left=0.04, right=0.88)
    sc = ax.scatter(t["longitude"], t["latitude"], c=t["gap_pct"].clip(-60, 60), cmap=cmap, norm=norm,
                    s=7, alpha=0.75, linewidths=0, rasterized=True)
    for key, (lat, lon) in REFERENCE_POINTS.items():
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            ax.scatter([lon], [lat], s=34, color=INK, edgecolors=SURFACE, linewidths=1.4, zorder=5)
            ax.text(lon + 0.004, lat + 0.003, REF_LABELS.get(key, key), fontsize=9.5, color=INK,
                    zorder=6, ha="left", va="bottom")
    ax.set_aspect(1 / np.cos(np.radians(t["latitude"].mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035, ticks=[-60, -30, 0, 30, 60])
    cb.ax.set_yticklabels(["≤ −60", "−30", "0", "+30", "≥ +60"])
    cb.set_label("Price vs. fair price (%)", fontsize=9.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=9, length=0)

    _header(fig,
            f"{s['flagged_overpriced']:,} listings flagged overpriced, {s['flagged_underpriced']:,} underpriced",
            f"Each dot is a listing; blue sits below its fair price, red above (flag threshold ±{s['flag_threshold_pct']:.0f}%)")
    _footer(fig, note + " No basemap; reference points are approximate.")
    _save(fig, out, "04_price_gap_map.png")


# ---- 5. drivers ---------------------------------------------------------------------------
def fig_drivers(rep: Path, out: Path, note: str):
    imp = _read_series(rep / "forest_importance.csv").sort_values(ascending=False).head(12)
    coef = _read_series(rep / "lasso_coefficients.csv")
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)
    pct = ((np.exp(coef) - 1) * 100).sort_values()
    imp = imp.sort_values()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.4))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.2, right=0.97, wspace=0.78)

    a1.barh(np.arange(len(imp)), imp.to_numpy() * 100, height=0.5, color=BLUE)
    for i, v in enumerate(imp.to_numpy() * 100):
        a1.text(v + imp.max() * 100 * 0.015, i, f"{v:.0f}%", va="center", fontsize=10, color=INK)
    a1.set_yticks(np.arange(len(imp)))
    a1.set_yticklabels([_pretty(c) for c in imp.index], fontsize=9.5)
    a1.set_xlim(0, imp.max() * 100 * 1.15)
    a1.set_title("Random forest: share of predictive importance", fontsize=10.5, color=INK2, loc="left", pad=10)
    a1.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))  # no 2.5 -> "2%"
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))

    cols = [BLUE if v > 0 else RED for v in pct.to_numpy()]
    a2.barh(np.arange(len(pct)), pct.to_numpy(), height=0.5, color=cols)
    span = np.abs(pct).max()
    for i, v in enumerate(pct.to_numpy()):
        a2.text(v + (span * 0.02 if v >= 0 else -span * 0.02), i, f"{v:+.0f}%".replace("-", "−"), va="center",
                ha="left" if v >= 0 else "right", fontsize=10, color=INK)
    a2.set_yticks(np.arange(len(pct)))
    a2.set_yticklabels([_pretty(c) for c in pct.index], fontsize=9.5)
    a2.set_xlim(-span * 1.3 if (pct < 0).any() else -span * 0.05, span * 1.3)
    a2.axvline(0, color=INK2, lw=0.9)
    a2.set_title("Lasso: price change per 1 SD of the feature", fontsize=10.5, color=INK2, loc="left", pad=10)
    a2.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10], integer=True))
    a2.xaxis.set_major_formatter(
        FuncFormatter(lambda v, p: "0%" if abs(v) < 1e-9 else f"{v:+.0f}%".replace("-", "−")))
    a2.legend(handles=[Patch(color=BLUE, label="Raises price"), Patch(color=RED, label="Lowers price")],
              loc="lower right", frameon=False, fontsize=9)
    for ax in (a1, a2):
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")

    top_f, top_l = imp.index[-1], coef.abs().idxmax()
    if top_f == top_l:
        title = f"{_short(top_f)} is the strongest price driver in both models"
    else:
        title = f"{_short(top_f)} tops the forest; {_short(top_l)} tops the linear model"
    _header(fig, title,
            "Two views of the same question: the forest ranks usefulness for prediction; "
            "the linear model shows direction and size")
    _footer(fig, note)
    _save(fig, out, "05_price_drivers.png")


# ---- 6. tree ------------------------------------------------------------------------------
def fig_tree(rep: Path, out: Path, note: str, max_depth: int = 3):
    obj = joblib.load(rep / "tree_depth4.joblib")
    t, feats = obj["tree"].tree_, obj["features"]
    pos: dict = {}
    counter = [0]

    def layout(node, depth):
        left, right = t.children_left[node], t.children_right[node]
        if depth == max_depth or left == -1:
            pos[node] = (counter[0], -depth, True)
            counter[0] += 1
            return
        layout(left, depth + 1)
        layout(right, depth + 1)
        pos[node] = ((pos[left][0] + pos[right][0]) / 2, -depth, False)

    layout(0, 0)
    leaf_logs = [t.value[n][0][0] for n, p in pos.items() if p[2]]
    lo, hi = min(leaf_logs), max(leaf_logs)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ[:6])

    fig, ax = plt.subplots(figsize=(16, 7.4))
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.01, right=0.99)
    for node, (x, y, is_leaf) in pos.items():
        if is_leaf:
            continue
        for child, side in ((t.children_left[node], "yes"), (t.children_right[node], "no")):
            cx, cy, _ = pos[child]
            ax.plot([x, cx], [y - 0.12, cy + 0.14], color=AXIS, lw=1.1, zorder=1)
            ax.text(x + (cx - x) * 0.45, y + (cy - y) * 0.42, side, fontsize=8, color=MUTED,
                    ha="center", va="center", zorder=2,
                    bbox=dict(fc=SURFACE, ec="none", pad=0.8))
    for node, (x, y, is_leaf) in pos.items():
        n = int(t.n_node_samples[node])
        if is_leaf:
            price = float(np.exp(t.value[node][0][0]))
            shade = (t.value[node][0][0] - lo) / (hi - lo) if hi > lo else 0.5
            fc = cmap(0.12 + 0.88 * shade)
            hexfc = matplotlib.colors.to_hex(fc)
            tc = "white" if _luminance(hexfc) < 0.45 else INK
            ax.text(x, y, f"≈ {price / 1e3:.0f}K COP\nn = {n:,}", ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=tc, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.45", fc=hexfc, ec="none"))
        else:
            name = _short(feats[t.feature[node]])
            thr = t.threshold[node]
            ax.text(x, y, f"{name}\n≤ {thr:.3g}\nn = {n:,}", ha="center", va="center", fontsize=9.5,
                    color=INK, zorder=3, bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=AXIS, lw=0.9))
    ax.set_xlim(-0.7, counter[0] - 0.3)
    ax.set_ylim(-max_depth - 0.55, 0.5)
    ax.axis("off")
    _header(fig, f"The first {max_depth} splits a host could apply by hand to estimate a nightly price",
            "'yes' = condition true. Features that are 0/1 flags split at 0.5. Leaf boxes show the typical "
            "price (COP) and listing count; darker = pricier.")
    _footer(fig, note + " Full depth-4 rules are in reports/tree_rules.txt.")
    _save(fig, out, "06_decision_tree.png")


# ---- entry point --------------------------------------------------------------------------
def main(reports: str = "reports", out: str | None = None, source: str = "Inside Airbnb, Bogotá"):
    rep = Path(reports)
    outdir = Path(out) if out else rep / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    note = f"Source: {source}. Prices in COP per night. Occupancy is a proxy, not observed bookings."
    for fn in (fig_model_comparison, fig_actual_vs_fair, fig_gap_occupancy, fig_gap_map, fig_drivers, fig_tree):
        try:
            fn(rep, outdir, note)
        except FileNotFoundError as e:
            print(f"[plots] skipped {fn.__name__}: missing {Path(e.filename).name} (run src.run first)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default="Inside Airbnb, Bogotá")
    a = ap.parse_args()
    main(a.reports, a.out, a.source)
