"""Exploratory data analysis: six figures that justify the modelling choices made later.

    python -m src.eda --listings raw/listings.csv.gz --reports reports \
        --source "Inside Airbnb, Bogotá, scraped 2026-09"

Each figure answers one design question:
  00a  Why model log price and trim the tails?        -> price distribution, raw vs. log
  00b  How much does location / room type matter?      -> price by locality and room type
  00c  What is the basic structure the models learn?   -> price by bedrooms and capacity
  00d  Why impute and add missing-value indicators?    -> missingness by column
  00e  Why is occupancy only a proxy?                  -> occupancy distribution and activity
  00f  Why hold out whole hosts?                       -> listings per host

Same visual rules as plots.py (one colour, thin marks, titles computed from the data). The numbers
behind each figure are written to reports/eda_*.csv and reports/eda_summary.json.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator

from . import data
from .plots import BLUE, COMPACT, DEEMPH, GRID, INK, INK2, MUTED, SEQ, _footer, _header, _save

TOP_LOCALITIES = 12
MIN_GROUP_N = 30


# ---- helpers ------------------------------------------------------------------------------
def _nice_ticks(lo: float, hi: float, max_ticks: int = 7) -> list[float]:
    """Round ticks inside [lo, hi], thinned to at most max_ticks."""
    cands = []
    for d in range(int(np.floor(np.log10(max(lo, 1e-9)))) - 1, int(np.ceil(np.log10(hi))) + 1):
        for m in (1, 1.5, 2, 3, 5, 7):
            v = m * 10.0 ** d
            if lo <= v <= hi:
                cands.append(v)
    if len(cands) > max_ticks:
        cands = [v for v in cands if round(v / 10.0 ** np.floor(np.log10(v)), 1) in (1.0, 2.0, 5.0)]
    return cands


def _log_price_axis(ax, lo: float, hi: float, axis: str = "x", max_ticks: int = 7):
    ticks = _nice_ticks(lo, hi, max_ticks)
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}lim")(lo, hi)
    a = getattr(ax, f"{axis}axis")
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(COMPACT)
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def _panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=10.5, color=INK2, pad=10)


def _pct_axis(ax, axis="y", decimals=0):
    fmt = FuncFormatter(lambda v, p: f"{v:.{decimals}f}%")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))  # never 2.5 -> "2%"
    a.set_major_formatter(fmt)


def _money(v: float) -> str:
    return f"{v:,.0f}"


# ---- 00a price distribution ---------------------------------------------------------------
def fig_price_distribution(raw_price: pd.Series, out: Path, note: str) -> dict:
    p = raw_price[raw_price > 0].dropna()
    lo, hi = p.quantile([data.PRICE_FLOOR_QUANTILE, data.PRICE_CAP_QUANTILE])
    shown_hi = p.quantile(0.995)
    shown_lo = p.quantile(0.005)
    trimmed = p[(p >= lo) & (p <= hi)]
    skew_all = float(p.skew())                          # untrimmed: driven by a few extreme prices
    skew_raw, skew_log = float(trimmed.skew()), float(np.log(trimmed).skew())
    minus = lambda v: f"{v:.1f}".replace("-", "\u2212")
    w = 100.0 / len(p)  # each listing counts for its share of all listings

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.97, wspace=0.2)

    v = p[p <= shown_hi]
    a1.hist(v, bins=np.linspace(0, shown_hi, 41), weights=np.full(len(v), w), color=BLUE,
            edgecolor="#fcfcfb", linewidth=0.5)
    a1.axvline(hi, color=INK2, lw=0.9)
    a1.annotate("99th percentile", (hi, a1.get_ylim()[1]), xytext=(-5, -4), textcoords="offset points",
                ha="right", va="top", fontsize=9, color=INK2)
    a1.xaxis.set_major_formatter(COMPACT)
    a1.set_xlim(0, shown_hi)
    _panel_title(a1, "Nightly price, linear scale (COP)")

    lv = np.log10(p[(p >= shown_lo) & (p <= shown_hi)])
    a2.hist(lv, bins=np.linspace(np.log10(shown_lo), np.log10(shown_hi), 41),
            weights=np.full(len(lv), w), color=BLUE, edgecolor="#fcfcfb", linewidth=0.5)
    for q, lab, ha, dx in [(lo, "1st percentile", "left", 5), (hi, "99th percentile", "right", -5)]:
        a2.axvline(np.log10(q), color=INK2, lw=0.9)
        a2.annotate(lab, (np.log10(q), a2.get_ylim()[1]), xytext=(dx, -4), textcoords="offset points",
                    ha=ha, va="top", fontsize=9, color=INK2)
    ticks = _nice_ticks(shown_lo, shown_hi)
    a2.set_xticks([np.log10(t) for t in ticks])
    a2.set_xticklabels([COMPACT(t) for t in ticks])
    a2.set_xlim(np.log10(shown_lo), np.log10(shown_hi))
    _panel_title(a2, "Nightly price, log scale (COP)")

    for ax in (a1, a2):
        ax.grid(axis="y")
        _pct_axis(ax)
        ax.set_ylabel("Share of listings per bar", fontsize=9.5)
        ax.tick_params(axis="x", length=0)

    if skew_raw > 0.8 and abs(skew_log) < abs(skew_raw) / 2:
        title = (f"Prices stay right-skewed even after trimming (skew {minus(skew_raw)}); "
                 f"a log scale brings it to {minus(skew_log)}")
    else:
        title = "Distribution of nightly prices, raw and on a log scale"
    _header(fig, title, f"Before trimming the skew is {minus(skew_all)}, driven by a few extreme prices. "
                        "Hence log price and a 1st-99th percentile trim.")
    _footer(fig, f"{note}  Bars show the middle 99% of prices; the highest listed price is "
                 f"{_money(p.max())} COP ({p.max() / p.median():.0f}x the median).")
    _save(fig, out, "00a_price_distribution.png")
    return {"price_skew_untrimmed": skew_all, "price_skew_trimmed": skew_raw,
            "price_skew_log_trimmed": skew_log, "trim_low": float(lo),
            "trim_high": float(hi), "max_price": float(p.max()), "median_price": float(p.median())}


# ---- 00b price by locality and room type --------------------------------------------------
def _group_stats(df: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    g = df.groupby(col)["price"].agg(median="median", q25=lambda s: s.quantile(0.25),
                                    q75=lambda s: s.quantile(0.75), n="size")
    g = g[g["n"] >= MIN_GROUP_N]
    if top:
        g = g.sort_values("n", ascending=False).head(top)
    return g.sort_values("median", ascending=False)


def _range_dots(ax, g: pd.DataFrame, lo: float, hi: float, max_ticks: int = 7):
    y = np.arange(len(g))[::-1]
    ax.hlines(y, g["q25"], g["q75"], color=SEQ[2], lw=3.0, zorder=2)
    ax.scatter(g["median"], y, s=38, color=BLUE, zorder=3, edgecolor="#fcfcfb", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i}   n={n:,}" for i, n in zip(g.index, g["n"])], fontsize=10)
    ax.tick_params(axis="y", length=0)
    _log_price_axis(ax, lo, hi, max_ticks=max_ticks)
    ax.grid(axis="x")
    ax.set_ylim(-0.7, len(g) - 0.3)


def fig_price_by_group(df: pd.DataFrame, out: Path, note: str, tables: dict) -> dict:
    loc = _group_stats(df, "neighbourhood_cleansed", TOP_LOCALITIES) if "neighbourhood_cleansed" in df else None
    room = _group_stats(df, "room_type") if "room_type" in df else None
    panels = [(t, g) for t, g in [("Locality (the 12 with the most listings)", loc), ("Room type", room)]
              if g is not None and len(g) > 1]
    if not panels:
        return {}
    lo = df["price"].quantile(0.01) * 0.9
    hi = df["price"].quantile(0.99) * 1.1
    ratios = [g["median"].iloc[0] / g["median"].iloc[-1] for _, g in panels[:1]]

    widths = [3, 1.8][: len(panels)]
    fig, axes = plt.subplots(1, len(panels), figsize=(11.5 if len(panels) > 1 else 8, 6.6),
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.2 if len(panels) > 1 else 0.3, right=0.97, wspace=0.7)
    for ax, (t, g) in zip(axes[0], panels):
        _range_dots(ax, g, lo, hi, max_ticks=7 if ax is axes[0][0] else 4)
        _panel_title(ax, t)
        ax.set_xlabel("Nightly price (COP, log scale)", fontsize=9.5)
    if loc is not None:
        tables["eda_price_by_locality.csv"] = loc
    if room is not None:
        tables["eda_price_by_room_type.csv"] = room

    first_t, first_g = panels[0]
    if first_t.startswith("Locality") and len(first_g) > 2:
        title = (f"Median price is {ratios[0]:.1f}\u00d7 higher in {first_g.index[0]} than in "
                 f"{first_g.index[-1]}, so location matters")
    else:
        title = "Nightly price by room type"
    _header(fig, title, "Dot = median, bar = middle 50% of listings. Localities with fewer than "
                        f"{MIN_GROUP_N} listings are left out.")
    _footer(fig, note)
    _save(fig, out, "00b_price_by_locality_room_type.png")
    return {"locality_median_ratio_top_to_bottom": float(ratios[0]) if ratios else None}


# ---- 00c price vs bedrooms and capacity ---------------------------------------------------
def _by_count(df: pd.DataFrame, col: str, cap: int) -> pd.DataFrame:
    s = df[[col, "price"]].dropna()
    s[col] = s[col].round().clip(lower=0, upper=cap).astype(int)
    g = s.groupby(col)["price"].agg(median="median", q25=lambda x: x.quantile(0.25),
                                    q75=lambda x: x.quantile(0.75), n="size")
    return g[g["n"] >= MIN_GROUP_N]


def fig_price_by_size(df: pd.DataFrame, out: Path, note: str, tables: dict) -> dict:
    specs = [("bedrooms", "Bedrooms", 5), ("accommodates", "Guests accommodated", 8)]
    specs = [s for s in specs if s[0] in df and df[s[0]].notna().sum() > 100]
    if not specs:
        return {}
    fig, axes = plt.subplots(1, len(specs), figsize=(11 if len(specs) > 1 else 6.5, 5.6), squeeze=False)
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.08, right=0.97, wspace=0.22)
    result = {}
    step = None
    small = False
    for ax, (col, label, cap) in zip(axes[0], specs):
        g = _by_count(df, col, cap)
        x = np.arange(len(g))
        ax.fill_between(x, g["q25"], g["q75"], color=SEQ[0], alpha=0.8, lw=0)
        ax.plot(x, g["median"], color=BLUE, lw=1.8, marker="o", ms=5.5, markeredgecolor="#fcfcfb")
        labels = [f"{i}{'+' if i == cap else ''}\nn={n:,}" for i, n in zip(g.index, g["n"])]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_formatter(COMPACT)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.set_xlabel(label, fontsize=9.5)
        ax.set_ylabel("Nightly price (COP)", fontsize=9.5)
        _panel_title(ax, f"By {label.lower()}")
        tables[f"eda_price_by_{col}.csv"] = g
        if col == "bedrooms" and {1, 3} <= set(g.index) and g.loc[[1, 3], "n"].min() >= 500:
            step = (g.loc[3, "median"] / g.loc[1, "median"]) ** (1 / 2) - 1   # only well-populated groups
            result["median_price_step_per_bedroom_pct_1_to_3"] = float(step * 100)
        if (g["n"] < 500).any():
            small = True

    if step is not None:
        title = f"Median price rises about {step * 100:.0f}% per extra bedroom from 1 to 3"
    else:
        title = "Price rises with the size of the listing"
    sub = "Line = median, band = middle 50% of listings; the last group is open-ended"
    if small:
        sub += ". Groups under 500 listings are noisy."
    _header(fig, title, sub)
    _footer(fig, note)
    _save(fig, out, "00c_price_by_size.png")
    return result


# ---- 00d missingness ----------------------------------------------------------------------
def fig_missingness(raw: pd.DataFrame, df: pd.DataFrame, out: Path, note: str, tables: dict) -> dict:
    share = raw.isna().mean() * 100
    # A column the cleaning step rebuilds (e.g. bathrooms from bathrooms_text) is not really missing.
    recovered = []
    for c in df.columns.intersection(raw.columns):
        clean = df[c].isna().mean() * 100
        if share[c] - clean > 5:
            share[c] = clean
            recovered.append(c)
    share = share.sort_values(ascending=False)
    tables["eda_missingness.csv"] = share.rename("missing_pct").to_frame()
    n_cols, n_any = len(share), int((share > 0).sum())
    empty = list(share.index[share >= 99.99])
    summary = {"columns": n_cols, "columns_with_missing": n_any, "columns_entirely_empty": len(empty),
               "empty_columns": empty}
    shown = share[(share > 1) & (share < 99.99)].head(18)   # partial gaps; empty columns are listed in text
    if shown.empty:
        return summary

    empty_txt = textwrap.wrap("Entirely empty, so dropped: " + ", ".join(empty), width=115) if empty else []
    foot = f"{note}  Bars: columns 1-99% missing (up to 18); all columns are in eda_missingness.csv."
    if recovered:
        foot += f"\n{', '.join(recovered)} is empty in the raw file but rebuilt from a related column (bathrooms_text)."
    h = 0.34 * len(shown) + 2.3 + 0.2 * len(empty_txt) + (0.2 if recovered else 0)
    fig, ax = plt.subplots(figsize=(9.5, h))
    fig.subplots_adjust(top=1 - 1.15 / h, bottom=(1.0 + 0.2 * len(empty_txt) + (0.2 if recovered else 0)) / h,
                        left=0.34, right=0.93)
    y = np.arange(len(shown))[::-1]
    ax.barh(y, shown.values, height=0.55, color=BLUE)
    for yi, v in zip(y, shown.values):
        ax.annotate(f"{v:.0f}%", (v, yi), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9.5, fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(shown.index, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.grid(axis="x")
    _pct_axis(ax, "x")
    ax.set_xlabel("Share of listings with the value missing", fontsize=9.5)
    if empty_txt:
        fig.text(0.04, (0.42 + (0.2 if recovered else 0)) / h, "\n".join(empty_txt), fontsize=8.5, color=INK2,
                 ha="left", va="bottom")

    title = f"{n_any} of {n_cols} columns have missing values"
    if empty:
        title += f"; {len(empty)} are entirely empty"
    _header(fig, title, "Partial gaps are imputed with the median and flagged with an indicator column; "
                        "empty columns are dropped")
    _footer(fig, foot)
    _save(fig, out, "00d_missing_values.png")
    return summary


# ---- 00e occupancy and activity -----------------------------------------------------------
def fig_occupancy(df: pd.DataFrame, out: Path, note: str, tables: dict) -> dict:
    if "occupancy" not in df:
        return {}
    occ = df["occupancy"].dropna()
    med = float(occ.median())
    has_active = "number_of_reviews_ltm" in df
    active = (df["number_of_reviews_ltm"].fillna(0) > 0) if has_active else None
    n_pan = 2 if has_active else 1

    fig, axes = plt.subplots(1, n_pan, figsize=(11 if has_active else 7.5, 5.4),
                             gridspec_kw={"width_ratios": [3, 2]} if has_active else None, squeeze=False)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.08 if has_active else 0.12, right=0.97, wspace=0.25)

    a1 = axes[0][0]
    a1.hist(occ * 100, bins=np.linspace(0, 100, 21), weights=np.full(len(occ), 100.0 / len(occ)),
            color=BLUE, edgecolor="#fcfcfb", linewidth=0.5)
    a1.axvline(med * 100, color=INK2, lw=0.9)
    a1.annotate(f"median {med * 100:.0f}%", (med * 100, a1.get_ylim()[1]), xytext=(6, -4),
                textcoords="offset points", ha="left", va="top", fontsize=9.5, color=INK2)
    a1.set_xlim(0, 100)
    _pct_axis(a1, "x")
    _pct_axis(a1, "y")
    a1.grid(axis="y")
    a1.set_xlabel("Occupancy proxy (share of the year)", fontsize=9.5)
    a1.set_ylabel("Share of listings per bar", fontsize=9.5)
    _panel_title(a1, "Occupancy proxy, all listings")

    result = {"median_occupancy_proxy": med, "share_occupancy_below_5pct": float((occ < 0.05).mean())}
    if has_active:
        a2 = axes[0][1]
        rows = [("All listings", occ), ("Active", df.loc[active, "occupancy"].dropna()),
                ("Inactive", df.loc[~active, "occupancy"].dropna())]
        vals = [float(s.median()) * 100 for _, s in rows]
        ns = [len(s) for _, s in rows]
        x = np.arange(3)
        a2.bar(x, vals, width=0.5, color=BLUE)
        for xi, v in zip(x, vals):
            a2.annotate(f"{v:.0f}%", (xi, v), xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=11, fontweight="bold", color=INK)
        a2.set_xticks(x)
        a2.set_xticklabels([f"{lab}\nn={n:,}" for (lab, _), n in zip(rows, ns)], fontsize=8.5)
        a2.tick_params(axis="x", length=0)
        a2.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        _pct_axis(a2)
        a2.grid(axis="y")
        a2.set_ylabel("Median occupancy proxy", fontsize=9.5)
        _panel_title(a2, "Median occupancy by activity")
        share_inactive = float((~active).mean())
        result.update(share_inactive_ltm=share_inactive, median_occ_active=vals[1] / 100,
                      median_occ_inactive=vals[2] / 100)
        title = (f"Median occupancy proxy is {med * 100:.0f}%; "
                 f"{share_inactive * 100:.0f}% of listings had no review in the past year")
        tables["eda_occupancy_by_activity.csv"] = pd.DataFrame(
            {"group": [r[0].split("\n")[0] for r in rows], "n": ns, "median_occupancy": np.array(vals) / 100})
    else:
        title = f"Median occupancy proxy is {med * 100:.0f}%"
    if "estimated_occupancy_l365d" in df:
        sub = "Estimated by Inside Airbnb from review counts (50% review rate, capped at 70%), not from bookings"
    else:
        sub = "A proxy from calendar availability, not confirmed bookings"
        if has_active:
            sub += ". Active = a review in the last 12 months"
    _header(fig, title, sub)
    mx = float(occ.max())
    at_cap = float((occ >= mx - 0.05).mean())
    foot = note
    lines = 1
    if has_active and "estimated_occupancy_l365d" in df:
        foot += "\nListings with no recent reviews show ~0% by construction, because the estimate is derived from reviews."
        lines += 1
    if mx < 0.9 and at_cap > 0.03:   # a pile-up at the top of a short range reads as a ceiling in the estimate
        foot += f"\nNothing exceeds {mx * 100:.0f}% and {at_cap * 100:.0f}% of listings sit in the top bar, so the estimate looks capped there."
        result["occupancy_proxy_max"] = mx
        lines += 1
    if lines > 1:
        fig.subplots_adjust(bottom=0.18 + 0.04 * lines)
    _footer(fig, foot)
    _save(fig, out, "00e_occupancy_and_activity.png")
    return result


# ---- 00f listings per host ----------------------------------------------------------------
def fig_host_concentration(df: pd.DataFrame, out: Path, note: str, tables: dict) -> dict:
    if "host_id" in df:
        size = df.groupby("host_id")["host_id"].transform("size")
        n_hosts = df["host_id"].nunique()
    elif "calculated_host_listings_count" in df:
        size = df["calculated_host_listings_count"].fillna(1).clip(lower=1)
        n_hosts = None
    else:
        return {}
    edges = [0, 1, 2, 5, 20, np.inf]
    labels = ["1 listing", "2", "3-5", "6-20", "21+"]
    b = pd.cut(size, edges, labels=labels)
    counts = b.value_counts().reindex(labels)
    share = counts / counts.sum() * 100
    if "host_id" in df:
        hosts = df.assign(_b=b).groupby("_b", observed=False)["host_id"].nunique().reindex(labels)
    else:
        hosts = pd.Series([np.nan] * len(labels), index=labels)
    tables["eda_host_sizes.csv"] = pd.DataFrame({"listings": counts, "share_pct": share, "hosts": hosts})

    multi = float(share.iloc[1:].sum())
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.96)
    x = np.arange(len(labels))
    ax.bar(x, share.values, width=0.5, color=[DEEMPH] + [BLUE] * (len(labels) - 1))
    for xi, v in zip(x, share.values):
        ax.annotate(f"{v:.0f}%", (xi, v), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold", color=INK)
    tick = [f"{lab}\n{int(h):,} hosts" if np.isfinite(h) else lab for lab, h in zip(labels, hosts)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick, fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, share.max() * 1.2)
    _pct_axis(ax)
    ax.grid(axis="y")
    ax.set_xlabel("Number of listings the host has in this file", fontsize=9.5)
    ax.set_ylabel("Share of all listings", fontsize=9.5)
    _header(fig, f"Hosts with 2+ listings own {multi:.0f}% of listings",
            "One host's units tend to resemble each other, so models are tested on whole held-out hosts")
    _footer(fig, note)
    _save(fig, out, "00f_listings_per_host.png")
    return {"share_listings_multi_host_pct": multi, "hosts": int(n_hosts) if n_hosts else None}


# ---- driver -------------------------------------------------------------------------------
def main(listings_path: str, reports: str = "reports", source: str = ""):
    rep = Path(reports)
    out = rep / "figures"
    out.mkdir(parents=True, exist_ok=True)
    note = f"Source: {source}. " if source else ""
    note += "Prices in COP per night."

    raw = data.load_listings(listings_path)
    raw_price = data._money_to_float(raw["price"])
    df = data.clean_listings(raw)
    try:
        df["occupancy"] = data.occupancy_proxy(df)
    except ValueError:
        print("[eda] no occupancy-like column; skipping occupancy figure")

    tables: dict[str, pd.DataFrame] = {}
    summary = {"listings_raw": int(len(raw)), "listings_after_trim": int(len(df))}
    summary.update(fig_price_distribution(raw_price, out, note))
    summary.update(fig_price_by_group(df, out, note, tables))
    summary.update(fig_price_by_size(df, out, note, tables))
    summary.update(fig_missingness(raw, df, out, note, tables))
    summary.update(fig_occupancy(df, out, note, tables))
    summary.update(fig_host_concentration(df, out, note, tables))

    for name, t in tables.items():
        t.to_csv(rep / name)
    (rep / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", rep / "eda_summary.json", "and", len(tables), "table files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listings", required=True)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--source", default="", help='e.g. "Inside Airbnb, Bogotá, scraped 2026-09"')
    a = ap.parse_args()
    main(a.listings, a.reports, a.source)
