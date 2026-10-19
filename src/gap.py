"""Price-gap analysis: how far is each listing's price from what comparable listings charge?

Fair price = out-of-fold random forest prediction, so a listing is never scored by a model that
saw it. When a host_id column exists the folds are grouped by host, so a host's near-identical
units in the same building cannot leak into each other's fair price and shrink the gaps.

The key check is whether gap relates to booking activity:
  * gap vs occupancy gets more negative -> pricier-than-peers listings book less (price response)
  * flat or positive -> the data cannot support a "raise/cut your price" claim; report that
Either way this is correlation across different listings, not an experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

SEED = 42
FLAG_PCT = 30.0  # keep this above typical model error (MAPE), or flags are mostly noise
BUCKET_EDGES = [-np.inf, -30, -15, 15, 30, np.inf]
BUCKET_LABELS = [
    "1 very underpriced (< -30%)",
    "2 underpriced (-30 to -15%)",
    "3 in line (-15 to +15%)",
    "4 overpriced (+15 to +30%)",
    "5 very overpriced (> +30%)",
]


def out_of_fold_fair_price(X: pd.DataFrame, price: pd.Series, groups=None, n_splits: int = 5) -> pd.Series:
    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, max_features=0.5, random_state=SEED, n_jobs=-1
    )
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        pred = cross_val_predict(rf, X, np.log(price.values), cv=cv, groups=np.asarray(groups))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        pred = cross_val_predict(rf, X, np.log(price.values), cv=cv)
    return pd.Series(np.exp(pred), index=X.index, name="fair_price")


def price_gap_table(df: pd.DataFrame, fair: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ["id", "name", "neighbourhood_cleansed", "room_type", "bedrooms", "accommodates",
              "latitude", "longitude"]:
        if c in df:
            out[c] = df[c]
    out["price"] = df["price"]
    out["fair_price"] = fair
    out["gap_pct"] = (df["price"] / fair - 1) * 100
    out["bucket"] = pd.cut(out["gap_pct"], BUCKET_EDGES, labels=BUCKET_LABELS)
    out["flag"] = np.select(
        [out["gap_pct"] >= FLAG_PCT, out["gap_pct"] <= -FLAG_PCT],
        ["review_overpriced", "review_underpriced"],
        default="in_line",
    )
    out["occupancy"] = df["occupancy"]
    if "neighbourhood_cleansed" in df:
        loc_median = df.groupby("neighbourhood_cleansed")["occupancy"].transform("median")
    else:
        loc_median = df["occupancy"].median()
    out["occ_vs_locality"] = df["occupancy"] - loc_median  # controls for locality-level demand
    if "reviews_per_month" in df:
        out["reviews_per_month"] = df["reviews_per_month"].fillna(0)  # no reviews = no activity
    if "number_of_reviews_ltm" in df:
        # Active = at least one review in the last 12 months. Separates live listings from ones that
        # are merely open all year, which would otherwise drag every bucket's occupancy toward zero.
        out["active_ltm"] = (df["number_of_reviews_ltm"].fillna(0) > 0).astype(int)
    return out


def summarise_gap(table: pd.DataFrame, price: pd.Series, fair: pd.Series):
    agg = {
        "listings": ("gap_pct", "size"),
        "median_price": ("price", "median"),
        "median_fair_price": ("fair_price", "median"),
        "median_occupancy": ("occupancy", "median"),
        "median_occ_vs_locality": ("occ_vs_locality", "median"),
    }
    if "reviews_per_month" in table:
        agg["median_reviews_per_month"] = ("reviews_per_month", "median")
    if "active_ltm" in table:
        agg["share_active"] = ("active_ltm", "mean")
    buckets = table.groupby("bucket", observed=False).agg(**agg)
    buckets["share_pct"] = buckets["listings"] / buckets["listings"].sum() * 100
    if "active_ltm" in table:
        buckets["share_active_ltm_pct"] = buckets.pop("share_active") * 100
        buckets["median_occupancy_active_only"] = (
            table[table["active_ltm"] == 1].groupby("bucket", observed=False)["occupancy"].median()
        )

    rho_occ = table["gap_pct"].corr(table["occupancy"], method="spearman")
    rho_loc = table["gap_pct"].corr(table["occ_vs_locality"], method="spearman")
    summary = {
        "listings": int(len(table)),
        "oof_MAPE_pct": float((np.abs(price - fair) / price).mean() * 100),
        "flag_threshold_pct": FLAG_PCT,
        "flagged_overpriced": int((table["flag"] == "review_overpriced").sum()),
        "flagged_underpriced": int((table["flag"] == "review_underpriced").sum()),
        "spearman_gap_vs_occupancy": float(rho_occ),
        "spearman_gap_vs_occupancy_within_locality": float(rho_loc),
    }
    summary.update(tail_shape(buckets))
    return buckets, summary


def tail_shape(buckets: pd.DataFrame, min_n: int = 100, drop: float = 0.8) -> dict:
    """Is occupancy lower at BOTH extremes than in the 'in line' bucket (a hump)?

    A rank correlation is monotonic, so it can read ~0 on a hump even when the pattern is strong.
    """
    occ = buckets["median_occupancy"].to_numpy(dtype=float)
    n = buckets["listings"].to_numpy()
    if len(occ) != 5 or not np.isfinite(occ[2]) or occ[2] <= 0:
        return {"inverted_u": False}
    lo_ratio, hi_ratio = occ[0] / occ[2], occ[4] / occ[2]
    enough = n[0] >= min_n and n[2] >= min_n and n[4] >= min_n
    out = {
        "very_underpriced_occupancy_vs_in_line": float(lo_ratio),
        "very_overpriced_occupancy_vs_in_line": float(hi_ratio),
        "inverted_u": bool(enough and lo_ratio < drop and hi_ratio < drop),
    }
    # Is the hump just inactive listings? Compare the share of listings with a review in the last 12
    # months, and occupancy among those active listings only, against the in-line bucket.
    if {"share_active_ltm_pct", "median_occupancy_active_only"} <= set(buckets.columns):
        sa = buckets["share_active_ltm_pct"].to_numpy(dtype=float)
        oa = buckets["median_occupancy_active_only"].to_numpy(dtype=float)
        if np.isfinite(oa[[0, 2, 4]]).all() and oa[2] > 0:
            out["active_share_gap_pts_underpriced"] = float(sa[2] - sa[0])
            out["active_share_gap_pts_overpriced"] = float(sa[2] - sa[4])
            out["active_occ_vs_in_line_underpriced"] = float(oa[0] / oa[2])
            out["active_occ_vs_in_line_overpriced"] = float(oa[4] / oa[2])
            out["tails_less_active"] = bool(enough and sa[2] - sa[0] >= 8 and sa[2] - sa[4] >= 8)
    return out


def interpret(summary: dict) -> str:
    r = summary["spearman_gap_vs_occupancy_within_locality"]
    if r <= -0.10:
        return ("Listings priced above peers tend to book less (within locality): consistent with a price "
                "response. Still correlation, not a causal estimate.")
    if summary.get("tails_less_active"):
        lo = summary["active_occ_vs_in_line_underpriced"] * 100
        hi = summary["active_occ_vs_in_line_overpriced"] * 100
        return ("Listings priced far from their peers are much less often active (a review in the last 12 months), "
                "which explains much of the raw occupancy hump. Among active listings, occupancy is "
                f"{lo:.0f}% of the in-line level for very underpriced and {hi:.0f}% for very overpriced listings. "
                "Treat flags as a review shortlist and keep claims to active listings; this is correlation, "
                "not a price effect.")
    if summary.get("inverted_u"):
        return (f"Listings priced far from their peers, in EITHER direction, show lower booking activity than "
                f"listings priced in line. The pattern is a hump, so the rank correlation ({r:+.2f}) understates "
                "it. Plausible reasons include quality the model cannot see (photos, condition) and stale or "
                "placeholder prices. Treat flags as a review shortlist, not proof that repricing would raise bookings.")
    if r >= 0.05:
        return ("Listings priced above peers do NOT book less (the relationship is flat or positive): the data "
                "does not support 'cut/raise price' revenue claims. Report the gaps as a benchmarking tool only.")
    return ("Weak relationship between price gap and occupancy: treat flags as benchmarking for review, "
            "not as evidence that repricing would change bookings.")
