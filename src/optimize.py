"""Price-occupancy optimizer.

For each listing: predict occupancy at a grid of candidate prices (with all other
features fixed) and pick the price maximising expected revenue = price * occupancy.

IMPORTANT caveats (put these in the README):
  * Occupancy is a proxy (see data.occupancy_proxy), not observed bookings.
  * Observational data: the model learns correlations between price and occupancy
    across *different* listings, which is not the same as the causal effect of changing
    one listing's price. Treat uplift as a model-based estimate, and restrict candidates
    to a modest band around the current price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

SEED = 42
DEFAULT_MULTIPLIERS = np.round(np.arange(0.70, 1.31, 0.05), 2)

# Features that are consequences of past bookings (a booked listing gets reviews). Using them to
# predict occupancy is leakage and hides the price effect, so the occupancy model excludes them.
OCC_EXCLUDE = [
    "number_of_reviews", "reviews_per_month", "reviews_per_month_missing",
    "review_scores_rating_missing",
]


def fit_occupancy_model(X: pd.DataFrame, price: pd.Series, occ: pd.Series):
    """Random forest on features + log(price). Returns (model, columns, held-out MAE)."""
    Xo = X.drop(columns=[c for c in OCC_EXCLUDE if c in X.columns])
    Xo["log_price"] = np.log(price.values)
    idx = np.random.RandomState(SEED).permutation(len(Xo))
    cut = int(0.8 * len(Xo))
    tr, te = idx[:cut], idx[cut:]
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(Xo.iloc[tr], occ.iloc[tr])
    mae = mean_absolute_error(occ.iloc[te], rf.predict(Xo.iloc[te]))
    return rf, list(Xo.columns), mae


def optimise_prices(
    rf, columns, X: pd.DataFrame, price: pd.Series, multipliers=DEFAULT_MULTIPLIERS
) -> pd.DataFrame:
    n = len(X)
    best_rev = np.full(n, -np.inf)
    best_mult = np.ones(n)
    best_occ = np.zeros(n)
    base_occ = None

    for m in multipliers:
        Xc = X.copy()
        Xc["log_price"] = np.log(price.values * m)
        occ = rf.predict(Xc[columns])
        rev = price.values * m * occ
        if np.isclose(m, 1.0):
            base_occ = occ
        better = rev > best_rev
        best_rev[better] = rev[better]
        best_mult[better] = m
        best_occ[better] = occ[better]

    if base_occ is None:
        raise ValueError("multipliers must include 1.0 to compute the baseline")

    out = pd.DataFrame({
        "current_price": price.values,
        "recommended_price": price.values * best_mult,
        "price_change_pct": (best_mult - 1) * 100,
        "pred_occ_current": base_occ,
        "pred_occ_recommended": best_occ,
        "pred_rev_per_night_current": price.values * base_occ,
        "pred_rev_per_night_recommended": best_rev,
    }, index=X.index)
    out["pred_uplift_pct"] = (
        out["pred_rev_per_night_recommended"] / out["pred_rev_per_night_current"] - 1
    ) * 100
    return out


def summarise(rec: pd.DataFrame, lower_pct: float = -30.0, upper_pct: float = 30.0) -> dict:
    return {
        "share_flagged_underpriced_pct": float((rec["price_change_pct"] > 0).mean() * 100),
        "share_flagged_overpriced_pct": float((rec["price_change_pct"] < 0).mean() * 100),
        "median_uplift_pct": float(rec["pred_uplift_pct"].median()),
        "portfolio_uplift_pct": float(
            (rec["pred_rev_per_night_recommended"].sum() / rec["pred_rev_per_night_current"].sum() - 1) * 100
        ),
        # Diagnostic: if most listings sit at the edge of the price band, the occupancy model
        # has no usable price response and the "uplift" is an artifact. Do not report it.
        "share_at_upper_bound_pct": float((rec["price_change_pct"] >= upper_pct - 1e-6).mean() * 100),
        "share_at_lower_bound_pct": float((rec["price_change_pct"] <= lower_pct + 1e-6).mean() * 100),
    }
