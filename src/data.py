"""Load and clean an Inside Airbnb-style listings file (Bogota, Colombia)."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

PRICE_CAP_QUANTILE = 0.99  # drop absurd outliers above this quantile
PRICE_FLOOR_QUANTILE = 0.01  # and below this one; quantiles work in any currency (COP, USD...)


def _money_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def _pct_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce") / 100.0


def load_listings(path: str) -> pd.DataFrame:
    """Read listings.csv or listings.csv.gz. Schema differs slightly across scrapes,
    so downstream code only uses columns that exist."""
    return pd.read_csv(path, low_memory=False)


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # price
    df["price"] = _money_to_float(df["price"])
    # Inside Airbnb quotes prices in the local currency for non-US cities (COP for Bogota),
    # so use quantile trimming rather than fixed dollar thresholds.
    df = df[df["price"].notna() & (df["price"] > 0)]
    lo, hi = df["price"].quantile([PRICE_FLOOR_QUANTILE, PRICE_CAP_QUANTILE])
    df = df[(df["price"] >= lo) & (df["price"] <= hi)]
    df["log_price"] = np.log(df["price"])

    # host fields
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df:
            df[col] = _pct_to_float(df[col])
    if "host_is_superhost" in df:
        df["host_is_superhost"] = df["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0)
    if "host_since" in df:
        since = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_years"] = (pd.Timestamp.today() - since).dt.days / 365.25

    # bathrooms: older scrapes have bathrooms_text ("1.5 baths"), newer have numeric
    if "bathrooms" not in df or df["bathrooms"].isna().all():
        if "bathrooms_text" in df:
            df["bathrooms"] = pd.to_numeric(
                df["bathrooms_text"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
            )

    # amenities -> count + a few key flags
    if "amenities" in df:
        am = df["amenities"].fillna("").str.lower()
        df["n_amenities"] = am.str.count(",") + 1
        for name, pat in {
            "has_pool": r"pool",
            "has_ac": r"air conditioning|\bac\b",
            "has_wifi": r"wifi|wi-fi",
            "has_parking": r"parking",
            "has_kitchen": r"kitchen",
            "has_beachfront": r"beach access|beachfront|waterfront",
        }.items():
            df[name] = am.str.contains(pat, regex=True).astype(int)

    return df.reset_index(drop=True)


def occupancy_proxy(df: pd.DataFrame) -> pd.Series:
    """Occupancy proxy in [0, 1].

    Neither signal is observed bookings, so use the best available one and say so in the README:
      1. estimated_occupancy_l365d / 365 if the scrape provides it. Inside Airbnb derives it from
         REVIEW counts (50% review rate x average stay, capped at 70%), so a listing with no recent
         reviews is ~0% by construction.
      2. otherwise 1 - availability_365 / 365 (calendar availability; hosts also block nights)
    """
    if "estimated_occupancy_l365d" in df:
        return (df["estimated_occupancy_l365d"] / 365.0).clip(0, 1)
    if "availability_365" in df:
        return (1 - df["availability_365"] / 365.0).clip(0, 1)
    raise ValueError("No occupancy-like column found (estimated_occupancy_l365d / availability_365).")
