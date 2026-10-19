"""Feature engineering for the pricing model."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate coordinates for reference points in Bogota, Colombia
REFERENCE_POINTS = {
    "zona_t": (4.6677, -74.0545),          # Zona Rosa / Zona T, Chapinero
    "la_candelaria": (4.5981, -74.0758),   # historic centre
    "usaquen": (4.6950, -74.0300),         # Usaquen plaza
    "el_dorado_airport": (4.7016, -74.1469),
}

NUMERIC_CANDIDATES = [
    "accommodates", "bedrooms", "beds", "bathrooms", "n_amenities",
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "review_scores_location", "review_scores_cleanliness",
    "host_years", "host_response_rate", "host_is_superhost",
    "calculated_host_listings_count",
    "has_pool", "has_ac", "has_wifi", "has_parking", "has_kitchen", "has_beachfront",
]
CATEGORICAL_CANDIDATES = ["room_type", "neighbourhood_cleansed", "property_type"]


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"latitude", "longitude"} <= set(df.columns):
        for name, (lat, lon) in REFERENCE_POINTS.items():
            df[f"km_to_{name}"] = _haversine_km(df["latitude"], df["longitude"], lat, lon)
    return df


def build_feature_matrix(df: pd.DataFrame, top_k_categories: int = 15) -> pd.DataFrame:
    """Numeric columns (median-imputed, with missing indicators for the noisy ones)
    plus one-hot categoricals with rare levels grouped as 'other'."""
    df = add_geo_features(df)

    absent = [c for c in NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES if c not in df.columns]
    if absent:
        print(f"[features] WARNING: expected columns not in file (skipped): {absent}")
        print("[features] If bedrooms/accommodates/amenities are listed, you probably loaded the "
              "summary listings.csv instead of the detailed listings.csv.gz.")

    num_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    num_cols += [c for c in df.columns if c.startswith("km_to_")]

    X = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Columns that exist but are entirely empty (happens for some cities/scrapes) have a NaN
    # median, so imputation cannot fill them and scikit-learn rejects the matrix. Drop them.
    empty = [c for c in X.columns if X[c].isna().all()]
    if empty:
        print(f"[features] WARNING: columns present but entirely empty (dropped): {empty}")
        X = X.drop(columns=empty)

    for c in ["review_scores_rating", "reviews_per_month", "host_response_rate"]:
        if c in X:
            X[f"{c}_missing"] = X[c].isna().astype(int)
    X = X.fillna(X.median(numeric_only=True))

    for c in [c for c in CATEGORICAL_CANDIDATES if c in df.columns]:
        s = df[c].fillna("unknown").astype(str)
        keep = s.value_counts().nlargest(top_k_categories).index
        s = s.where(s.isin(keep), "other")
        X = pd.concat([X, pd.get_dummies(s, prefix=c, drop_first=True).astype(int)], axis=1)

    # Final guard: nothing downstream accepts NaN.
    assert not X.isna().any().any(), "NaN left in feature matrix"
    return X
