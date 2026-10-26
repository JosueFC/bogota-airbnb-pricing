"""End-to-end run: python -m src.run --listings raw/listings.csv.gz --group-by host"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from . import data, features, gap, models, optimize


def main(
    listings_path: str,
    out_dir: str = "reports",
    group_by: str = "none",
    fast: bool = False,
    with_optimizer: bool = False,
):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    df = data.clean_listings(data.load_listings(listings_path))
    df["occupancy"] = data.occupancy_proxy(df)
    X = features.build_feature_matrix(df)
    y = df["log_price"]
    print(f"{len(df)} listings, {X.shape[1]} features")

    # 1) Price models: regression vs tree vs random forest on the same held-out set
    # 'host' keeps all of a host's listings on one side of the split (near-identical units otherwise
    # leak across it); 'neighbourhood' holds out whole localities.
    group_col = {"host": "host_id", "neighbourhood": "neighbourhood_cleansed"}.get(group_by)
    if group_col and group_col not in df:
        print(f"[run] WARNING: {group_col} not in file; falling back to a random split.")
        group_col = None
    groups = df[group_col] if group_col else None
    split_desc = f"grouped by {group_col}" if group_col else "random"
    print(f"Held-out split: {split_desc}")
    X_tr, X_te, y_tr, y_te = models.split(X, y, groups)

    fitted = models.fit_price_models(X_tr, y_tr, tune_forest=not fast)
    results = models.evaluate(fitted, X_te, y_te)
    results["split"] = split_desc
    print("\nPrice models (held-out):\n", results.to_string(index=False))
    results.to_csv(out / "price_model_comparison.csv", index=False)

    models.lasso_coefficients(fitted, X.columns).to_csv(out / "lasso_coefficients.csv")
    models.forest_importance(fitted, X.columns).to_csv(out / "forest_importance.csv")
    (out / "tree_rules.txt").write_text(models.tree_rules(fitted, X.columns))
    joblib.dump({"tree": fitted["tree_depth4"], "features": list(X.columns)}, out / "tree_depth4.joblib")

    # 2) Price-gap analysis (main business output)
    host_groups = df["host_id"] if "host_id" in df else None
    print(f"\nFair-price folds grouped by: {'host_id' if host_groups is not None else 'none (random)'}")
    fair = gap.out_of_fold_fair_price(X, df["price"], groups=host_groups)
    table = gap.price_gap_table(df, fair)
    buckets, gap_summary = gap.summarise_gap(table, df["price"], fair)
    print("\nOccupancy by price-gap bucket:\n", buckets.round(3).to_string())
    print("\nPrice-gap summary:", json.dumps(gap_summary, indent=2))
    print("\nReading:", gap.interpret(gap_summary))

    table.sort_values("gap_pct", key=abs, ascending=False).to_csv(out / "price_gap_listings.csv", index=False)
    buckets.to_csv(out / "price_gap_buckets.csv")
    (out / "price_gap_summary.json").write_text(json.dumps(gap_summary, indent=2))

    # 3) Optional, experimental: price-occupancy optimizer (see README for why it is off by default)
    if with_optimizer:
        rf_occ, cols, occ_mae = optimize.fit_occupancy_model(X, df["price"], df["occupancy"])
        rec = optimize.optimise_prices(rf_occ, cols, X, df["price"])
        summary = optimize.summarise(rec) | {"occupancy_model_MAE": occ_mae}
        print("\nOptimizer summary (experimental):", json.dumps(summary, indent=2))
        rec.assign(id=df.get("id")).to_csv(out / "recommendations.csv", index=False)
        (out / "optimizer_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listings", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--group-by", choices=["none", "host", "neighbourhood"], default="none",
                    help="hold out whole hosts or whole localities in the model comparison")
    ap.add_argument("--group-split", action="store_true", help="same as --group-by neighbourhood")
    ap.add_argument("--fast", action="store_true", help="skip forest hyperparameter search")
    ap.add_argument("--with-optimizer", action="store_true", help="also run the experimental optimizer")
    a = ap.parse_args()
    main(a.listings, a.out, "neighbourhood" if a.group_split else a.group_by, a.fast, a.with_optimizer)
