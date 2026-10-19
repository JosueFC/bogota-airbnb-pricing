"""Models: regression (Ridge/Lasso), decision tree, random forest. Evaluated on the same split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

SEED = 42


def split(X, y, groups=None, test_size=0.2):
    """Random split, or a grouped split (e.g. by neighbourhood) to test generalisation."""
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        tr, te = next(gss.split(X, y, groups))
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    return train_test_split(X, y, test_size=test_size, random_state=SEED)


def fit_price_models(X_tr: pd.DataFrame, y_tr_log: pd.Series, tune_forest: bool = True) -> dict:
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED, max_iter=20000)),
        "tree_depth4": DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=SEED),
    }
    if tune_forest:
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 12, 20],
                "min_samples_leaf": [1, 3, 5],
                "max_features": ["sqrt", 0.5, 1.0],
            },
            n_iter=8, cv=3, scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1,
        )
        search.fit(X_tr, y_tr_log)
        models["random_forest"] = search.best_estimator_
    else:
        models["random_forest"] = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ).fit(X_tr, y_tr_log)

    for name in ("ridge", "lasso", "tree_depth4"):
        models[name].fit(X_tr, y_tr_log)
    return models


def evaluate(models: dict, X_te: pd.DataFrame, y_te_log: pd.Series) -> pd.DataFrame:
    """Report error in dollars and percent (back-transformed), not only R^2."""
    y_true = np.exp(y_te_log)
    rows = []
    for name, m in models.items():
        pred = np.exp(m.predict(X_te))
        rows.append({
            "model": name,
            "MAE_local_currency": mean_absolute_error(y_true, pred),  # COP for Bogota
            "MAPE_pct": float(np.mean(np.abs(pred - y_true) / y_true) * 100),
            "R2_log_price": r2_score(y_te_log, np.log(pred)),
        })
    return pd.DataFrame(rows).sort_values("MAE_local_currency").reset_index(drop=True)


def lasso_coefficients(models: dict, columns) -> pd.Series:
    """Coefficients on standardised features: effect on log-price per 1 SD change."""
    coefs = models["lasso"][-1].coef_
    return pd.Series(coefs, index=columns).loc[lambda s: s != 0].sort_values(key=abs, ascending=False)


def tree_rules(models: dict, columns) -> str:
    return export_text(models["tree_depth4"], feature_names=list(columns))


def forest_importance(models: dict, columns, top=15) -> pd.Series:
    return pd.Series(models["random_forest"].feature_importances_, index=columns).nlargest(top)
