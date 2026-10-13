
import json
import os

import joblib
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from features import (RAIN_HORIZON, RAIN_MM_24H, TEMP_HORIZON, TEST_DAYS,
                      build_features, feature_columns)


# Config

CITIES = ["Pune", "Mumbai", "Delhi"]       # same order as the latitude/longitude lists below

GAP = max(TEMP_HORIZON, RAIN_HORIZON)      # hours dropped between train and test (target overlap)
TEST_HOURS = TEST_DAYS * 24
N_ITER_RANDOM = 20                         # RandomizedSearchCV candidates
CV_SPLITS = 3

HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
          "apparent_temperature", "rain", "weather_code",
          "wind_speed_10m", "wind_direction_10m",
          "pressure_msl", "cloud_cover"]

# 1. Download data

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
openmeteo = openmeteo_requests.Client(session=retry(cache_session, retries=5, backoff_factor=0.2))

params = {
    "latitude": [18.5344, 19.0761, 28.61394],
    "longitude": [73.7314, 72.8774, 77.20902],
    "start_date": "2024-01-01",
    "end_date": "2026-10-03",
    "hourly": HOURLY,
    "timezone": "Asia/Kolkata",
}
responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)

frames = []
for city, resp in zip(CITIES, responses):
    h = resp.Hourly()
    df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(h.Time(), unit="s", utc=True),
            end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=h.Interval()),
            inclusive="left",
        ).tz_convert("Asia/Kolkata"),
        "city": city,
        **{name: h.Variables(i).ValuesAsNumpy() for i, name in enumerate(HOURLY)},
    })
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
data["weather_code"] = data["weather_code"].astype("Int64")
print(data.shape)
print(data.groupby("city").size())
print(data.isna().sum())


# 2. Features and targets (shared code in features.py)

data = build_features(data)

FEATURES = feature_columns(data)
data = data.dropna(subset=FEATURES + ["temp_next", "rain_next"]).reset_index(drop=True)
print(data.groupby("city")["rain_next"].mean())

os.makedirs("models", exist_ok=True)
with open("models/features.json", "w") as f:
    json.dump(FEATURES, f)

# 3. Hyperparameter search (RandomizedSearchCV, time-aware CV)

REG_SPACE = {
    "n_estimators": [100],
    "max_depth": [10, 20, 30, None],
    "min_samples_leaf": [5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features": [0.2, 0.3, 0.4, 0.6, 0.8],
}
CLF_SPACE = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [10, 20, 30, None],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.2, 0.3, 0.4],
}


def tune(base_estimator, space, X, y, scoring):
    """Returns (best params, best CV score). The gap keeps validation folds clear of train targets."""
    cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=GAP)
    rs = RandomizedSearchCV(base_estimator, space, n_iter=N_ITER_RANDOM, scoring=scoring,
                            cv=cv, n_jobs=-1, random_state=42, refit=False)
    rs.fit(X, y)
    return rs.best_params_, rs.best_score_


# 4. Train / evaluate / save per city

results, all_params = {}, {}

for city, d in data.groupby("city"):
    d = d.reset_index(drop=True)
    test_start = len(d) - TEST_HOURS
    train_end = test_start - GAP

    X_tr, X_te = d[FEATURES].iloc[:train_end], d[FEATURES].iloc[test_start:]

    # ---- regressor: temperature 6 hours ahead ----
    y_tr, y_te = d["temp_next"].iloc[:train_end], d["temp_next"].iloc[test_start:]
    reg_params, reg_cv = tune(RandomForestRegressor(n_jobs=1, random_state=42),
                              REG_SPACE, X_tr, y_tr, scoring="neg_mean_absolute_error")
    reg = RandomForestRegressor(**reg_params, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    mae_rf = mean_absolute_error(y_te, reg.predict(X_te))
    mae_base = mean_absolute_error(y_te, X_te["temperature_2m"])            # "same as now"
    mae_diurnal = mean_absolute_error(y_te, X_te["temperature_2m_lag18"])   # same hour yesterday

    # ---- classifier: >= 1 mm rain in the next 24 hours ----
    c_tr, c_te = d["rain_next"].iloc[:train_end], d["rain_next"].iloc[test_start:]
    clf_base = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=42)
    clf_params, clf_cv = tune(clf_base, CLF_SPACE, X_tr, c_tr, scoring="average_precision")
    clf = RandomForestClassifier(**clf_params, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=42).fit(X_tr, c_tr)
    p = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    base_p = (X_te["rain_sum24"] > RAIN_MM_24H).astype(int)   # "rained in last 24h -> rains in next 24h"

    results[city] = {
        "temp_MAE_rf": round(mae_rf, 3),
        "temp_MAE_baseline": round(mae_base, 3),
        "rain_F1_rf": round(f1_score(c_te, p), 3),
        "rain_F1_baseline": round(f1_score(c_te, base_p), 3),
        "rain_recall_rf": round(recall_score(c_te, p), 3),
        "rain_precision_rf": round(precision_score(c_te, p, zero_division=0), 3),
        "rain_ROC_AUC_rf": round(roc_auc_score(c_te, proba), 3) if c_te.nunique() > 1 else np.nan,
        "rain_rate_test": round(float(c_te.mean()), 3),
        "cv_temp_MAE": round(-reg_cv, 3),
        "cv_rain_AP": round(clf_cv, 3),
        "temp_MAE_diurnal": round(mae_diurnal, 3),
        "rain_F1_always": round(f1_score(c_te, np.ones(len(c_te))), 3),
        "rain_PR_AUC_rf": round(average_precision_score(c_te, proba), 3),
        "rain_PR_AUC_chance": round(float(c_te.mean()), 3),
    }
    all_params[city] = {"regressor": reg_params, "classifier": clf_params}

    # refit on ALL data with the tuned params for deployment, then save
    reg = RandomForestRegressor(**reg_params, n_jobs=-1, random_state=42).fit(d[FEATURES], d["temp_next"])
    clf = RandomForestClassifier(**clf_params, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=42).fit(d[FEATURES], d["rain_next"])
    os.makedirs(f"models/{city}", exist_ok=True)
    joblib.dump(reg, f"models/{city}/regressor_6h.joblib")
    joblib.dump(clf, f"models/{city}/classifier_24h.joblib")
    print(f"{city}: done | best reg params {reg_params} | best clf params {clf_params}")

metrics = pd.DataFrame(results).T
print(metrics)
metrics.to_csv("models/metrics.csv")
with open("models/best_params.json", "w") as f:
    json.dump(all_params, f, indent=2, default=str)
