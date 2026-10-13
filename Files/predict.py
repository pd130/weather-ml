"""Load the trained models and produce a prediction for one city."""
import json
import os
import time
from functools import lru_cache

import joblib
import pandas as pd

from features import CITY_COORDS, MIN_HISTORY_HOURS, RAIN_CUTOFF, TZ, build_features
from weather import fetch_recent

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
CACHE_SECONDS = 600     # reuse a result for 10 min: faster, and spares the upstream API
_cache = {}


@lru_cache(maxsize=None)
def _load_models(city):
    reg = joblib.load(os.path.join(MODEL_DIR, city, "regressor_6h.joblib"))
    clf = joblib.load(os.path.join(MODEL_DIR, city, "classifier_24h.joblib"))
    return reg, clf


@lru_cache(maxsize=1)
def _feature_names():
    with open(os.path.join(MODEL_DIR, "features.json")) as f:
        return json.load(f)


def warm_up():
    """Load the feature list and every model now, so the first request isn't slow."""
    _feature_names()
    for city in CITY_COORDS:
        _load_models(city)


def predict_city(city):
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city: {city}")

    hit = _cache.get(city)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    raw = fetch_recent(city)
    if len(raw) < MIN_HISTORY_HOURS + 1:
        raise RuntimeError("Not enough recent weather data to build features")

    feats = build_features(raw, with_targets=False)
    cols = _feature_names()                 # same columns, same order as training
    last = feats.iloc[[-1]][cols]           # newest hour = "now"
    if last.isna().any().any():
        raise RuntimeError("Latest weather row has missing values")

    reg, clf = _load_models(city)
    temp_now = float(last["temperature_2m"].iloc[0])
    temp_6h = float(reg.predict(last)[0])
    p_rain = float(clf.predict_proba(last)[0, 1])

    as_of = feats["date"].iloc[-1]
    age_h = (pd.Timestamp.now(tz=TZ) - as_of).total_seconds() / 3600
    recent = raw.tail(24)

    result = {
        "city": city,
        "as_of": as_of.isoformat(),
        "data_age_hours": round(age_h, 1),
        "temp_now_c": round(temp_now, 1),
        "temp_in_6h_c": round(temp_6h, 1),
        "rain_likelihood_24h": round(p_rain, 3),
        "rain_likely": p_rain >= RAIN_CUTOFF,
        "recent": [
            {"time": t.isoformat(), "temperature_2m": round(float(x), 1), "rain": round(float(r), 2)}
            for t, x, r in zip(recent["date"], recent["temperature_2m"], recent["rain"])
        ],
    }
    _cache[city] = (time.time(), result)
    return result
