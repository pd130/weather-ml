"""Shared feature engineering and constants.

Used by train.py, predict.py and streamlit_app.py, so training, serving and the UI
always agree on the same features, cities and thresholds.
"""
import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"

CITY_COORDS = {
    "Pune": (18.5344, 73.7314),
    "Mumbai": (19.0761, 72.8774),
    "Delhi": (28.61394, 77.20902),
}

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "rain", "weather_code", "wind_speed_10m", "wind_direction_10m",
    "pressure_msl", "cloud_cover",
]

TEMP_HORIZON = 6        # predict temperature 6 hours ahead
RAIN_HORIZON = 24       # predict rain over the next 24 hours
RAIN_MM_24H = 1.0       # total mm in the next 24h that counts as a rainy day
RAIN_CUTOFF = 0.5       # model score at or above this -> "rain likely"
MIN_HISTORY_HOURS = 25  # longest look-back of any feature (lag 24 / 24h window)
TEST_DAYS = 180         # last N days of each city are held out as the test set

# not model inputs: identifiers, raw angle/categorical columns, targets
NON_FEATURES = ["date", "city", "weather_code", "wind_direction_10m", "temp_next", "rain_next"]


def future_rain_total(s, horizon=RAIN_HORIZON, mm=RAIN_MM_24H):
    """1.0 if total rain over the next `horizon` hours (t+1 ... t+horizon) exceeds mm."""
    tot = s.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return (tot > mm).astype(float).where(tot.notna())


def build_features(data, with_targets=True):
    """data: one row per (city, hour) with HOURLY_VARS plus 'date' and 'city'.

    with_targets=False is for serving: the future is unknown, so no targets.
    """
    data = data.sort_values(["city", "date"]).reset_index(drop=True)
    g = data.groupby("city")

    # circular wind direction + wet flag (weather_code 51+ = drizzle/rain/storm)
    data["wind_dir_sin"] = np.sin(np.deg2rad(data["wind_direction_10m"]))
    data["wind_dir_cos"] = np.cos(np.deg2rad(data["wind_direction_10m"]))
    data["wet"] = (data["weather_code"].astype(int) >= 51).astype(int)

    # time features
    data["hour_sin"] = np.sin(2 * np.pi * data["date"].dt.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["date"].dt.hour / 24)
    doy = data["date"].dt.dayofyear
    data["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    data["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # lags (lag 18 doubles as the "same hour yesterday" baseline)
    lag_cols = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "apparent_temperature", "rain", "wind_speed_10m", "wet",
                "pressure_msl", "cloud_cover"]
    for lag in [1, 3, 6, 12, 18, 24]:
        for c in lag_cols:
            data[f"{c}_lag{lag}"] = g[c].shift(lag)

    # rolling totals / means
    for w in [6, 24]:
        data[f"rain_sum{w}"] = g["rain"].transform(lambda s, w=w: s.rolling(w).sum())
    data["temp_mean24"] = g["temperature_2m"].transform(lambda s: s.rolling(24).mean())
    data["humidity_mean24"] = g["relative_humidity_2m"].transform(lambda s: s.rolling(24).mean())
    data["dew_spread"] = data["temperature_2m"] - data["dew_point_2m"]
    for c in ["pressure_msl", "cloud_cover", "relative_humidity_2m"]:
        for n in [3, 6]:
            data[f"{c}_chg{n}"] = data[c] - g[c].shift(n)

    if with_targets:
        data["temp_next"] = g["temperature_2m"].shift(-TEMP_HORIZON)   # temp at t+6h
        data["rain_next"] = g["rain"].transform(future_rain_total)      # >= 1 mm in next 24h
    return data


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]
