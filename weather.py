"""Fetch the most recent hours of weather for one city (used when serving)."""
import pandas as pd
import requests

from features import CITY_COORDS, HOURLY_VARS, TZ

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_recent(city, past_days=3):
    """Last ~3 days of hourly data up to the current hour, same columns/units as training.

    The forecast API also returns the rest of today as forecast values, so rows after
    the current hour are dropped: the models may only see what is known right now.
    """
    lat, lon = CITY_COORDS[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TZ,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=15)
    r.raise_for_status()

    h = pd.DataFrame(r.json()["hourly"])
    h["date"] = pd.to_datetime(h.pop("time")).dt.tz_localize(TZ)
    h = h[h["date"] <= pd.Timestamp.now(tz=TZ).floor("h")]     # drop future hours
    h = h.dropna(subset=HOURLY_VARS).reset_index(drop=True)     # newest complete hour last
    h.insert(1, "city", city)
    return h
