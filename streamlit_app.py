"""Front-end. Run:  streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from features import CITY_COORDS, RAIN_MM_24H, TEST_DAYS

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
CITIES = list(CITY_COORDS)

st.set_page_config(page_title="City Weather ML", page_icon="🌦️")
st.title("🌦️ City weather: ML predictions")
st.caption("Random-forest models retrained on Open-Meteo history. A small ML demo, not an official forecast.")

city = st.selectbox("Choose a city", CITIES)


@st.cache_data(ttl=600, show_spinner="Fetching latest weather and predicting...")
def get_prediction(city):
    r = requests.get(f"{API_URL}/predict", params={"city": city}, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def get_metrics():
    r = requests.get(f"{API_URL}/metrics", timeout=15)
    r.raise_for_status()
    return r.json()


try:
    res = get_prediction(city)
except requests.RequestException:
    st.error("Could not get a prediction from the back-end. Please try again in a minute.")
    st.stop()

now_t, t6, p = res["temp_now_c"], res["temp_in_6h_c"], res["rain_likelihood_24h"]
as_of = pd.Timestamp(res["as_of"]).strftime("%d %b %Y, %H:%M")
st.subheader(res["city"])
st.caption(f"Latest data: {as_of} IST ({res['data_age_hours']} h old)")

c1, c2, c3 = st.columns(3)
c1.metric("Temperature now", f"{now_t} °C")
c2.metric("Predicted in 6 hours", f"{t6} °C", delta=f"{t6 - now_t:+.1f} °C", delta_color="off")
c3.metric("Rain likelihood (24 h)", f"{p:.0%}")
st.progress(min(max(p, 0.0), 1.0))
st.write("🌧️ Rain is likely in the next 24 hours." if res["rain_likely"]
         else "☀️ Rain is unlikely in the next 24 hours.")
st.caption(f"Rain means at least {RAIN_MM_24H:g} mm of total rainfall in the next 24 hours. "
           "The percentage is a model score, not a calibrated probability.")

recent = pd.DataFrame(res["recent"])
recent["time"] = pd.to_datetime(recent["time"])
recent = recent.set_index("time")
st.markdown("**Last 24 hours**")
temp_chart = alt.Chart(recent.reset_index()).mark_line().encode(
    x=alt.X("time:T", title=None),
    y=alt.Y("temperature_2m:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
)
st.altair_chart(temp_chart)
st.bar_chart(recent["rain"], y_label="Rain (mm/h)")

with st.expander("How good are these models?"):
    try:
        m = pd.DataFrame(get_metrics()).set_index("city")
        cols = {
            "temp_MAE_rf": "Model temp error (°C)",
            "temp_MAE_diurnal": "Same hour yesterday (°C)",
            "temp_MAE_baseline": "Same as now (°C)",
            "rain_F1_rf": "Rain F1: model",
            "rain_F1_baseline": "Rain F1: persistence",
            "rain_PR_AUC_rf": "Rain PR-AUC: model",
            "rain_PR_AUC_chance": "Rain PR-AUC: chance",
        }
        shown = m[[c for c in cols if c in m.columns]].rename(columns=cols)
        st.dataframe(shown.loc[[city]].T)
        st.caption("Lower is better for temperature error, higher for rain scores. "
                   f"Tested on the last {TEST_DAYS} days of history.")
    except requests.RequestException:
        st.write("Metrics are not available right now.")
