"""Back-end API.

Run (bind to this VM's PRIVATE IP so only the front-end VM can reach it):
    uvicorn api:app --host <BACKEND_PRIVATE_IP> --port 8000
For local testing use --host 127.0.0.1.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from enum import Enum

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

import predict as P
from features import CITY_COORDS

# allowed cities as an Enum: anything else gets HTTP 422 automatically
City = Enum("City", {c: c for c in CITY_COORDS}, type=str)

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    try:
        P.warm_up()
        log.info("All models loaded")
    except Exception as e:      # keep the API up so /health still answers
        log.error("Model warm-up failed: %s", e)
    yield


app = FastAPI(title="Weather ML API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(city: City):
    try:
        return P.predict_city(city.value)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model files not found on the server")
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Upstream weather service unavailable")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
def metrics():
    """Evaluation table written by train.py (model vs baselines, per city)."""
    path = os.path.join(P.MODEL_DIR, "metrics.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="metrics.csv not found")
    df = pd.read_csv(path, index_col=0)
    df.index.name = "city"
    return json.loads(df.reset_index().to_json(orient="records"))   # NaN -> null
