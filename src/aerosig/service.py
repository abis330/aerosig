"""Minimal REST scoring service: POST a flight id, get anomaly events + attribution."""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from flightlake import Lake

from .score import Detector

app = FastAPI(title="aerosig", description="Self-supervised fault detection for flight telemetry")


@lru_cache(maxsize=1)
def _det() -> Detector:
    return Detector.load(os.environ.get("AEROSIG_ARTIFACTS", "artifacts"))


@lru_cache(maxsize=1)
def _lake() -> Lake:
    return Lake(os.environ.get("FLIGHTLAKE_ROOT", "data/lake"))


@app.get("/health")
def health():
    return {"status": "ok", "threshold": _det().calib["threshold"]}


@app.get("/score/{flight_id}")
def score(flight_id: str):
    try:
        tel = _lake().telemetry(flight_id)
    except KeyError:
        raise HTTPException(404, f"unknown flight {flight_id}")
    det = _det()
    s = det.score_flight(tel)
    evs = det.events(s)
    for e in evs:
        e.pop("profile"), e.pop("window_rows")
    return {"flight_id": flight_id, "threshold": det.calib["threshold"], "events": evs,
            "max_score": float(s["score"].max()) if len(s) else None}
