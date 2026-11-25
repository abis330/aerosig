"""``aerosig`` command line."""
from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, help="Self-supervised flight fault detection & signatures")


@app.command()
def train(lake: str = "data/lake", out: str = "artifacts", epochs: int = 8, time_budget_min: float = 30,
          seed: int = 0):
    """Pre-train on nominal flights and calibrate the alarm threshold."""
    from .train import train as _train

    _train(lake, out, epochs=epochs, time_budget_min=time_budget_min, seed=seed)


@app.command()
def evaluate(lake: str = "data/lake", art: str = "artifacts", baselines: bool = True):
    """Event-level evaluation vs injected-fault ground truth (+ z-score / IsolationForest baselines)."""
    from .evaluate import evaluate as _ev

    _ev(lake, art, baselines)


@app.command()
def signatures(lake: str = "data/lake", art: str = "artifacts"):
    """Build the fault-signature library and evaluate signature matching."""
    from .signatures import run

    run(lake, art)


@app.command()
def score(flight_id: str, lake: str = "data/lake", art: str = "artifacts"):
    """Score one flight and print detected events with channel attribution."""
    from flightlake import Lake

    from .score import Detector

    det = Detector.load(art)
    s = det.score_flight(Lake(lake).telemetry(flight_id))
    for e in det.events(s):
        e.pop("profile"), e.pop("window_rows")
        typer.echo(json.dumps(e))


@app.command("export-onnx")
def export_onnx(art: str = "artifacts"):
    from .export import export_onnx as _ex

    typer.echo(str(_ex(art)))


@app.command()
def serve(lake: str = "data/lake", art: str = "artifacts", port: int = 8001):
    """REST scoring service."""
    import os

    import uvicorn

    os.environ.update(FLIGHTLAKE_ROOT=lake, AEROSIG_ARTIFACTS=art)
    uvicorn.run("aerosig.service:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    app()
