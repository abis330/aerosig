"""Event-level evaluation against injected-fault ground truth + baselines.

Metrics (what a flight-test / safety engineer actually cares about):
  * fault recall (per fault type)  - was the fault flagged at all while it was active?
  * detection latency              - streaming: time from fault onset to the decision window end
  * false alarms per flight hour   - on held-out nominal flights AND pre-onset parts of faulty flights
  * event precision                - fraction of raised events that correspond to a real fault
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flightlake import Lake
from sklearn.ensemble import IsolationForest

from .data import FEATURES, Normalizer, eligible_windows, fault_intervals, features, windows
from .score import Detector, extract_events, window_score


def match_events(events: list[dict], faults: list[tuple], win_s: float) -> tuple[list[dict], list[dict]]:
    """Returns (per-fault results, false-alarm events)."""
    used, results = set(), []
    for ftype, fs, fe in faults:
        hit = None
        for k, e in enumerate(events):
            if fs <= e["t_detect"] <= fe + win_s:
                hit = k
                break
        if hit is not None:
            used.add(hit)
            # any further events inside the same fault interval are the same fault, not false alarms
            used.update(k for k, e in enumerate(events) if fs <= e["t_detect"] <= fe + win_s)
        results.append({"fault": ftype, "onset": fs, "detected": hit is not None,
                        "latency_s": events[hit]["t_detect"] - fs if hit is not None else None,
                        "event": events[hit] if hit is not None else None})
    false = [e for k, e in enumerate(events) if k not in used]
    return results, false


class _Baseline:
    """Interface-compatible baselines that reuse the same windowing/event logic."""

    def __init__(self, kind: str, norm: Normalizer, win: int, stride: int):
        self.kind, self.norm, self.win, self.stride = kind, norm, win, stride

    def _feat(self, W):
        return np.concatenate([W.mean(axis=2), W.std(axis=2)], axis=1)

    def fit(self, lake, train_ids, val_ids, q):
        Wt = np.concatenate([windows(self.norm(features(lake.telemetry(f))), self.win, self.stride)[0]
                             for f in train_ids])
        if self.kind == "iforest":
            self.m = IsolationForest(n_estimators=200, random_state=0).fit(self._feat(Wt))
        s = []
        for f in val_ids:
            tel = lake.telemetry(f)
            W, end = windows(self.norm(features(tel)), self.win, self.stride)
            s.append(self._scores(W[eligible_windows(tel, end, self.win)]))
        s = np.concatenate(s)
        self.threshold = float(np.quantile(s, q))
        return self

    def _scores(self, W):
        if self.kind == "zscore":  # robust z of window means vs fleet-normal statistics
            return window_score(np.abs(W.mean(axis=2)))
        return -self.m.score_samples(self._feat(W))

    def score_flight(self, tel):
        W, end = windows(self.norm(features(tel)), self.win, self.stride)
        s = self._scores(W)
        return pd.DataFrame({"t_end": tel.t_s.to_numpy()[end], "t_start": tel.t_s.to_numpy()[end - self.win + 1],
                             "score": s, "eligible": eligible_windows(tel, end, self.win),
                             "phase": tel.phase_est.to_numpy()[end]})


def run_eval(lake: Lake, scorer, flights_faulty, flights_nominal, threshold, min_consecutive, win_s,
             detector: Detector | None = None):
    rows, fa_events, hours = [], 0, 0.0
    per_flight = {}
    for fid in list(flights_faulty) + list(flights_nominal):
        tel = lake.telemetry(fid)
        s = scorer.score_flight(tel)
        evs = extract_events(s, threshold, min_consecutive)
        meta = lake.summary(fid)["meta"]
        faults = fault_intervals(meta.get("faults", []), float(tel.t_s.max()))
        res, false = match_events(evs, faults, win_s)
        # false-alarm exposure: whole nominal flights + pre-onset part of faulty flights
        exposure = float(tel.t_s.max()) if not faults else min(f[1] for f in faults)
        hours += exposure / 3600
        fa_events += sum(e["t_detect"] < exposure for e in false) if faults else len(false)
        for r in res:
            r["flight_id"] = fid
            rows.append(r)
        per_flight[fid] = {"events": evs, "results": res}
    df = pd.DataFrame(rows)
    by_type = df.groupby("fault").agg(n=("detected", "size"), recall=("detected", "mean"),
                                      median_latency_s=("latency_s", "median")).round(3)
    n_events_true = int(df.detected.sum())
    summary = {
        "faults": int(len(df)), "recall": round(float(df.detected.mean()), 3),
        "median_latency_s": round(float(df.latency_s.median()), 2) if df.detected.any() else None,
        "false_alarms": int(fa_events), "false_alarms_per_hour": round(fa_events / max(hours, 1e-9), 3),
        "event_precision": round(n_events_true / max(n_events_true + fa_events, 1), 3),
        "exposure_hours": round(hours, 2),
    }
    return summary, by_type, per_flight


def evaluate(lake_root: str, art_dir: str = "artifacts", baselines: bool = True) -> dict:
    lake, art = Lake(lake_root), Path(art_dir)
    det = Detector.load(art)
    split = json.loads((art / "split.json").read_text())
    cfg = det.cfg
    win_s = cfg["win"] / 10.0
    faulty, nominal = split["test_faulty"], split["test_nominal"]
    out = {}
    s, bt, per_flight = run_eval(lake, det, faulty, nominal, det.calib["threshold"], cfg["min_consecutive"], win_s)
    out["ssl_crosschannel"] = {"summary": s, "by_fault": bt.reset_index().to_dict(orient="records")}
    print("SSL cross-channel model:", s)
    print(bt.to_string())
    (art / "per_flight_events.json").write_text(json.dumps(
        {k: {"results": [{kk: vv for kk, vv in r.items() if kk != "event"} for r in v["results"]],
             "events": [{kk: vv for kk, vv in e.items() if kk not in ("profile", "window_rows")}
                        for e in v["events"]]} for k, v in per_flight.items()}, indent=1, default=str))
    if baselines:
        for kind in ("zscore", "iforest"):
            b = _Baseline(kind, det.norm, cfg["win"], cfg["stride"]).fit(lake, split["train"], split["val"],
                                                                         cfg["calib_quantile"])
            s, bt, _ = run_eval(lake, b, faulty, nominal, b.threshold, cfg["min_consecutive"], win_s)
            out[kind] = {"summary": s, "by_fault": bt.reset_index().to_dict(orient="records")}
            print(f"baseline {kind}:", s)
    (art / "eval.json").write_text(json.dumps(out, indent=2, default=str))
    return out


__all__ = ["evaluate", "run_eval", "match_events", "FEATURES"]
