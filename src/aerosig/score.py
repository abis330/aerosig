"""Inference: per-channel anomaly scores, events with attribution, streaming-style latency."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flightlake import Lake

from .data import FEATURES, Normalizer, eligible_windows, features, windows
from .model import CrossChannelPatchTST, group_masks


class Detector:
    def __init__(self, model: CrossChannelPatchTST, norm: Normalizer, cfg: dict, calib: dict | None = None):
        self.model, self.norm, self.cfg, self.calib = model.eval(), norm, cfg, calib or {}
        self.masks = group_masks(len(FEATURES), model.n_patch, cfg["n_groups"])
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @classmethod
    def load(cls, art_dir: str | Path, calibrated: bool = True) -> Detector:
        art = Path(art_dir)
        ck = torch.load(art / "model.pt", map_location="cpu", weights_only=False)
        c = ck["config"]
        model = CrossChannelPatchTST(len(ck["features"]), c["win"], c["patch"], c["d"], c["depth"], c["heads"], 0.0)
        model.load_state_dict(ck["state_dict"])
        norm = Normalizer.from_dict(json.loads((art / "normalizer.json").read_text()))
        calib = json.loads((art / "calibration.json").read_text()) if calibrated else None
        det = cls(model, norm, c, calib)
        det.art_dir = art
        return det

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def channel_errors(self, W: np.ndarray, batch: int = 256) -> np.ndarray:
        """(N, C, T) windows -> (N, C) mean squared reconstruction error with each channel hidden
        (channels are hidden group-wise: n_groups forward passes instead of C)."""
        N, C, _ = W.shape
        err = np.zeros((N, C), np.float32)
        X = torch.from_numpy(W)
        for i in range(0, N, batch):
            xb = X[i:i + batch]
            for m in self.masks:
                mb = m[None].expand(len(xb), -1, -1)
                rec = self.model(xb, mb)
                chans = m[:, 0]
                e = ((rec - xb) ** 2).mean(dim=2)  # B,C
                err[i:i + batch, chans.numpy()] = e[:, chans].numpy()
        return err

    def _z(self, err: np.ndarray) -> np.ndarray:
        """Normalized error: reconstruction error as a multiple of that channel's nominal 99th percentile.
        (A log-error robust z was tried first; its heavy nominal tails pushed the threshold so high that
        persistent single-channel faults were missed - chosen on the library/dev split, see README.)"""
        return err / np.array(self.calib["q99"])

    @torch.no_grad()
    def embeddings(self, W: np.ndarray, batch: int = 256) -> np.ndarray:
        return np.concatenate([self.model.embedding(torch.from_numpy(W[i:i + batch])).numpy()
                               for i in range(0, len(W), batch)]) if len(W) else np.zeros((0, 1))

    def flight_windows(self, tel: pd.DataFrame, stride: int | None = None):
        z = self.norm(features(tel))
        return windows(z, self.cfg["win"], stride or self.cfg["stride"])

    # ------------------------------------------------------------------ calibration
    def calibrate(self, lake: Lake, val_ids: list[str]) -> dict:
        errs = []
        for fid in val_ids:
            tel = lake.telemetry(fid)
            W, end = self.flight_windows(tel)
            errs.append(self.channel_errors(W[eligible_windows(tel, end, self.cfg["win"])]))
        E = np.concatenate(errs)
        self.calib = {"q99": np.maximum(np.quantile(E, 0.99, axis=0), 1e-6).tolist()}
        Z = self._z(E)
        s = window_score(Z)
        q = self.cfg["calib_quantile"]
        self.calib.update(threshold=float(np.quantile(s, q)), quantile=q, n_val_windows=int(len(s)),
                          features=FEATURES)
        (self.art_dir / "calibration.json").write_text(json.dumps(self.calib, indent=2))
        return self.calib

    # ------------------------------------------------------------------ flight scoring
    def score_flight(self, tel: pd.DataFrame, stride: int | None = None) -> pd.DataFrame:
        W, end_idx = self.flight_windows(tel, stride)
        if not len(W):
            return pd.DataFrame()
        Z = self._z(self.channel_errors(W))
        df = pd.DataFrame(Z, columns=[f"z_{c}" for c in FEATURES])
        df.insert(0, "t_end", tel["t_s"].to_numpy()[end_idx])
        df.insert(1, "t_start", tel["t_s"].to_numpy()[end_idx - self.cfg["win"] + 1])
        df.insert(2, "score", window_score(Z))
        df.insert(3, "eligible", eligible_windows(tel, end_idx, self.cfg["win"]))
        df.insert(4, "flagged", df["eligible"] & (df["score"] > self.calib["threshold"]))
        df.insert(5, "phase", tel["phase_est"].to_numpy()[end_idx])
        return df

    def events(self, scored: pd.DataFrame, min_consecutive: int | None = None) -> list[dict]:
        return extract_events(scored, self.calib["threshold"], min_consecutive or self.cfg["min_consecutive"])


def window_score(Z: np.ndarray) -> np.ndarray:
    """Mean of the top-2 channel z-scores: robust to a single noisy channel, still sensitive to
    single-channel faults (an airspeed probe) and multi-channel ones (GNSS)."""
    top = np.sort(Z, axis=1)[:, -2:]
    return top.mean(axis=1)


def extract_events(scored: pd.DataFrame, threshold: float, min_consecutive: int = 2) -> list[dict]:
    if scored.empty:
        return []
    f = (scored["score"] > threshold).to_numpy().copy()
    if "eligible" in scored:
        f = f & scored["eligible"].to_numpy(dtype=bool)
    zcols = [c for c in scored.columns if c.startswith("z_")]
    evs, i, n = [], 0, len(f)
    while i < n:
        if not f[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (f[j + 1] or (j + 2 < n and f[j + 2])):  # bridge 1-window holes
            j += 1
        run = scored.iloc[i:j + 1]
        if f[i:j + 1].sum() >= min_consecutive:
            k = np.flatnonzero(f[i:j + 1])[min_consecutive - 1]
            head = run.iloc[: min(len(run), 5)]
            prof = head[zcols].mean()
            top = prof.sort_values(ascending=False).head(5)
            evs.append({
                "t_start": float(run["t_start"].iloc[0]), "t_detect": float(run["t_end"].iloc[k]),
                "t_end": float(run["t_end"].iloc[-1]), "peak_score": float(run["score"].max()),
                "phase_at_detect": str(run["phase"].iloc[k]),
                "top_channels": {c[2:]: round(float(v), 2) for c, v in top.items()},
                "profile": prof.to_numpy().round(3).tolist(),
                "window_rows": list(range(i, j + 1)),
            })
        i = j + 1
    return evs
