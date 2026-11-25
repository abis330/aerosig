"""Self-supervised pre-training on nominal flights + threshold calibration."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from flightlake import Lake

from .data import FEATURES, Normalizer, features, split_flights, windows
from .model import CrossChannelPatchTST, make_mask
from .score import Detector

DEFAULTS = dict(win=128, patch=16, d=64, depth=3, heads=4, dropout=0.1, stride=32, epochs=8, batch=128, lr=1e-3,
                p_patch=0.25, max_ch=6, n_groups=5, seed=0, calib_quantile=0.998, min_consecutive=2,
                max_train_windows=40000, time_budget_min=30)


def load_windows(lake: Lake, fids, norm: Normalizer, win, stride):
    out = []
    for fid in fids:
        z = norm(features(lake.telemetry(fid)))
        w, _ = windows(z, win, stride)
        out.append(w)
    return np.concatenate(out) if out else np.zeros((0, len(FEATURES), win), np.float32)


def train(lake_root: str, out_dir: str = "artifacts", **overrides) -> dict:
    cfg = {**DEFAULTS, **overrides}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lake = Lake(lake_root)
    split = split_flights(lake.catalog(), seed=cfg["seed"])
    (out / "split.json").write_text(json.dumps(split, indent=2))

    norm = Normalizer.fit([features(lake.telemetry(f)) for f in split["train"]])
    (out / "normalizer.json").write_text(json.dumps(norm.to_dict()))
    X = load_windows(lake, split["train"], norm, cfg["win"], cfg["stride"])
    if len(X) > cfg["max_train_windows"]:
        X = X[np.random.choice(len(X), cfg["max_train_windows"], replace=False)]
    Xv = load_windows(lake, split["val"], norm, cfg["win"], cfg["stride"] * 2)
    print(f"train windows {X.shape}, val windows {Xv.shape}")

    model = CrossChannelPatchTST(len(FEATURES), cfg["win"], cfg["patch"], cfg["d"], cfg["depth"], cfg["heads"],
                                 cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps = cfg["epochs"] * math.ceil(len(X) / cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps, pct_start=0.1)
    Xt, Xvt = torch.from_numpy(X), torch.from_numpy(Xv)
    gen = torch.Generator().manual_seed(cfg["seed"])
    P = cfg["win"] // cfg["patch"]
    history, t0 = [], time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot = n = 0
        for i in range(0, len(Xt), cfg["batch"]):
            xb = Xt[perm[i:i + cfg["batch"]]]
            m = make_mask(len(xb), xb.shape[1], P, cfg["p_patch"], cfg["max_ch"], gen)
            rec = model(xb, m)
            mt = m.repeat_interleave(cfg["patch"], dim=2)
            loss = ((rec - xb) ** 2)[mt].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(xb), n + len(xb)
        model.eval()
        with torch.no_grad():
            vg = torch.Generator().manual_seed(123)
            vl = []
            for i in range(0, len(Xvt), 512):
                xb = Xvt[i:i + 512]
                m = make_mask(len(xb), xb.shape[1], P, cfg["p_patch"], cfg["max_ch"], vg)
                rec = model(xb, m)
                vl.append(((rec - xb) ** 2)[m.repeat_interleave(cfg["patch"], dim=2)].mean().item())
        history.append({"epoch": ep, "train_loss": tot / n, "val_loss": float(np.mean(vl)),
                        "elapsed_s": round(time.time() - t0, 1)})
        print(history[-1])
        if (time.time() - t0) / 60 > cfg["time_budget_min"]:
            print("time budget reached")
            break
    torch.save({"state_dict": model.state_dict(), "config": cfg, "features": FEATURES}, out / "model.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2))

    # ---- calibration on held-out *nominal* flights only
    det = Detector.load(out, calibrated=False)
    det.calibrate(lake, split["val"])
    print("calibration:", {k: v for k, v in det.calib.items() if k in ("threshold", "quantile")})
    return {"history": history, "threshold": det.calib["threshold"]}
