"""Fault-signature library: what does each failure mode *look like* in telemetry?

Library flights (with labeled, injected faults) contribute signatures taken from the first
minute after onset. Each signature = per-channel anomaly z-profile (interpretable) + a
PCA-compressed SSL embedding (captures temporal shape). New detections are matched to the
library by k-NN; a large distance to every prototype marks a *novel* signature that should
go to an engineer for labeling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flightlake import Lake
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, fault_intervals
from .score import Detector


def _signature_windows(det: Detector, tel: pd.DataFrame, t0: float, t1: float):
    W, end = det.flight_windows(tel, stride=16)
    t_end = tel.t_s.to_numpy()[end]
    sel = (t_end >= t0) & (t_end <= t1)
    return W[sel]


def _sig(det: Detector, W: np.ndarray):
    Z = det._z(det.channel_errors(W))
    return Z.mean(axis=0), det.embeddings(W).mean(axis=0)


def build_library(lake: Lake, det: Detector, fids: list[str], horizon_s: float = 60.0):
    prof, emb, y, src = [], [], [], []
    for fid in fids:
        tel = lake.telemetry(fid)
        for ftype, fs, fe in fault_intervals(lake.summary(fid)["meta"]["faults"], float(tel.t_s.max())):
            W = _signature_windows(det, tel, fs + 3.0, min(fe, fs + horizon_s) + det.cfg["win"] / 10)
            if len(W) == 0:
                continue
            p, e = _sig(det, W)
            prof.append(p)
            emb.append(e)
            y.append(ftype)
            src.append(fid)
    return np.array(prof), np.array(emb), np.array(y), src


def run(lake_root: str, art_dir: str = "artifacts") -> dict:
    lake, art = Lake(lake_root), Path(art_dir)
    det = Detector.load(art)
    split = json.loads((art / "split.json").read_text())
    P_lib, E_lib, y_lib, _ = build_library(lake, det, split["library"])

    # test signatures come from *detected* events (what you'd have in operations)
    ev = json.loads((art / "per_flight_events.json").read_text())
    P_te, E_te, y_te = [], [], []
    for fid in split["test_faulty"]:
        tel = None
        for r in ev[fid]["results"]:
            if not r["detected"]:
                continue
            tel = tel if tel is not None else lake.telemetry(fid)
            t_det = r["onset"] + r["latency_s"]
            W = _signature_windows(det, tel, t_det, t_det + 60.0)
            if len(W):
                p, e = _sig(det, W)
                P_te.append(p)
                E_te.append(e)
                y_te.append(r["fault"])
    P_te, E_te, y_te = np.array(P_te), np.array(E_te), np.array(y_te)

    pca = PCA(n_components=min(16, len(E_lib) - 1), random_state=0).fit(E_lib)
    variants = {
        "z_profile": (P_lib, P_te),
        "ssl_embedding": (pca.transform(E_lib), pca.transform(E_te)),
        "combined": (np.hstack([P_lib, pca.transform(E_lib)]), np.hstack([P_te, pca.transform(E_te)])),
    }
    results = {}
    labels = sorted(set(y_lib))
    for name, (A, B) in variants.items():
        sc = StandardScaler().fit(A)
        knn = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(sc.transform(A), y_lib)
        pred = knn.predict(sc.transform(B))
        results[name] = {"accuracy": round(float(accuracy_score(y_te, pred)), 3),
                         "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 3)}
        if name == "combined":
            cm = confusion_matrix(y_te, pred, labels=labels)
            results["confusion"] = {"labels": labels, "matrix": cm.tolist()}
    km = KMeans(n_clusters=len(labels), n_init=10, random_state=0).fit(StandardScaler().fit_transform(P_lib))
    results["unsupervised_ari_library"] = round(float(adjusted_rand_score(y_lib, km.labels_)), 3)
    results["n_library"], results["n_test"] = int(len(y_lib)), int(len(y_te))

    # interpretable signature cards
    cards = {}
    for ft in labels:
        mean = pd.Series(P_lib[y_lib == ft].mean(axis=0), index=FEATURES).sort_values(ascending=False)
        cards[ft] = {k: round(float(v), 1) for k, v in mean.head(5).items()}
    results["signature_cards"] = cards
    np.savez(art / "signature_library.npz", profiles=P_lib, embeddings=E_lib, labels=y_lib,
             pca_components=pca.components_, pca_mean=pca.mean_)
    (art / "signatures.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "confusion"}, indent=2))
    return results
