import numpy as np
import pandas as pd
import pytest
import torch

from aerosig.data import FEATURES, Normalizer, features, windows
from aerosig.model import CrossChannelPatchTST, group_masks, make_mask
from aerosig.score import extract_events


def test_model_shapes_and_masks():
    C, T = len(FEATURES), 64
    m = CrossChannelPatchTST(C, win=T, patch=16, d=16, depth=1, heads=2)
    x = torch.randn(3, C, T)
    mask = make_mask(3, C, T // 16)
    assert m(x, mask).shape == (3, C, T)
    assert m.embedding(x).shape == (3, C * 16)
    g = group_masks(C, T // 16, 5)
    cover = torch.stack([gm[:, 0] for gm in g]).sum(0)
    assert (cover == 1).all()  # every channel hidden in exactly one inference pass


def test_windows_and_normalizer():
    z = np.random.randn(1000, len(FEATURES)).astype(np.float32)
    w, end = windows(z, 128, 32)
    assert w.shape == (len(end), len(FEATURES), 128) and end[0] == 127
    n = Normalizer.fit([z])
    assert np.abs(np.median(n(z), axis=0)).max() < 0.1


def test_extract_events_requires_persistence():
    s = pd.DataFrame({"t_start": np.arange(10.0), "t_end": np.arange(10.0) + 12.8, "score": [0, 9, 0, 9, 9, 9, 0, 0, 0, 0],
                      "phase": ["cruise"] * 10, "z_a": np.arange(10.0)})
    ev = extract_events(s, threshold=5, min_consecutive=2)
    assert len(ev) == 1 and ev[0]["t_detect"] == s.t_end[3]  # 1-window holes are bridged


@pytest.mark.slow
def test_train_calibrate_score(tmp_path):
    from flightlake.lake import Lake
    from flightlake.sim.writer import generate_fleet

    from aerosig.score import Detector
    from aerosig.train import train

    paths = generate_fleet(10, tmp_path / "raw", seed=1, fault_rate=0.5)
    lake = Lake(tmp_path / "lake")
    lake.ingest_many(paths, workers=1)
    out = train(str(tmp_path / "lake"), str(tmp_path / "art"), epochs=1, d=16, depth=1, heads=2, max_train_windows=300)
    assert out["threshold"] > 0
    det = Detector.load(tmp_path / "art")
    s = det.score_flight(lake.telemetry(lake.flight_ids()[0]))
    assert {"score", "flagged", "z_airspeed_mps"} <= set(s.columns)
    tel = lake.telemetry(lake.flight_ids()[0])
    assert np.isfinite(features(tel)).mean() > 0.9
