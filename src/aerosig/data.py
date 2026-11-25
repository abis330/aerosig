"""Feature engineering + windowing on top of a flightlake lake."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Model input channels. Order matters only for checkpoints; related channels (e.g. the four
# motors) are kept adjacent so the interleaved masking groups split them across groups.
FEATURES = [
    "airspeed_mps", "groundspeed_mps", "alt_rel_m", "vz_mps", "roll_deg", "pitch_deg", "p_dps", "q_dps",
    "r_dps", "ax_mps2", "az_mps2", "vibe_mps2", "throttle", "elevator_cmd_deg", "elevator_pos_deg",
    "motor1_rpm", "motor2_rpm", "motor3_rpm", "motor4_rpm", "batt_v", "batt_a", "batt_temp_c", "gen_kw",
    "gnss_nsat", "gnss_hdop", "ekf_pos_innov_m", "ekf_vel_innov_mps", "ekf_hacc_m",
]
LOG_FEATURES = {"gnss_hdop", "ekf_pos_innov_m", "ekf_vel_innov_mps", "ekf_hacc_m", "vibe_mps2"}


def features(tel: pd.DataFrame) -> np.ndarray:
    """(T, C) float32 matrix of model features (raw physical units, log-compressed where heavy-tailed)."""
    x = tel[FEATURES].copy()
    x["roll_deg"] = x["roll_deg"].clip(-60, 60)
    for c in LOG_FEATURES:
        x[c] = np.log1p(x[c].clip(lower=0))
    x = x.ffill(limit=20)
    return x.to_numpy(dtype=np.float32)


@dataclass
class Normalizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, mats: list[np.ndarray]) -> Normalizer:
        allx = np.concatenate(mats)
        center = np.nanmedian(allx, axis=0)
        q75, q25 = np.nanpercentile(allx, 75, axis=0), np.nanpercentile(allx, 25, axis=0)
        scale = np.maximum(q75 - q25, 1e-3) / 1.349
        return cls(center.astype(np.float32), scale.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.center) / self.scale
        return np.clip(np.nan_to_num(z, nan=0.0), -12, 12).astype(np.float32)

    def to_dict(self):
        return {"center": self.center.tolist(), "scale": self.scale.tolist(), "features": FEATURES}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["center"], np.float32), np.array(d["scale"], np.float32))


def windows(z: np.ndarray, win: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows -> (N, C, win) and the end-index of each window (for streaming latency)."""
    T = len(z)
    if T < win:
        return np.zeros((0, z.shape[1], win), np.float32), np.zeros(0, int)
    starts = np.arange(0, T - win + 1, stride)
    idx = starts[:, None] + np.arange(win)[None, :]
    return z[idx].transpose(0, 2, 1).copy(), starts + win - 1


def airborne_mask(tel: pd.DataFrame) -> np.ndarray:
    return tel["phase_est"].isin(["climb", "cruise", "descent", "approach"]).to_numpy()


def eligible_windows(tel: pd.DataFrame, end_idx: np.ndarray, win: int, margin: int = 100) -> np.ndarray:
    """A window is scored only if it lies entirely in airborne phases and >= `margin` samples away from
    lift-off / touchdown (rotation and flare are separate regimes with their own monitors)."""
    air = airborne_mask(tel).astype(int)
    ok = pd.Series(air).rolling(win + 2 * margin, center=False, min_periods=1).min().to_numpy()
    idx = np.clip(end_idx + margin, 0, len(air) - 1)
    return ok[idx] == 1


def fault_intervals(meta_faults: list[dict], t_end: float) -> list[tuple[str, float, float]]:
    return [(f["type"], float(f["t_start"]), float(f["t_end"]) if f.get("t_end") is not None else t_end)
            for f in meta_faults]


def split_flights(catalog: pd.DataFrame, seed: int = 0, n_nominal_train: float = 0.6, n_nominal_val: float = 0.2,
                  faulty_lib_frac: float = 0.5) -> dict[str, list[str]]:
    """Nominal flights -> train / val (threshold calibration) / test (false alarms).
    Faulty flights -> library (labeled signatures) / test. Faulty flights are never used to train the
    detector, which mirrors reality: the detector learns 'normal' from flights signed off as nominal."""
    rng = np.random.default_rng(seed)
    nom = catalog.loc[catalog.injected_faults == "none", "flight_id"].to_numpy()
    flt = catalog.loc[catalog.injected_faults != "none", "flight_id"].to_numpy()
    rng.shuffle(nom)
    rng.shuffle(flt)
    a, b = int(len(nom) * n_nominal_train), int(len(nom) * (n_nominal_train + n_nominal_val))
    c = int(len(flt) * faulty_lib_frac)
    return {"train": sorted(nom[:a]), "val": sorted(nom[a:b]), "test_nominal": sorted(nom[b:]),
            "library": sorted(flt[:c]), "test_faulty": sorted(flt[c:])}
