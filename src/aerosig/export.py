"""Export the encoder+reconstruction head to ONNX for edge / ground-station inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .score import Detector


def export_onnx(art_dir: str = "artifacts", out: str | None = None) -> Path:
    det = Detector.load(art_dir, calibrated=False)
    m = det.model.eval()
    C, T, P = m.n_ch, m.win, m.n_patch
    x = torch.randn(2, C, T)
    mask = torch.zeros(2, C, P, dtype=torch.bool)
    out = Path(out or Path(art_dir) / "aerosig.onnx")
    torch.onnx.export(m, (x, mask), str(out), input_names=["x", "mask"], output_names=["recon"],
                      dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"}, "recon": {0: "batch"}},
                      opset_version=17, dynamo=False)
    try:  # parity check if onnxruntime is installed
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out))
        ref = m(x, mask).detach().numpy()
        got = sess.run(None, {"x": x.numpy(), "mask": mask.numpy()})[0]
        print("onnx max abs diff:", float(np.abs(ref - got).max()))
    except ImportError:
        print("onnxruntime not installed; skipped parity check")
    return out
