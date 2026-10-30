"""Recortes (referência, science, diferença) por detecção, entrada da CNN real/bogus."""
from __future__ import annotations

import numpy as np

from spherex.domain.models import Detection
from spherex.pipeline.preprocess import DifferencePair


def cutout(img: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    h = size // 2
    xi, yi = int(round(x)), int(round(y))
    out = np.zeros((size, size), np.float32)
    ny, nx = img.shape
    x0, x1 = max(xi - h, 0), min(xi + h + 1, nx)
    y0, y1 = max(yi - h, 0), min(yi + h + 1, ny)
    if x0 < x1 and y0 < y1:
        out[y0 - (yi - h):y1 - (yi - h), x0 - (xi - h):x1 - (xi - h)] = np.nan_to_num(img[y0:y1, x0:x1])
    return out


def triplet(pair: DifferencePair, det: Detection, size: int) -> np.ndarray:
    """Array (3, size, size) normalizado pelo ruído da diferença."""
    s = max(pair.noise, 1e-6)
    return np.stack([cutout(pair.reference_image, det.x, det.y, size) / s,
                     cutout(pair.science_image, det.x, det.y, size) / s,
                     cutout(pair.difference, det.x, det.y, size) / s]).astype(np.float32)


def handcrafted_features(stamp: np.ndarray, det: Detection) -> np.ndarray:
    """Features do baseline: SNR, nitidez, simetria e contraste com a referência."""
    diff = stamp[2]
    c = diff.shape[0] // 2
    peak = diff[c, c]
    core = diff[c - 1:c + 2, c - 1:c + 2].sum()
    ring = diff[c - 3:c + 4, c - 3:c + 4].sum() - core
    sharp = peak / (abs(core) + 1e-6)
    sym = np.abs(diff - diff[::-1, ::-1]).sum() / (np.abs(diff).sum() + 1e-6)
    ref_core = stamp[0][c - 1:c + 2, c - 1:c + 2].sum()
    return np.array([np.log1p(max(det.snr, 0)), sharp, ring / (abs(core) + 1e-6), sym,
                     np.sign(ref_core) * np.log1p(abs(ref_core))], np.float32)
