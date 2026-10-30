"""Ligação de detecções entre épocas (tracklets) e shift-and-stack para objetos lentos."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
from scipy.spatial import cKDTree

from spherex.domain.models import Detection, Tracklet
from spherex.pipeline.geometry import tangent_offsets_arcsec


def link(dets: list[Detection], ra0: float, dec0: float, min_detections: int,
         max_rate_arcsec_per_day: float, match_radius_arcsec: float) -> list[Tracklet]:
    """Hipóteses de movimento linear a partir de pares de épocas, verificadas nas demais.

    Complexidade controlada por KD-tree: para cada detecção da época a, só os
    vizinhos da época b dentro de max_rate * dt viram hipótese.
    """
    if not dets:
        return []
    mjds = sorted({d.mjd for d in dets})
    if len(mjds) < min_detections:
        return []
    by_epoch = {t: [d for d in dets if d.mjd == t] for t in mjds}
    xy = {}
    trees = {}
    for t, ds in by_epoch.items():
        x, y = tangent_offsets_arcsec([d.ra for d in ds], [d.dec for d in ds], ra0, dec0)
        xy[t] = np.c_[x, y]
        trees[t] = cKDTree(xy[t]) if ds else None

    found: dict[frozenset[str], Tracklet] = {}
    for ta, tb in itertools.combinations(mjds, 2):
        if trees[ta] is None or trees[tb] is None:
            continue
        dt = tb - ta
        pairs = trees[ta].query_ball_tree(trees[tb], r=max_rate_arcsec_per_day * dt + match_radius_arcsec)
        for i, js in enumerate(pairs):
            for j in js:
                v = (xy[tb][j] - xy[ta][i]) / dt
                members = {ta: i, tb: j}
                for tc in mjds:
                    if tc in members or trees[tc] is None:
                        continue
                    pred = xy[ta][i] + v * (tc - ta)
                    dist, k = trees[tc].query(pred, k=1)
                    if dist <= match_radius_arcsec:
                        members[tc] = int(k)
                if len(members) < min_detections:
                    continue
                key = frozenset(by_epoch[t][k].id for t, k in members.items())
                if key in found:
                    continue
                tr = _fit(key, [by_epoch[t][k] for t, k in sorted(members.items())],
                          np.array([xy[t][k] for t, k in sorted(members.items())]), ra0, dec0)
                if tr.rms_arcsec <= match_radius_arcsec and tr.rate_arcsec_per_day <= max_rate_arcsec_per_day:
                    found[key] = tr
    return _dedupe(list(found.values()))


def _fit(key: frozenset[str], ds: list[Detection], pts: np.ndarray, ra0: float, dec0: float) -> Tracklet:
    t = np.array([d.mjd for d in ds])
    A = np.c_[np.ones_like(t), t - t.mean()]
    coef_x, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
    resid = np.hypot(pts[:, 0] - A @ coef_x, pts[:, 1] - A @ coef_y)
    # hashlib, não hash(): hash() de str muda a cada processo e quebraria a reprodutibilidade
    tid = "trk_" + hashlib.sha1("|".join(sorted(key)).encode()).hexdigest()[:12]
    return Tracklet(id=tid, field_id=ds[0].field_id, detections=ds,
                    rate_ra_arcsec_per_day=float(coef_x[1]), rate_dec_arcsec_per_day=float(coef_y[1]),
                    rms_arcsec=float(np.sqrt(np.mean(resid**2))))


def _dedupe(trs: list[Tracklet]) -> list[Tracklet]:
    """Cada detecção pertence a no máximo um tracklet: prioriza mais épocas e menor RMS."""
    used: set[str] = set()
    out = []
    for tr in sorted(trs, key=lambda t: (-t.n, round(t.rms_arcsec, 9), t.id)):  # desempate determinístico
        ids = {d.id for d in tr.detections}
        if ids & used:
            continue
        used |= ids
        out.append(tr)
    return out


def shift_and_stack(diffs: list[np.ndarray], mjds: list[float], rates_px_per_day: np.ndarray,
                    noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Co-adiciona as diferenças deslocadas ao longo de cada taxa hipotética (vx, vy).

    Para objetos lentos e fracos (regime Planeta X) que somem numa época isolada:
    o sinal soma coerentemente N vezes e o ruído só sqrt(N). Deslocamento inteiro
    (rápido, suficiente para prova de conceito); retorna (mapa de SNR máximo, índice da taxa).
    """
    t0 = mjds[0]
    best = np.full(diffs[0].shape, -np.inf)
    best_idx = np.zeros(diffs[0].shape, int)
    n = len(diffs)
    for k, (vx, vy) in enumerate(rates_px_per_day):
        acc = np.zeros_like(diffs[0], dtype=float)
        for d, t in zip(diffs, mjds, strict=True):
            sx, sy = int(round(-vx * (t - t0))), int(round(-vy * (t - t0)))
            acc += np.roll(np.nan_to_num(d), (sy, sx), axis=(0, 1))
        snr = acc / (noise * np.sqrt(n))
        better = snr > best
        best[better], best_idx[better] = snr[better], k
    return best, best_idx
