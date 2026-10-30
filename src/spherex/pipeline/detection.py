"""Detecção de fontes na imagem diferença e remoção das estáticas."""
from __future__ import annotations

import numpy as np
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from spherex.domain.models import Detection
from spherex.pipeline.geometry import angular_sep_arcsec, pixel_to_world, tangent_offsets_arcsec
from spherex.pipeline.preprocess import DifferencePair


def detect(pair: DifferencePair, threshold_sigma: float, npixels: int) -> list[Detection]:
    """Fontes positivas (presentes na science, ausentes na referência)."""
    if pair.skipped_reason or not np.isfinite(pair.noise) or pair.noise <= 0:
        return []
    diff = np.nan_to_num(pair.difference, nan=0.0)
    sigma = pair.science.psf_sigma_px
    smooth = gaussian_filter(diff, sigma)
    # ruído da imagem suavizada pelo filtro casado (escala ~ 1/(2*sqrt(pi)*sigma))
    noise_s = pair.noise / (2 * np.sqrt(np.pi) * sigma)
    seg = detect_sources(smooth, threshold_sigma * noise_s, n_pixels=npixels)
    if seg is None:
        return []
    # separa fontes vizinhas fundidas (campos densos + PSF subamostrada)
    seg = deblend_sources(smooth, seg, n_pixels=npixels, n_levels=32, contrast=0.01)
    cat = SourceCatalog(diff, seg)
    x = np.asarray(cat.x_centroid, float)
    y = np.asarray(cat.y_centroid, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    flux = np.asarray(cat.segment_flux, float)[ok]
    area = np.asarray(cat.area.value, float)[ok]
    ra, dec = pixel_to_world(pair.science.wcs_header, x, y)
    snr = flux / (pair.noise * np.sqrt(np.maximum(area, 1)))
    e = pair.science
    return [Detection(id=f"{e.id}_d{i:04d}", exposure_id=e.id, field_id=e.field_id, mjd=e.mjd,
                      ra=float(ra[i]), dec=float(dec[i]), x=float(x[i]), y=float(y[i]),
                      flux=float(flux[i]), snr=float(snr[i]))
            for i in range(len(x)) if snr[i] > 0]


def remove_static(dets: list[Detection], static: list[tuple[float, float]], radius_arcsec: float,
                  ra0: float, dec0: float) -> tuple[list[Detection], list[Detection]]:
    """Separa detecções que coincidem com fontes estáticas (resíduos de subtração)."""
    if not dets or not static:
        return dets, []
    sra, sdec = np.array(static).T
    sx, sy = tangent_offsets_arcsec(sra, sdec, ra0, dec0)
    tree = cKDTree(np.c_[sx, sy])
    dx, dy = tangent_offsets_arcsec([d.ra for d in dets], [d.dec for d in dets], ra0, dec0)
    dist, _ = tree.query(np.c_[dx, dy], k=1)
    keep = [d for d, r in zip(dets, dist, strict=True) if r > radius_arcsec]
    drop = [d for d, r in zip(dets, dist, strict=True) if r <= radius_arcsec]
    return keep, drop


def match_to_truth(dets: list[Detection], truth: list, radius_arcsec: float) -> dict[str, str]:
    """detection.id -> nome do objeto verdadeiro mais próximo dentro do raio."""
    out: dict[str, str] = {}
    for d in dets:
        best, best_r = None, radius_arcsec
        for t in truth:
            r = float(angular_sep_arcsec(d.ra, d.dec, t.ra, t.dec))
            if r <= best_r:
                best, best_r = t.name, r
        if best:
            out[d.id] = best
    return out
