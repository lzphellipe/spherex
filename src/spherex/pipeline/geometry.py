"""WCS, conversões de coordenadas e reprojeção (astropy + scipy)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


def make_wcs_header(ra: float, dec: float, shape: tuple[int, int], pixel_scale_arcsec: float,
                    rotation_deg: float = 0.0) -> dict[str, Any]:
    """Cabeçalho TAN simples centrado em (ra, dec)."""
    ny, nx = shape
    scale = pixel_scale_arcsec / 3600.0
    th = np.deg2rad(rotation_deg)
    return {
        "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": float(ra), "CRVAL2": float(dec),
        "CRPIX1": (nx + 1) / 2.0, "CRPIX2": (ny + 1) / 2.0,
        "CD1_1": -scale * np.cos(th), "CD1_2": scale * np.sin(th),
        "CD2_1": scale * np.sin(th), "CD2_2": scale * np.cos(th),
        "CUNIT1": "deg", "CUNIT2": "deg",
    }


def to_wcs(header: dict[str, Any]) -> WCS:
    """WCS a partir do dict; cacheado porque construir WCS (com SIP) é caro e repetido."""
    items = tuple(sorted((k, v) for k, v in header.items()
                         if not k.startswith("NAXIS") or k in ("NAXIS1", "NAXIS2")))
    return _wcs_cached(items)


@lru_cache(maxsize=512)
def _wcs_cached(items: tuple) -> WCS:
    return WCS(dict(items), naxis=2)


def world_to_pixel(header: dict[str, Any], ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = to_wcs(header).all_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0)
    return x, y


def pixel_to_world(header: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ra, dec = to_wcs(header).all_pix2world(np.atleast_1d(x), np.atleast_1d(y), 0)
    return ra, dec


def reproject(image: np.ndarray, src_header: dict[str, Any], dst_header: dict[str, Any],
              dst_shape: tuple[int, int], order: int = 3) -> np.ndarray:
    """Reamostra `image` (grade de src) na grade de dst. Spline cúbica por padrão:
    a PSF do SPHEREx é subamostrada e a bilinear achata o pico, deixando resíduos
    de estrelas na subtração (medido: ~2x mais detecções espúrias com order=1).

    Pixels fora da cobertura da origem viram NaN (nunca zero, para não criar
    falsas fontes negativas na subtração).
    """
    ny, nx = dst_shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = pixel_to_world(dst_header, xx.ravel(), yy.ravel())
    sx, sy = world_to_pixel(src_header, ra, dec)
    img = np.nan_to_num(image.astype(np.float64), nan=0.0)
    out = map_coordinates(img, [sy, sx], order=order, mode="constant", cval=np.nan, prefilter=order > 1)
    ny_s, nx_s = image.shape
    outside = (sx < -0.5) | (sx > nx_s - 0.5) | (sy < -0.5) | (sy > ny_s - 0.5)
    out[outside] = np.nan
    return out.reshape(ny, nx)


def tangent_offsets_arcsec(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """Projeção gnomônica em arcsec relativa a (ra0, dec0): coordenadas lineares para ajuste de movimento."""
    ra, dec = np.deg2rad(np.asarray(ra, float)), np.deg2rad(np.asarray(dec, float))
    a0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    cosc = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - a0)
    xi = np.cos(dec) * np.sin(ra - a0) / cosc
    eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(ra - a0)) / cosc
    return np.rad2deg(xi) * 3600.0, np.rad2deg(eta) * 3600.0


def offsets_to_world(xi_arcsec: np.ndarray, eta_arcsec: np.ndarray, ra0: float,
                     dec0: float) -> tuple[np.ndarray, np.ndarray]:
    xi, eta = np.deg2rad(np.asarray(xi_arcsec) / 3600.0), np.deg2rad(np.asarray(eta_arcsec) / 3600.0)
    a0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    denom = np.cos(d0) - eta * np.sin(d0)
    ra = a0 + np.arctan2(xi, denom)
    dec = np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, denom))
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def angular_sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    r1, d1, r2, d2 = map(np.deg2rad, (ra1, dec1, ra2, dec2))
    s = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600.0
