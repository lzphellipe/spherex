"""Renderização de fontes pontuais e injeção sintética.

A mesma função injeta fontes no céu sintético e em imagens reais do SPHEREx:
é assim que a curva de completude mede o sistema de verdade.
"""
from __future__ import annotations

import numpy as np

from spherex.domain.models import Exposure, KnownObject
from spherex.pipeline.geometry import offsets_to_world, world_to_pixel


def mag_to_flux(mag: np.ndarray | float, zeropoint: float) -> np.ndarray:
    return 10.0 ** (-0.4 * (np.asarray(mag, float) - zeropoint))


def add_gaussian_sources(image: np.ndarray, x: np.ndarray, y: np.ndarray, flux: np.ndarray,
                         sigma: float, radius: int | None = None) -> None:
    """Soma PSFs gaussianas normalizadas (fluxo total = flux) in-place."""
    radius = radius or int(np.ceil(4 * sigma))
    ny, nx = image.shape
    norm = 1.0 / (2 * np.pi * sigma**2)
    for xi, yi, fi in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(flux), strict=True):
        if not (-radius <= xi < nx + radius and -radius <= yi < ny + radius):
            continue
        x0, x1 = max(int(xi) - radius, 0), min(int(xi) + radius + 1, nx)
        y0, y1 = max(int(yi) - radius, 0), min(int(yi) + radius + 1, ny)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        image[y0:y1, x0:x1] += fi * norm * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma**2))


def mover_position(ra0: float, dec0: float, rate_ra: float, rate_dec: float,
                   t0: float, mjd: float) -> tuple[float, float]:
    """Movimento linear no plano tangente (arcsec/dia) a partir de (ra0, dec0) em t0."""
    dt = mjd - t0
    ra, dec = offsets_to_world(np.array([rate_ra * dt]), np.array([rate_dec * dt]), ra0, dec0)
    return float(ra[0]), float(dec[0])


def make_injection_plan(field_id: str, ra: float, dec: float, radius_deg: float, t0: float,
                        n: int, mag_range: tuple[float, float], rate_range: tuple[float, float],
                        rng: np.random.Generator, travel_days: float = 0.0) -> list[dict]:
    """Sorteia fontes móveis sintéticas. Taxa em escala log (cobre lentos e rápidos).

    O início é escolhido para que a trajetória inteira (travel_days) fique no campo.
    """
    plan = []
    for i in range(n):
        rate = float(np.exp(rng.uniform(np.log(rate_range[0]), np.log(rate_range[1]))))
        ang = rng.uniform(0, 2 * np.pi)
        r = 0.9 * radius_deg * 3600 * np.sqrt(rng.uniform())
        th = rng.uniform(0, 2 * np.pi)
        xi = r * np.cos(th) - 0.5 * rate * np.cos(ang) * travel_days
        eta = r * np.sin(th) - 0.5 * rate * np.sin(ang) * travel_days
        r0, d0 = offsets_to_world(np.array([xi]), np.array([eta]), ra, dec)
        plan.append({
            "name": f"inj_{field_id}_{i:03d}", "ra0": float(r0[0]), "dec0": float(d0[0]), "t0": t0,
            "rate_ra": rate * np.cos(ang), "rate_dec": rate * np.sin(ang),
            "rate": rate, "mag": float(rng.uniform(*mag_range)),
        })
    return plan


def plan_positions(plan: list[dict], mjd: float, injected: bool = True) -> list[KnownObject]:
    out = []
    for p in plan:
        ra, dec = mover_position(p["ra0"], p["dec0"], p["rate_ra"], p["rate_dec"], p["t0"], mjd)
        out.append(KnownObject(name=p["name"], mjd=mjd, ra=ra, dec=dec, mag=p["mag"],
                               rate_arcsec_per_day=p["rate"], injected=injected))
    return out


def inject(exposure: Exposure, plan: list[dict], zeropoint: float) -> list[KnownObject]:
    """Injeta o plano na exposição (in-place) e retorna as posições verdadeiras."""
    objs = plan_positions(plan, exposure.mjd)
    if not objs:
        return objs
    x, y = world_to_pixel(exposure.wcs_header, np.array([o.ra for o in objs]), np.array([o.dec for o in objs]))
    add_gaussian_sources(exposure.image, x, y, mag_to_flux([o.mag for o in objs], zeropoint),
                         exposure.psf_sigma_px)
    return objs
