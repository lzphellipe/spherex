"""Adapters JPL: Horizons (posição do SPHEREx) e Small-Body Identification (asteroides no campo).

Status de verificação dos parâmetros em docs/dados.md. Escritos sem acesso à rede;
os testes de integração estão marcados como `network`.
"""
from __future__ import annotations

import re
from functools import lru_cache

import httpx
from astropy.time import Time

from spherex.domain.models import Exposure, KnownObject
from spherex.pipeline.geometry import pixel_to_world

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBIDENT_URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"
SPHEREX_ID = "-163182"


class HorizonsEphemeris:
    """EphemerisSource: vetor heliocêntrico do SPHEREx (au, au/dia, eclíptica J2000)."""

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60.0)

    @lru_cache(maxsize=4096)  # noqa: B019 - cache por instância é aceitável aqui
    def state(self, mjd: float) -> tuple[float, float, float, float, float, float]:
        jd = Time(mjd, format="mjd").jd
        params = {
            "format": "json", "COMMAND": f"'{SPHEREX_ID}'", "EPHEM_TYPE": "VECTORS",
            "CENTER": "'500@10'", "TLIST": f"'{jd}'", "OUT_UNITS": "AU-D",
            "REF_PLANE": "ECLIPTIC", "VEC_TABLE": "2", "CSV_FORMAT": "YES",
        }
        r = self.client.get(HORIZONS_URL, params=params)
        r.raise_for_status()
        text = r.json()["result"]
        block = text.split("$$SOE")[1].split("$$EOE")[0]
        nums = [float(v) for v in re.findall(r"[-+]?\d+\.\d+E[-+]\d+", block)]
        x, y, z, vx, vy, vz = nums[-6:]
        return x, y, z, vx, vy, vz

    def observer_position(self, mjd: float) -> tuple[float, float, float]:
        x, y, z, *_ = self.state(mjd)
        return x, y, z


class SbIdentKnownObjects:
    """KnownObjectSource: asteroides/cometas conhecidos no campo de uma exposição."""

    def __init__(self, ephemeris: HorizonsEphemeris, client: httpx.Client | None = None):
        self.eph = ephemeris
        self.client = client or httpx.Client(timeout=120.0)

    def known_objects(self, exposure: Exposure) -> list[KnownObject]:
        ny, nx = exposure.image.shape
        ra_c, dec_c = pixel_to_world(exposure.wcs_header, [nx / 2], [ny / 2])
        ra_e, dec_e = pixel_to_world(exposure.wcs_header, [0, nx - 1], [0, ny - 1])
        hw_ra = abs(float(ra_e[1] - ra_e[0])) / 2
        hw_dec = abs(float(dec_e[1] - dec_e[0])) / 2
        state = ",".join(f"{v:.10e}" for v in self.eph.state(exposure.mjd))
        params = {
            "xobs": state,
            "obs-time": f"{Time(exposure.mjd, format='mjd').jd:.6f}",
            "fov-ra-center": f"{float(ra_c[0]):.6f}", "fov-dec-center": f"{float(dec_c[0]):.6f}",
            "fov-ra-hwidth": f"{hw_ra:.5f}", "fov-dec-hwidth": f"{hw_dec:.5f}",
            "two-pass": "true", "suppress-first": "true",
        }
        r = self.client.get(SBIDENT_URL, params=params)
        r.raise_for_status()
        data = r.json()
        fields = data.get("fields_second") or data.get("fields_first") or []
        rows = data.get("data_second_pass") or data.get("data_first_pass") or []
        out = []
        for row in rows:
            rec = dict(zip(fields, row, strict=False))
            ra, dec = _parse_radec(rec)
            if ra is None:
                continue
            mag = _to_float(rec.get("Visual magnitude (V)") or rec.get("V"))
            out.append(KnownObject(name=str(rec.get("Object name", rec.get("name", "?"))),
                                   mjd=exposure.mjd, ra=ra, dec=dec, mag=mag))
        return out


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_radec(rec: dict) -> tuple[float | None, float | None]:
    """sb_ident devolve RA/Dec sexagesimais (ex.: '10:02:03.12', '+02:01:00.1')."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    ra = rec.get("Astrometric RA (hh:mm:ss)") or rec.get("RA")
    dec = rec.get("Astrometric Dec (dd mm'ss\")") or rec.get("DEC")
    if ra is None or dec is None:
        return None, None
    c = SkyCoord(str(ra).replace(" ", ":"), str(dec).replace(" ", ":").replace("'", ":").replace('"', ""),
                 unit=(u.hourangle, u.deg))
    return float(c.ra.deg), float(c.dec.deg)
