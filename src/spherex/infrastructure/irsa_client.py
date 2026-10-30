"""Adapter real do SPHEREx no IRSA (SIA v2 + download de imagens/cutouts).

Endpoints e nomes de colunas estão em docs/dados.md, com o status de verificação.
Confira na documentação oficial do IRSA antes do hackathon: este módulo foi
escrito sem acesso à rede e os testes de integração estão marcados como `network`.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import httpx
import numpy as np
from astropy.io import fits
from astropy.io.votable import parse_single_table

from spherex.domain.models import Exposure
from spherex.infrastructure.fits_io import WCS_KEYS

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
COLLECTION = "spherex_qr2"


class IrsaSpherexSource:
    """ExposureSource real. Use sempre embrulhado em CachedExposureSource."""

    def __init__(self, cutout_size_deg: float | None = 0.1, timeout: float = 120.0,
                 min_interval_s: float = 1.0, client: httpx.Client | None = None):
        self.cutout_size_deg = cutout_size_deg
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.min_interval_s = min_interval_s       # respeito ao serviço: no máx. 1 req/s
        self._last = 0.0
        self._rows: dict[str, dict] = {}
        self._field_of: dict[str, tuple[str, float, float]] = {}

    def _get(self, url: str, **params) -> httpx.Response:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            try:
                r = self.client.get(url, params=params or None)
                self._last = time.monotonic()
                if r.status_code < 500:
                    r.raise_for_status()
                    return r
            except httpx.TransportError:
                if attempt == 3:
                    raise
            time.sleep(2**attempt)
        r.raise_for_status()
        return r

    def list_exposures(self, field_id: str, ra: float, dec: float, radius_deg: float) -> list[str]:
        r = self._get(SIA_URL, COLLECTION=COLLECTION, POS=f"CIRCLE {ra} {dec} {radius_deg}",
                      RESPONSEFORMAT="VOTABLE")
        table = parse_single_table(io.BytesIO(r.content)).to_table()
        rows = []
        for row in table:
            did = str(row["obs_publisher_did"])
            em_min, em_max = float(row["em_min"]), float(row["em_max"])
            rec = {
                "id": did, "access_url": str(row["access_url"]),
                "mjd": float(row["t_min"]),
                "wavelength_um": 0.5 * (em_min + em_max) * 1e6,   # ObsCore usa metros
            }
            self._rows[did] = rec
            self._field_of[did] = (field_id, ra, dec)
            rows.append(rec)
        rows.sort(key=lambda r_: r_["mjd"])
        return [r_["id"] for r_ in rows]

    def load(self, exposure_id: str) -> Exposure:
        rec = self._rows[exposure_id]
        field_id, ra, dec = self._field_of[exposure_id]
        url = rec["access_url"]
        params = {}
        if self.cutout_size_deg:
            params = {"center": f"{ra},{dec}", "size": f"{self.cutout_size_deg}"}
        r = self._get(url, **params)
        with fits.open(io.BytesIO(r.content), memmap=False) as hdul:
            img_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
            h = img_hdu.header
            wcs_header = {k: h[k] for k in WCS_KEYS if k in h}
            wcs_header["NAXIS1"], wcs_header["NAXIS2"] = h["NAXIS1"], h["NAXIS2"]
            mask = None
            if "FLAGS" in hdul:
                mask = np.asarray(hdul["FLAGS"].data) != 0
            image = np.asarray(img_hdu.data, dtype=np.float32)
        return Exposure(id=exposure_id, field_id=field_id, mjd=rec["mjd"],
                        detector=str(h.get("DETECTOR", h.get("DETNAME", "?"))),
                        wavelength_um=rec["wavelength_um"], image=image, wcs_header=wcs_header,
                        psf_sigma_px=float(h.get("PSFSIG", 1.2)), mask=mask, source_path=url)


def save_raw(content: bytes, path: Path) -> None:
    from spherex.infrastructure.checkpoint import atomic_write
    atomic_write(path, lambda p: p.write_bytes(content))
