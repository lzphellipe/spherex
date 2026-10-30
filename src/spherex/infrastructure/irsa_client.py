"""Adapter real do SPHEREx no IRSA (SIA v2 + cutouts do IBE).

Conferido contra a documentação oficial do IRSA (out/2026), ver docs/dados.md:
- SIA v2: https://irsa.ipac.caltech.edu/SIA, COLLECTION=spherex_qr2,
  POS=circle ra dec r (graus), BAND em metros, MAXREC, RESPONSEFORMAT.
- Cutouts: <access_url>?center=ra,dec&size=<graus> (sintaxe do IBE). Cutouts do
  SPHEREx preservam a estrutura multi-extensão (IMAGE, FLAGS, VARIANCE, ZODI,
  PSF, WCS-WAVE) e não vêm comprimidos.
Ainda a confirmar com `make test-network`: semântica dos bits de FLAGS e o
nome da WCS espectral alternativa ("W"), ver `_central_wavelength`.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import httpx
import numpy as np
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.wcs import WCS

from spherex.domain.models import Exposure
from spherex.infrastructure.fits_io import celestial_header

log = logging.getLogger(__name__)

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
COLLECTION = "spherex_qr2"
_UNIT_TO_UM = {"um": 1.0, "micron": 1.0, "m": 1e6, "nm": 1e-3, "angstrom": 1e-4, "Angstrom": 1e-4}


class IrsaSpherexSource:
    """ExposureSource real. Use sempre embrulhado em CachedExposureSource."""

    def __init__(self, cutout_size_deg: float | None = 0.1, timeout: float = 120.0,
                 min_interval_s: float = 1.0, maxrec: int = 2000,
                 band_um: tuple[float, float] | None = None, flag_bits: int | None = None,
                 client: httpx.Client | None = None):
        """band_um: filtra na busca as exposições cujo intervalo cobre essa faixa (µm).
        flag_bits: máscara de bits de FLAGS que invalida o pixel; None = qualquer bit != 0
        (conservador até conferirmos o significado de cada bit na documentação do SPHEREx)."""
        self.cutout_size_deg = cutout_size_deg
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.min_interval_s = min_interval_s       # respeito ao serviço: no máx. 1 req/s
        self.maxrec = maxrec                       # SIA cancela jobs > 5 min; limite explícito
        self.band_um = band_um
        self.flag_bits = flag_bits
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
        params = {"COLLECTION": COLLECTION, "POS": f"circle {ra} {dec} {radius_deg}",
                  "RESPONSEFORMAT": "VOTABLE", "MAXREC": str(self.maxrec)}
        if self.band_um:
            params["BAND"] = f"{self.band_um[0] * 1e-6:.6e} {self.band_um[1] * 1e-6:.6e}"
        r = self._get(SIA_URL, **params)
        table = parse_single_table(io.BytesIO(r.content)).to_table()
        if len(table) >= self.maxrec:
            log.warning("SIA atingiu MAXREC=%d em %s: reduza o raio ou aumente o limite", self.maxrec, field_id)
        rows = []
        for row in table:
            fmt = str(row["access_format"]) if "access_format" in table.colnames else "image/fits"
            if "fits" not in fmt.lower():
                continue
            did = str(row["obs_publisher_did"])
            em_min, em_max = float(row["em_min"]), float(row["em_max"])
            rec = {
                "id": did, "access_url": str(row["access_url"]),
                "mjd": float(row["t_min"]),                      # ObsCore: MJD
                "wavelength_um_sia": 0.5 * (em_min + em_max) * 1e6,  # ObsCore: metros
                "band": str(row["energy_bandpassname"]) if "energy_bandpassname" in table.colnames else None,
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
            params = {"center": f"{ra},{dec}", "size": f"{self.cutout_size_deg}", "gzip": "false"}
        r = self._get(url, **params)
        with fits.open(io.BytesIO(r.content), memmap=False) as hdul:
            img_hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
            h = img_hdu.header
            wcs_header = celestial_header(h)
            image = np.asarray(img_hdu.data, dtype=np.float32)
            mask = None
            if "FLAGS" in hdul:
                flags = np.asarray(hdul["FLAGS"].data).astype(np.int64)
                mask = (flags & self.flag_bits) != 0 if self.flag_bits is not None else flags != 0
            ny, nx = image.shape
            wavelength, source = _central_wavelength(hdul, h, nx, ny, rec["wavelength_um_sia"])
        if source != "wcs-wave":
            log.warning("%s: comprimento de onda da linha SIA (média do detector), não do pixel central",
                        exposure_id)
        return Exposure(id=exposure_id, field_id=field_id, mjd=rec["mjd"],
                        detector=str(h.get("DETECTOR", h.get("DETNAME", "?"))),
                        wavelength_um=wavelength, image=image, wcs_header=wcs_header,
                        psf_sigma_px=float(h.get("PSFSIG", 1.2)), mask=mask, source_path=url)


def _central_wavelength(hdul: fits.HDUList, h: fits.Header, nx: int, ny: int,
                        fallback_um: float) -> tuple[float, str]:
    """Comprimento de onda no centro do recorte.

    Com o LVF o valor muda ao longo do detector, então a média do SIA (em_min/em_max)
    pode errar a banda de um recorte pequeno. O SPHEREx traz a tabela de lookup
    WCS-WAVE; aqui tentamos a WCS espectral alternativa "W" que a referencia.
    Se não existir ou falhar, devolve a média do SIA e marca a origem.
    """
    if "WCS-WAVE" not in hdul:
        return fallback_um, "sia-em-mid"
    try:
        w = WCS(h, fobj=hdul, key="W")
        vals = w.pixel_to_world_values(nx / 2.0 - 0.5, ny / 2.0 - 0.5)
        value = float(np.atleast_1d(vals[0] if isinstance(vals, tuple | list) else vals)[0])
        unit = str(w.wcs.cunit[0]) or "um"
        value *= _UNIT_TO_UM.get(unit, 1.0)
        if np.isfinite(value) and 0.5 < value < 6.0:   # faixa do SPHEREx: 0,75–5 µm
            return value, "wcs-wave"
    except Exception as e:  # noqa: BLE001 - qualquer falha cai no fallback documentado
        log.debug("WCS espectral indisponível: %r", e)
    return fallback_um, "sia-em-mid"


def save_raw(content: bytes, path: Path) -> None:
    from spherex.infrastructure.checkpoint import atomic_write
    atomic_write(path, lambda p: p.write_bytes(content))
