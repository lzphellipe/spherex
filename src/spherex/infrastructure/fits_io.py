"""Leitura/escrita de exposições em FITS (formato interno do cache)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from spherex.domain.models import Exposure
from spherex.infrastructure.checkpoint import atomic_write


def celestial_header(h: fits.Header) -> dict:
    """WCS celeste completo (inclui distorção SIP), não só as palavras-chave básicas.

    Copiar só CRVAL/CD/CTYPE descartaria os coeficientes SIP e erraria posições
    na borda do detector justamente onde os objetos entram e saem do campo.
    """
    hdr = WCS(h, naxis=2, relax=True).to_header(relax=True)
    out = {k: hdr[k] for k in hdr}
    for k in ("WCSAXES", "DATE-OBS", "MJD-OBS", "DATEREF", "MJDREF"):
        out.pop(k, None)
    out["NAXIS1"], out["NAXIS2"] = int(h["NAXIS1"]), int(h["NAXIS2"])
    return out


def write_exposure(exp: Exposure, path: Path) -> None:
    hdr = fits.Header()
    for k, v in exp.wcs_header.items():
        if not k.startswith("NAXIS"):  # NAXISn é definido pelo próprio array
            hdr[k] = v
    hdr["EXPID"] = exp.id
    hdr["FIELDID"] = exp.field_id
    hdr["MJD-OBS"] = exp.mjd
    hdr["DETECTOR"] = exp.detector
    hdr["WAVELEN"] = (exp.wavelength_um, "central wavelength [um]")
    hdr["PSFSIG"] = (exp.psf_sigma_px, "PSF sigma [px]")
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(exp.image.astype(np.float32), hdr, name="IMAGE")]
    if exp.mask is not None:
        hdus.append(fits.ImageHDU(exp.mask.astype(np.uint8), name="FLAGS"))
    atomic_write(path, lambda p: fits.HDUList(hdus).writeto(p, overwrite=True))


def read_exposure(path: Path) -> Exposure:
    with fits.open(path, memmap=False) as hdul:
        img_hdu = hdul["IMAGE"]
        h = img_hdu.header
        wcs_header = celestial_header(h)
        mask = hdul["FLAGS"].data.astype(bool) if "FLAGS" in hdul else None
        return Exposure(id=h["EXPID"], field_id=h["FIELDID"], mjd=float(h["MJD-OBS"]),
                        detector=h.get("DETECTOR", "?"), wavelength_um=float(h["WAVELEN"]),
                        image=np.asarray(img_hdu.data, dtype=np.float32), wcs_header=wcs_header,
                        psf_sigma_px=float(h.get("PSFSIG", 1.2)), mask=mask, source_path=str(path))
