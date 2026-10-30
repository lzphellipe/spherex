"""Leitura/escrita de exposições em FITS (formato interno do cache)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from spherex.domain.models import Exposure
from spherex.infrastructure.checkpoint import atomic_write

WCS_KEYS = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2", "PC1_1", "PC1_2",
            "PC2_1", "PC2_2", "CUNIT1", "CUNIT2", "NAXIS1", "NAXIS2", "LONPOLE", "LATPOLE",
            "RADESYS", "EQUINOX", "A_ORDER", "B_ORDER")


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
        wcs_header = {k: h[k] for k in WCS_KEYS if k in h}
        wcs_header["NAXIS1"], wcs_header["NAXIS2"] = h["NAXIS1"], h["NAXIS2"]
        mask = hdul["FLAGS"].data.astype(bool) if "FLAGS" in hdul else None
        return Exposure(id=h["EXPID"], field_id=h["FIELDID"], mjd=float(h["MJD-OBS"]),
                        detector=h.get("DETECTOR", "?"), wavelength_um=float(h["WAVELEN"]),
                        image=np.asarray(img_hdu.data, dtype=np.float32), wcs_header=wcs_header,
                        psf_sigma_px=float(h.get("PSFSIG", 1.2)), mask=mask, source_path=str(path))
