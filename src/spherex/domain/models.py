"""Entidades do domínio. Sem dependências de I/O, rede ou frameworks."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


@dataclass
class Exposure:
    """Uma exposição de um detector, já com pixels em memória.

    image: fluxo em unidades arbitrárias (array 2D, float32)
    wcs_header: cabeçalho FITS com a WCS (dict), para não acoplar o domínio ao astropy
    wavelength_um: comprimento de onda central no campo (LVF varia ao longo do detector)
    psf_sigma_px: largura estimada da PSF em pixels
    """

    id: str
    field_id: str
    mjd: float
    detector: str
    wavelength_um: float
    image: np.ndarray
    wcs_header: dict[str, Any]
    psf_sigma_px: float = 1.2
    mask: np.ndarray | None = None
    source_path: str | None = None


@dataclass
class Detection:
    id: str
    exposure_id: str
    field_id: str
    mjd: float
    ra: float
    dec: float
    x: float
    y: float
    flux: float
    snr: float
    sign: int = 1  # +1: fonte nova na época mais recente; -1: sumiu da referência


@dataclass
class KnownObject:
    """Objeto conhecido (JPL) ou injetado, com posição prevista numa época."""

    name: str
    mjd: float
    ra: float
    dec: float
    mag: float | None = None
    rate_arcsec_per_day: float | None = None
    injected: bool = False


@dataclass
class Tracklet:
    id: str
    field_id: str
    detections: list[Detection]
    rate_ra_arcsec_per_day: float
    rate_dec_arcsec_per_day: float
    rms_arcsec: float

    @property
    def rate_arcsec_per_day(self) -> float:
        return float(np.hypot(self.rate_ra_arcsec_per_day, self.rate_dec_arcsec_per_day))

    @property
    def n(self) -> int:
        return len(self.detections)


class Label(StrEnum):
    REAL = "real"
    ARTIFACT = "artifact"
    STAR = "star"


@dataclass
class Provenance:
    exposure_ids: list[str]
    config_hash: str
    model_version: str | None
    code_version: str
    created_at: str


@dataclass
class Candidate:
    """Candidato para revisão humana. Nunca é uma detecção confirmada."""

    id: str
    tracklet: Tracklet
    score: float
    provenance: Provenance
    matched_known: str | None = None
    labels: list[Label] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.labels:
            return "pending"
        values = sorted(lab.value for lab in self.labels)
        return max(values, key=values.count)
