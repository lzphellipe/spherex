"""Pré-processamento: casamento de banda (LVF), reprojeção, fundo e PSF.

Por que o casamento de banda importa: os filtros lineares variáveis do SPHEREx
fazem o comprimento de onda mudar ao longo do detector. Duas exposições do mesmo
céu podem ter observado o campo em bandas diferentes; estrelas com cor mudam de
brilho entre elas e a subtração ingênua gera "fontes" que não existem.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from photutils.background import Background2D, MedianBackground
from scipy.ndimage import gaussian_filter

from spherex.domain.models import Exposure
from spherex.pipeline.geometry import reproject


@dataclass
class DifferencePair:
    science: Exposure
    references: list[Exposure]
    reference_image: np.ndarray      # na grade da science, fundo removido
    science_image: np.ndarray        # fundo removido
    difference: np.ndarray
    noise: float
    skipped_reason: str | None = None


def band_matched_references(exposures: list[Exposure], science: Exposure, tol_um: float) -> list[Exposure]:
    """Outras épocas do mesmo campo com comprimento de onda central dentro da tolerância."""
    return [e for e in exposures
            if e.id != science.id and abs(e.wavelength_um - science.wavelength_um) <= tol_um]


def subtract_background(image: np.ndarray, box: int, mask: np.ndarray | None = None) -> np.ndarray:
    img = np.where(np.isfinite(image), image, np.nan)
    finite = np.isfinite(img)
    bad = (~finite) | (mask if mask is not None else False)
    bkg = Background2D(np.nan_to_num(img, nan=float(np.nanmedian(img))), box, filter_size=3,
                       bkg_estimator=MedianBackground(), mask=bad, exclude_percentile=50.0)
    return img - bkg.background


def match_psf(image: np.ndarray, sigma_from: float, sigma_to: float) -> np.ndarray:
    """Degrada a imagem mais nítida para a PSF mais larga (kernel gaussiano)."""
    if sigma_to <= sigma_from + 1e-6:
        return image
    k = np.sqrt(sigma_to**2 - sigma_from**2)
    filled = np.where(np.isfinite(image), image, 0.0)
    w = gaussian_filter(np.isfinite(image).astype(float), k)
    out = gaussian_filter(filled, k) / np.maximum(w, 1e-6)
    out[~np.isfinite(image)] = np.nan
    return out


def robust_sigma(x: np.ndarray) -> float:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def build_difference(science: Exposure, all_exposures: list[Exposure], tol_um: float,
                     background_box: int, enforce_band_match: bool = True) -> DifferencePair:
    """Referência = mediana das outras épocas casadas em banda, reprojetadas na grade da science."""
    refs = band_matched_references(all_exposures, science, tol_um) if enforce_band_match \
        else [e for e in all_exposures if e.id != science.id]
    shape = science.image.shape
    sci = subtract_background(science.image, background_box, science.mask)
    if not refs:
        nan = np.full(shape, np.nan)
        return DifferencePair(science, [], nan, sci, nan, float("nan"),
                              skipped_reason=f"nenhuma referência dentro de ±{tol_um} µm")
    target_sigma = max([science.psf_sigma_px] + [r.psf_sigma_px for r in refs])
    sci = match_psf(sci, science.psf_sigma_px, target_sigma)
    stack = []
    for r in refs:
        rimg = subtract_background(r.image, background_box, r.mask)
        rimg = match_psf(rimg, r.psf_sigma_px, target_sigma)
        rproj = reproject(rimg, r.wcs_header, science.wcs_header, shape)
        if r.mask is not None:
            mproj = reproject(r.mask.astype(float), r.wcs_header, science.wcs_header, shape)
            rproj[mproj > 0.01] = np.nan
        stack.append(rproj)
    with warnings.catch_warnings():  # pixels sem cobertura em nenhuma referência viram NaN (esperado)
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(np.stack(stack), axis=0) if len(stack) > 1 else stack[0]
    diff = sci - ref
    if science.mask is not None:
        diff[science.mask] = np.nan
    return DifferencePair(science, refs, ref, sci, diff, robust_sigma(diff))
