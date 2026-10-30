"""Céu sintético com as dificuldades reais do SPHEREx, para testar sem rede.

Implementa ExposureSource, KnownObjectSource e StaticCatalog. Inclui de propósito:
- estrelas com cor (lei de potência em λ): comparar épocas de bandas diferentes
  gera resíduos falsos, que o casamento de banda precisa evitar;
- apontamento ligeiramente diferente por época: exige reprojeção;
- raios cósmicos e uma coluna ruim: negativos para o classificador real/bogus;
- catálogo estático incompleto (~92%), como um AllWISE real.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spherex.domain.models import Exposure, KnownObject
from spherex.pipeline.geometry import make_wcs_header, offsets_to_world, world_to_pixel
from spherex.pipeline.injection import add_gaussian_sources, mag_to_flux, make_injection_plan, plan_positions

REF_WAVELENGTH_UM = 1.10


@dataclass
class _Field:
    id: str
    ra: float
    dec: float
    radius_deg: float
    stars: dict[str, np.ndarray]
    movers: list[dict]
    epochs: list[dict] = field(default_factory=list)


class SyntheticSky:
    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.fields: dict[str, _Field] = {}
        self._build()

    # ---------- construção ----------
    def _build(self) -> None:
        c = self.cfg
        shape = tuple(c["shape"])
        half_deg = 0.5 * min(shape) * c["pixel_scale_arcsec"] / 3600.0
        for f in range(c["n_fields"]):
            fid = f"syn_{f:03d}"
            ra0, dec0 = 30.0 + 20.0 * f, -5.0 + 3.0 * f
            n = c["n_stars"]
            r = half_deg * 3600 * 1.1 * np.sqrt(self.rng.uniform(size=n))
            th = self.rng.uniform(0, 2 * np.pi, n)
            sra, sdec = offsets_to_world(r * np.cos(th), r * np.sin(th), ra0, dec0)
            stars = {
                "ra": sra, "dec": sdec,
                "mag": self.rng.uniform(*c["star_mag"], n),
                "slope": self.rng.uniform(*c["spectral_slope_range"], n),
                "in_catalog": self.rng.uniform(size=n) < 0.92,
            }
            t0 = 61000.0 + 10 * f
            travel = c["epoch_spacing_days"] * (c["n_epochs"] - 1)
            movers = make_injection_plan(fid, ra0, dec0, half_deg * 0.9, t0, c["n_movers"],
                                         tuple(c["mover_mag"]), tuple(c["mover_rate_arcsec_per_day"]),
                                         self.rng, travel_days=travel)
            for m in movers:
                m["name"] = m["name"].replace("inj_", "ast_")
            fld = _Field(fid, ra0, dec0, half_deg, stars, movers)
            for e in range(c["n_epochs"]):
                fld.epochs.append({
                    "id": f"{fid}_e{e}",
                    "mjd": t0 + e * c["epoch_spacing_days"],
                    "wavelength_um": float(c["wavelength_um"][e % len(c["wavelength_um"])]),
                    "dx_arcsec": float(self.rng.uniform(-12, 12)),
                    "dy_arcsec": float(self.rng.uniform(-12, 12)),
                    "rot_deg": float(self.rng.uniform(-0.4, 0.4)),
                    "seed": int(self.rng.integers(1 << 31)),
                })
            self.fields[fid] = fld

    def field_specs(self) -> list[dict]:
        return [{"id": f.id, "ra": f.ra, "dec": f.dec, "radius_deg": f.radius_deg} for f in self.fields.values()]

    # ---------- ExposureSource ----------
    def list_exposures(self, field_id: str, ra: float = 0, dec: float = 0, radius_deg: float = 0) -> list[str]:
        return [e["id"] for e in self.fields[field_id].epochs]

    def load(self, exposure_id: str) -> Exposure:
        fid = exposure_id.rsplit("_e", 1)[0]
        fld = self.fields[fid]
        ep = next(e for e in fld.epochs if e["id"] == exposure_id)
        c = self.cfg
        shape = tuple(c["shape"])
        rng = np.random.default_rng(ep["seed"])
        cra, cdec = offsets_to_world(np.array([ep["dx_arcsec"]]), np.array([ep["dy_arcsec"]]), fld.ra, fld.dec)
        hdr = make_wcs_header(float(cra[0]), float(cdec[0]), shape, c["pixel_scale_arcsec"], ep["rot_deg"])
        img = np.full(shape, 100.0)
        sigma = c["psf_sigma_px"]

        # estrelas: fluxo depende do comprimento de onda (cor)
        s = fld.stars
        x, y = world_to_pixel(hdr, s["ra"], s["dec"])
        color = (ep["wavelength_um"] / REF_WAVELENGTH_UM) ** s["slope"]
        add_gaussian_sources(img, x, y, mag_to_flux(s["mag"], c["zeropoint"]) * color, sigma)

        # asteroides "conhecidos"
        movers = plan_positions(fld.movers, ep["mjd"], injected=False)
        mx, my = world_to_pixel(hdr, np.array([m.ra for m in movers]), np.array([m.dec for m in movers]))
        add_gaussian_sources(img, mx, my, mag_to_flux([m.mag for m in movers], c["zeropoint"]), sigma)

        img += rng.normal(0, c["noise_sigma"], shape)
        mask = np.zeros(shape, bool)
        # raios cósmicos: blobs de 2x2 muito nítidos
        for _ in range(8):
            cx, cy = rng.integers(2, shape[1] - 3), rng.integers(2, shape[0] - 3)
            img[cy:cy + 2, cx:cx + 2] += rng.uniform(150, 600)
        # coluna ruim, marcada na máscara (como nos FLAGS reais)
        col = int(rng.integers(5, shape[1] - 5))
        img[:, col] += 60.0
        mask[:, col] = True

        return Exposure(id=exposure_id, field_id=fid, mjd=ep["mjd"], detector="SYN-D1",
                        wavelength_um=ep["wavelength_um"], image=img.astype(np.float32),
                        wcs_header=hdr, psf_sigma_px=sigma, mask=mask)

    # ---------- KnownObjectSource ----------
    def known_objects(self, exposure: Exposure) -> list[KnownObject]:
        return plan_positions(self.fields[exposure.field_id].movers, exposure.mjd, injected=False)

    # ---------- StaticCatalog ----------
    def sources(self, ra: float, dec: float, radius_deg: float) -> list[tuple[float, float]]:
        out = []
        for f in self.fields.values():
            s = f.stars
            sel = s["in_catalog"]
            out += list(zip(s["ra"][sel].tolist(), s["dec"][sel].tolist(), strict=True))
        return out
