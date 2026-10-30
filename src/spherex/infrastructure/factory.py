"""Monta os adapters a partir da configuração (único lugar que conhece as implementações)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spherex.domain.ports import ExposureSource, KnownObjectSource, StaticCatalog


@dataclass
class Sources:
    exposures: ExposureSource
    known: KnownObjectSource
    static: StaticCatalog
    fields: list[dict]


def build_sources(cfg: dict, root: Path = Path(".")) -> Sources:
    from spherex.infrastructure.storage import CachedExposureSource

    raw = root / cfg["paths"]["data_raw"]
    if cfg["source"] == "synthetic":
        from spherex.infrastructure.synthetic import SyntheticSky
        sky = SyntheticSky(cfg["synthetic"], seed=cfg["seed"])
        return Sources(CachedExposureSource(sky, raw / "synthetic"), sky, sky, sky.field_specs())
    if cfg["source"] == "irsa":
        from spherex.infrastructure.allwise_client import AllWiseCatalog
        from spherex.infrastructure.irsa_client import IrsaSpherexSource
        from spherex.infrastructure.jpl_clients import HorizonsEphemeris, SbIdentKnownObjects
        eph = HorizonsEphemeris()
        return Sources(CachedExposureSource(IrsaSpherexSource(), raw / "irsa"),
                       SbIdentKnownObjects(eph), AllWiseCatalog(), cfg["fields"])
    raise ValueError(f"source desconhecida: {cfg['source']}")
