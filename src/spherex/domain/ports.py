"""Portas (interfaces) do domínio. A infraestrutura implementa; o pipeline consome.

Trocar IRSA real por cache ou céu sintético é trocar o adapter, nunca o pipeline.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spherex.domain.models import Exposure, KnownObject


@runtime_checkable
class ExposureSource(Protocol):
    def list_exposures(self, field_id: str, ra: float, dec: float, radius_deg: float) -> list[str]:
        """IDs das exposições que cobrem o campo, ordenados por época."""

    def load(self, exposure_id: str) -> Exposure:
        """Carrega a exposição (pixels + WCS + metadados)."""


@runtime_checkable
class KnownObjectSource(Protocol):
    def known_objects(self, exposure: Exposure) -> list[KnownObject]:
        """Objetos do Sistema Solar conhecidos previstos dentro da exposição."""


@runtime_checkable
class StaticCatalog(Protocol):
    def sources(self, ra: float, dec: float, radius_deg: float) -> list[tuple[float, float]]:
        """(ra, dec) de fontes estáticas (ex.: AllWISE) no campo."""


@runtime_checkable
class EphemerisSource(Protocol):
    def observer_position(self, mjd: float) -> tuple[float, float, float]:
        """Posição heliocêntrica do observador (SPHEREx) em au, eclíptica J2000."""


@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, state: dict, step: int, metrics: dict | None = None) -> Path: ...
    def load_latest(self) -> tuple[dict, int] | None: ...
    def load_best(self) -> dict | None: ...


@runtime_checkable
class WorkLedger(Protocol):
    def run(self, stage: str, tiles: Iterable[str], cfg: dict,
            fn: Callable[[str, dict], Path]) -> dict[str, int]: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def write_table(self, name: str, rows: list[dict[str, Any]], config_hash: str) -> Path: ...
    def read_table(self, path: Path) -> list[dict[str, Any]]: ...
