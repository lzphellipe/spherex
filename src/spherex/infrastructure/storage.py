"""Cache de exposições (dados brutos imutáveis) e armazenamento de tabelas."""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pandas as pd

from spherex.domain.models import Exposure
from spherex.domain.ports import ExposureSource
from spherex.infrastructure.checkpoint import atomic_write
from spherex.infrastructure.fits_io import read_exposure, write_exposure


def _safe_name(exposure_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in exposure_id)


class CachedExposureSource:
    """Embrulha qualquer ExposureSource: baixa uma vez, grava em data/raw e marca somente leitura.

    Se o arquivo em cache existir, a rede nunca é chamada de novo.
    """

    def __init__(self, inner: ExposureSource, raw_dir: Path):
        self.inner = inner
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def list_exposures(self, field_id: str, ra: float, dec: float, radius_deg: float) -> list[str]:
        return self.inner.list_exposures(field_id, ra, dec, radius_deg)

    def path_for(self, exposure_id: str) -> Path:
        return self.raw_dir / f"{_safe_name(exposure_id)}.fits"

    def load(self, exposure_id: str) -> Exposure:
        path = self.path_for(exposure_id)
        if path.exists():
            return read_exposure(path)
        exp = self.inner.load(exposure_id)
        write_exposure(exp, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # dado bruto: somente leitura
        exp.source_path = str(path)
        return exp


class ParquetArtifactStore:
    """Tabelas de resultado com o config_hash no nome: reprocessar nunca sobrescreve."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str, config_hash: str) -> Path:
        return self.root / f"{name}__{config_hash}.parquet"

    def write_table(self, name: str, rows: list[dict[str, Any]], config_hash: str) -> Path:
        path = self.path(name, config_hash)
        df = pd.DataFrame(rows)
        atomic_write(path, lambda p: df.to_parquet(p, index=False))
        return path

    def read_table(self, path: Path) -> list[dict[str, Any]]:
        return pd.read_parquet(path).to_dict("records")
