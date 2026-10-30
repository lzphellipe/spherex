"""Registro de progresso do pipeline: retoma de onde parou, nunca refaz à toa.

Local no repo: src/spherex/infrastructure/ledger.py

Cada unidade de trabalho é (stage, tile). O ledger guarda status, hash da
configuração e caminho do resultado em SQLite (modo WAL, resistente a queda).

- Se o processo morrer, ao rodar de novo só as unidades pendentes executam.
- Se a configuração mudar (limiar, versão do modelo), o config_hash muda e a
  unidade é reprocessada. A função de cada estágio deve incluir o config_hash
  no nome do arquivo de saída (ArtifactStore faz isso), para nunca sobrescrever.
- Resultados são gravados de forma atômica antes de marcar "done".
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS work (
                stage TEXT, tile TEXT, config_hash TEXT, status TEXT,
                output TEXT, error TEXT, updated_at TEXT,
                PRIMARY KEY (stage, tile, config_hash))""")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def is_done(self, stage: str, tile: str, chash: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT status, output FROM work WHERE stage=? AND tile=? AND config_hash=?",
                            (stage, tile, chash)).fetchone()
        return bool(row and row[0] == "done" and row[1] and Path(row[1]).exists())

    def _mark(self, stage, tile, chash, status, output=None, error=None):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO work VALUES (?,?,?,?,?,?,?)",
                      (stage, tile, chash, status, output, error, time.strftime("%Y-%m-%dT%H:%M:%S")))

    def run(self, stage: str, tiles: Iterable[str], cfg: dict,
            fn: Callable[[str, dict], Path]) -> dict[str, int]:
        """Executa fn(tile, cfg) -> caminho do resultado, só para o que falta."""
        chash = config_hash(cfg)
        stats = {"skipped": 0, "done": 0, "failed": 0}
        for tile in tiles:
            if self.is_done(stage, tile, chash):
                stats["skipped"] += 1
                continue
            self._mark(stage, tile, chash, "running")
            try:
                out = fn(tile, cfg)  # fn deve gravar com atomic_write
                self._mark(stage, tile, chash, "done", output=str(out))
                stats["done"] += 1
            except Exception as e:  # uma falha não derruba o lote inteiro
                self._mark(stage, tile, chash, "failed", error=repr(e))
                stats["failed"] += 1
        return stats

    def summary(self, stage: str) -> list[tuple[str, int]]:
        with self._conn() as c:
            return c.execute("SELECT status, COUNT(*) FROM work WHERE stage=? GROUP BY status",
                             (stage,)).fetchall()
