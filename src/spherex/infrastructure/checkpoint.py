"""Checkpoints atômicos, verificáveis e retomáveis para o treino.

Local no repo: src/spherex/infrastructure/checkpoint.py

Garantias:
- Escrita atômica: grava em arquivo temporário, faz fsync e só então renomeia.
  Uma queda no meio do save nunca corrompe o último checkpoint bom.
- Integridade: cada checkpoint tem SHA-256 no manifesto; o load verifica e,
  se o mais recente estiver corrompido, cai para o anterior.
- Retenção: mantém os N mais recentes + o melhor (nunca apaga o "best").
- Preempção: SIGTERM/SIGINT (spot, Ctrl+C, shutdown da instância) dispara
  um save antes de sair.
- Backup remoto: copia cada checkpoint para fora da instância via rclone
  (copy, nunca sync, para nunca propagar deleções).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Executa writer(tmp) e promove tmp -> path de forma atômica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write(path, lambda p: p.write_text(json.dumps(obj, indent=2, sort_keys=True)))


def _default_save(state: dict, path: Path) -> None:
    import torch
    torch.save(state, path)


def _default_load(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class CheckpointManager:
    root: Path
    keep_last: int = 3
    remote: str | None = None          # ex.: "backup:spherex/checkpoints"
    metric: str = "val_auc"            # métrica que define o "best"
    higher_is_better: bool = True
    save_fn: Callable[[dict, Path], None] = _default_save
    load_fn: Callable[[Path], dict] = _default_load

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- manifesto ----------
    def _manifest(self) -> dict:
        p = self.root / MANIFEST
        if p.exists():
            return json.loads(p.read_text())
        return {"checkpoints": [], "best": None}

    def _write_manifest(self, m: dict) -> None:
        atomic_write_json(self.root / MANIFEST, m)

    # ---------- save ----------
    def save(self, state: dict, step: int, metrics: dict | None = None) -> Path:
        metrics = metrics or {}
        path = self.root / f"ckpt_{step:08d}.pt"
        atomic_write(path, lambda p: self.save_fn(state, p))

        m = self._manifest()
        entry = {"file": path.name, "step": step, "sha256": sha256_of(path),
                 "metrics": metrics, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        m["checkpoints"] = [c for c in m["checkpoints"] if c["file"] != path.name] + [entry]

        if self.metric in metrics and self._is_better(metrics[self.metric], m.get("best")):
            best = self.root / "best.pt"
            atomic_write(best, lambda p: p.write_bytes(path.read_bytes()))
            m["best"] = {**entry, "file": best.name, "source": path.name}

        self._prune(m)
        self._write_manifest(m)
        self._backup([path, self.root / MANIFEST] + ([self.root / "best.pt"] if m.get("best") else []))
        return path

    def _is_better(self, value: float, best: dict | None) -> bool:
        if not best or self.metric not in best["metrics"]:
            return True
        prev = best["metrics"][self.metric]
        return value > prev if self.higher_is_better else value < prev

    def _prune(self, m: dict) -> None:
        ordered = sorted(m["checkpoints"], key=lambda c: c["step"])
        for old in ordered[:-self.keep_last] if self.keep_last > 0 else []:
            (self.root / old["file"]).unlink(missing_ok=True)
        m["checkpoints"] = ordered[-self.keep_last:] if self.keep_last > 0 else ordered

    def _backup(self, files: list[Path]) -> None:
        if not self.remote:
            return
        for f in files:
            if f.exists():
                # copy (não sync): o remoto só recebe, nunca perde arquivos.
                subprocess.Popen(["rclone", "copy", str(f), self.remote, "--no-traverse"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---------- load ----------
    def load_latest(self) -> tuple[dict, int] | None:
        """Retorna (state, step) do checkpoint íntegro mais recente, ou None."""
        for entry in sorted(self._manifest()["checkpoints"], key=lambda c: c["step"], reverse=True):
            path = self.root / entry["file"]
            if path.exists() and sha256_of(path) == entry["sha256"]:
                return self.load_fn(path), entry["step"]
            print(f"[checkpoint] {entry['file']} ausente ou corrompido, tentando o anterior")
        return None

    def latest_entry(self) -> dict | None:
        """Metadados (step, métricas) do checkpoint mais recente registrado no manifesto."""
        entries = self._manifest()["checkpoints"]
        return max(entries, key=lambda c: c["step"]) if entries else None

    def load_best(self) -> dict | None:
        best = self._manifest().get("best")
        path = self.root / "best.pt"
        if best and path.exists() and sha256_of(path) == best["sha256"]:
            return self.load_fn(path)
        return None


def install_preemption_handler(on_signal: Callable[[], None]) -> None:
    """Salva antes de sair em SIGTERM (spot/shutdown) e SIGINT (Ctrl+C)."""
    fired = {"done": False}

    def handler(signum, _frame):
        if not fired["done"]:
            fired["done"] = True
            print(f"[checkpoint] sinal {signum} recebido, salvando antes de sair")
            on_signal()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
