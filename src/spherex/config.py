"""Carregamento da configuração YAML e hash reprodutível."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Lê o YAML padrão, aplica um YAML do usuário e overrides (merge profundo)."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def section_hash(cfg: dict[str, Any], *sections: str) -> str:
    """Hash estável das seções que afetam um estágio (inclui sempre a seed)."""
    payload = {"seed": cfg.get("seed")} | {s: cfg.get(s) for s in sections}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
