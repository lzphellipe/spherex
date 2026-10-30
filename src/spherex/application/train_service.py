"""Treino do classificador real/bogus a partir das saídas de detecção."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spherex.application.detect_service import detect_hash, load_field_output, output_dir
from spherex.config import section_hash
from spherex.infrastructure.checkpoint import CheckpointManager, atomic_write_json
from spherex.pipeline.models import grouped_split, train_baseline, train_cnn


def train_hash(cfg: dict) -> str:
    """Identidade do experimento. `epochs` e `keep_last` ficam de fora: estender o treino
    precisa retomar o mesmo experimento, não começar outro do zero."""
    model = {k: v for k, v in cfg["model"].items() if k not in ("epochs", "keep_last")}
    return section_hash({"seed": cfg["seed"], "model": model}, "model") + "_" + detect_hash(cfg)


def gather(cfg: dict, root: Path, field_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dfs, stamps, feats = [], [], []
    for fid in field_ids:
        out = output_dir(cfg, root, fid)
        if not (out / "_SUCCESS.json").exists():
            continue
        o = load_field_output(out)
        if o["detections"].empty:
            continue
        dfs.append(o["detections"])
        stamps.append(o["stamps"])
        feats.append(o["feats"])
    if not dfs:
        raise RuntimeError("Nenhuma saída de detecção encontrada: rode `spherex detect` antes.")
    return pd.concat(dfs, ignore_index=True), np.concatenate(stamps), np.concatenate(feats)


def run_training(cfg: dict, root: Path, field_ids: list[str], val_fraction: float = 0.34) -> dict:
    df, stamps, feats = gather(cfg, root, field_ids)
    y = df["label"].to_numpy().astype(int)
    groups = df["field_id"].to_numpy()
    tr, va = grouped_split(groups, val_fraction, cfg["seed"])
    th = train_hash(cfg)
    ckdir = root / cfg["paths"]["checkpoints"] / th
    ckpt = CheckpointManager(ckdir, keep_last=cfg["model"]["keep_last"], remote=cfg["backup"]["remote"],
                             metric=cfg["model"]["metric"])
    base_auc, _ = train_baseline(feats, y, tr, va)
    net, auc, epochs, resumed, val_probs = train_cnn(stamps, y, tr, va, cfg, ckpt, cfg["seed"])
    report = {
        "train_hash": th, "n_train": int(len(tr)), "n_val": int(len(va)),
        "pos_rate_train": float(y[tr].mean()), "pos_rate_val": float(y[va].mean()),
        "val_fields": sorted(set(groups[va].tolist())), "train_fields": sorted(set(groups[tr].tolist())),
        "cnn_val_auc": auc, "baseline_logreg_val_auc": base_auc,
        "epochs_run": epochs, "resumed_from_epoch": resumed, "checkpoint_dir": str(ckdir),
    }
    rep_dir = root / cfg["paths"]["reports"]
    atomic_write_json(rep_dir / f"train_{th}.json", report)
    np.savez(rep_dir / f"val_predictions_{th}.npz", probs=val_probs, labels=y[va])
    return report


def load_model(cfg: dict, root: Path):
    from spherex.pipeline.models import RealBogusNet
    ckpt = CheckpointManager(root / cfg["paths"]["checkpoints"] / train_hash(cfg), keep_last=cfg["model"]["keep_last"])
    state = ckpt.load_best()
    if state is None:
        raise RuntimeError("Nenhum modelo treinado para esta configuração: rode `spherex train`.")
    net = RealBogusNet(cfg["stamps"]["size"])
    net.load_state_dict(state["model"])
    net.eval()
    return net


def model_version(cfg: dict) -> str:
    return f"realbogus-{train_hash(cfg)}"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


