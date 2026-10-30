"""Classificador real/bogus: CNN pequena + baseline de regressão logística.

Regras: split agrupado por campo (sem vazamento entre treino e validação),
baseline sempre reportado ao lado, seeds fixas, checkpoints seguros e retomáveis.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from spherex.infrastructure.checkpoint import CheckpointManager, install_preemption_handler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class RealBogusNet(nn.Module):
    """Entrada (B, 3, 21, 21): referência, science e diferença normalizadas pelo ruído."""

    def __init__(self, size: int = 21):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compressão assinada: estabiliza o treino entre fontes de brilhos muito diferentes
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        return self.head(self.features(x)).squeeze(-1)


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Rotações de 90° e espelhamentos: o céu não tem orientação preferida."""
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


@dataclass
class TrainResult:
    val_auc: float
    baseline_auc: float
    epochs_run: int
    resumed_from: int | None
    val_probs: np.ndarray
    val_labels: np.ndarray


def grouped_split(groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Separa por campo do céu inteiro: nenhum campo aparece no treino e na validação."""
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_groups = set(uniq[:n_val].tolist())
    is_val = np.array([g in val_groups for g in groups])
    return np.where(~is_val)[0], np.where(is_val)[0]


def train_baseline(feats: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[float, object]:
    scaler = StandardScaler().fit(feats[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(feats[tr]), y[tr])
    p = clf.predict_proba(scaler.transform(feats[va]))[:, 1]
    return _auc(y[va], p), (scaler, clf)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")


def train_cnn(stamps: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, cfg: dict,
              ckpt: CheckpointManager, seed: int) -> tuple[RealBogusNet, float, int, int | None, np.ndarray]:
    set_seed(seed)
    m = cfg["model"]
    net = RealBogusNet(stamps.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=m["lr"])
    start_epoch, resumed = 0, None
    restored = ckpt.load_latest()
    if restored:
        state, step = restored
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optim"])
        torch.set_rng_state(state["torch_rng"])
        # checkpoint parcial (preempção no meio de uma época): refaz essa época inteira
        partial = state.get("partial", False)
        start_epoch, resumed = (step if partial else step + 1), step

    xt, yt = torch.from_numpy(stamps[tr]), torch.from_numpy(y[tr].astype(np.float32))
    xv = torch.from_numpy(stamps[va])
    pos_weight = torch.tensor(max((yt == 0).sum().item(), 1) / max((yt == 1).sum().item(), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    current = {"epoch": start_epoch}  # época em andamento

    def snapshot(partial: bool = False) -> dict:
        return {"model": net.state_dict(), "optim": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                "partial": partial}

    def on_preempt() -> None:
        # salva o progresso da época em andamento, marcado como parcial e sem métrica
        # (nunca pode virar o "best"); a retomada refaz essa época a partir daqui
        ckpt.save(snapshot(partial=True), current["epoch"], {"preempted": True, "partial": True})

    try:
        install_preemption_handler(on_preempt)
    except ValueError:
        pass  # sinais só podem ser instalados na thread principal (ex.: treino dentro do worker da API)
    g = torch.Generator().manual_seed(seed)
    epoch = start_epoch - 1
    for epoch in range(start_epoch, m["epochs"]):
        current["epoch"] = epoch
        net.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(perm), m["batch_size"]):
            idx = perm[i:i + m["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(net(_augment(xt[idx])), yt[idx])
            loss.backward()
            opt.step()
        probs = predict(net, xv)
        auc = _auc(y[va], probs)
        ckpt.save(snapshot(), epoch, {"val_auc": auc, "loss": float(loss.item())})
    best = ckpt.load_best()
    if best:
        net.load_state_dict(best["model"])
    probs = predict(net, xv)
    return net, _auc(y[va], probs), epoch + 1, resumed, probs


@torch.no_grad()
def predict(net: RealBogusNet, x: torch.Tensor | np.ndarray, batch: int = 512) -> np.ndarray:
    net.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    out = [torch.sigmoid(net(x[i:i + batch])) for i in range(0, len(x), batch)]
    return torch.cat(out).numpy() if out else np.zeros(0)
