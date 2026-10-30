"""Avaliação e geração de candidatos: completude, recall, precisão, calibração e limite Planet X."""
from __future__ import annotations

import subprocess
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spherex.application.detect_service import detect_hash, load_field_output, output_dir  # noqa: E402
from spherex.application.train_service import load_model, model_version, train_hash  # noqa: E402
from spherex.infrastructure.checkpoint import atomic_write, atomic_write_json  # noqa: E402
from spherex.pipeline.evaluation import (  # noqa: E402
    completeness_curve,
    expected_calibration_error,
    magnitude_at_completeness,
    planet_x_detectability,
    precision_at_k,
)
from spherex.pipeline.models import predict  # noqa: E402

SCORE_THRESHOLD = 0.5


def code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
    except Exception:
        return "unversioned"


def _tracklet_truth(det_ids: list[str], det_truth: dict[str, str | None]) -> str | None:
    names = [det_truth.get(i) for i in det_ids]
    names = [n for n in names if isinstance(n, str) and n]  # parquet devolve NaN onde não houve match
    if not names:
        return None
    # desempate por nome: max(set(...)) dependeria da ordem de hash e mudaria entre execuções
    best, count = sorted(Counter(names).items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best if count >= 2 else None


def run_evaluation(cfg: dict, root: Path, field_ids: list[str]) -> dict:
    net = load_model(cfg, root)
    min_det = cfg["linking"]["min_detections"]
    cands, inj_rows, known_rows = [], [], []
    for fid in field_ids:
        out = output_dir(cfg, root, fid)
        if not (out / "_SUCCESS.json").exists():
            continue
        o = load_field_output(out)
        dets, trks, truth = o["detections"], o["tracklets"], o["truth"]
        probs = predict(net, o["stamps"]) if len(o["stamps"]) else np.zeros(0)
        det_prob = dict(zip(dets["id"], probs, strict=True)) if len(dets) else {}
        det_truth = dict(zip(dets["id"], dets["truth"], strict=True)) if len(dets) else {}
        det_stamp = dict(zip(dets["id"], dets["stamp_idx"], strict=True)) if len(dets) else {}

        recovered_raw, recovered_scored = set(), set()
        for _, t in trks.iterrows():
            ids = list(t["detection_ids"])
            score = float(np.mean([det_prob[i] for i in ids]))
            name = _tracklet_truth(ids, det_truth)
            if name:
                recovered_raw.add(name)
                if score >= SCORE_THRESHOLD:
                    recovered_scored.add(name)
            cands.append({
                "id": t["id"], "field_id": fid, "score": score, "n": int(t["n"]), "rate": float(t["rate"]),
                "rms": float(t["rms"]), "matched_known": name, "is_true": name is not None,
                "detection_ids": ids, "exposure_ids": list(t["exposure_ids"]),
                "stamps": [o["stamps"][det_stamp[i]].tolist() for i in ids[:4]],
                "provenance": {"exposure_ids": list(t["exposure_ids"]), "config_hash": detect_hash(cfg),
                               "model_version": model_version(cfg), "code_version": code_version(),
                               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            })

        if len(truth):
            # um objeto só conta como "observável" se caiu em >= min_det épocas processáveis
            ok_epochs = set(o["epochs"].loc[o["epochs"]["skipped_reason"].isna(), "exposure_id"])
            tv = truth[truth["exposure_id"].isin(ok_epochs)]
            for name, g in tv.groupby("name"):
                row = {"field_id": fid, "name": name, "mag": float(g["mag"].iloc[0]),
                       "rate": float(g["rate"].iloc[0]) if pd.notna(g["rate"].iloc[0]) else np.nan,
                       "n_epochs": len(g), "observable": len(g) >= min_det,
                       "recovered": name in recovered_raw, "recovered_scored": name in recovered_scored}
                (inj_rows if bool(g["injected"].iloc[0]) else known_rows).append(row)

    inj = pd.DataFrame(inj_rows)
    known = pd.DataFrame(known_rows)
    inj_obs = inj[inj["observable"]] if len(inj) else inj
    mag_edges = np.arange(15.0, 21.5, 0.5)
    rate_edges = np.array([1, 2, 4, 8, 16, 32, 64], float)
    curve_mag = completeness_curve(inj_obs["mag"].to_numpy(), inj_obs["recovered_scored"].to_numpy(), mag_edges) \
        if len(inj_obs) else []
    curve_rate = completeness_curve(inj_obs["rate"].to_numpy(), inj_obs["recovered_scored"].to_numpy(), rate_edges) \
        if len(inj_obs) else []

    cdf = pd.DataFrame(cands)
    scores = cdf["score"].to_numpy() if len(cdf) else np.zeros(0)
    is_true = cdf["is_true"].to_numpy() if len(cdf) else np.zeros(0, bool)

    th = train_hash(cfg)
    rep_dir = root / cfg["paths"]["reports"]
    val = np.load(rep_dir / f"val_predictions_{th}.npz")
    ece, rel = expected_calibration_error(val["probs"], val["labels"])
    # limite em magnitude medido só nas taxas que o cadenciamento resolve (>= rate_floor):
    # objetos mais lentos se autossubtraem e entram na curva por taxa, não aqui
    rate_floor = cfg["evaluation"]["rate_floor_arcsec_per_day"]
    fast = inj_obs[inj_obs["rate"] >= rate_floor] if len(inj_obs) else inj_obs
    m50 = magnitude_at_completeness(fast["mag"].to_numpy(), fast["recovered_scored"].to_numpy(),
                                    seed=cfg["seed"]) if len(fast) else None
    known_obs = known[known["observable"]] if len(known) else known

    report = {
        "disclaimer": "Candidatos para revisão humana. Este sistema NÃO detecta nem identifica o Planeta X.",
        "config_hash": detect_hash(cfg), "model_version": model_version(cfg), "code_version": code_version(),
        "n_candidates": int(len(cdf)),
        "precision_at_10": precision_at_k(scores, is_true, 10),
        "precision_at_25": precision_at_k(scores, is_true, 25),
        "precision_all_above_threshold": float(is_true[scores >= SCORE_THRESHOLD].mean())
        if (scores >= SCORE_THRESHOLD).any() else None,
        "known_objects": {"observable": int(len(known_obs)),
                          "recovered": int(known_obs["recovered_scored"].sum()) if len(known_obs) else 0},
        "injections": {"total": int(len(inj)), "observable": int(len(inj_obs)),
                       "recovered_linking": int(inj_obs["recovered"].sum()) if len(inj_obs) else 0,
                       "recovered_after_classifier": int(inj_obs["recovered_scored"].sum()) if len(inj_obs) else 0},
        "completeness_by_mag": curve_mag, "completeness_by_rate": curve_rate,
        "mag_at_50pct_completeness": m50,
        "mag_limit_rate_floor_arcsec_per_day": rate_floor,
        "calibration": {"ece": ece, "bins": rel},
        "planet_x_limit": planet_x_detectability(m50["mag"], h_mag=cfg["evaluation"]["planet_x_h_mag"])
        if m50 else None,
    }
    atomic_write_json(rep_dir / f"eval_{th}.json", report)
    _plot(curve_mag, curve_rate, rep_dir / f"completeness_{th}.png")
    atomic_write(rep_dir / f"candidates_{th}.parquet",
                 lambda p: cdf.drop(columns=["stamps"]).to_parquet(p, index=False) if len(cdf) else
                 pd.DataFrame().to_parquet(p))
    return {"report": report, "candidates": cands}


def _plot(curve_mag: list[dict], curve_rate: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, curve, xl in ((axes[0], curve_mag, "magnitude"), (axes[1], curve_rate, "taxa (arcsec/dia)")):
        rows = [r for r in curve if r["n"]]
        if rows:
            x = [0.5 * (r["lo"] + r["hi"]) for r in rows]
            yv = [r["completeness"] for r in rows]
            lo = [max(0.0, r["completeness"] - r["ci95"][0]) for r in rows]
            hi = [max(0.0, r["ci95"][1] - r["completeness"]) for r in rows]
            ax.errorbar(x, yv, yerr=[lo, hi], fmt="o-", capsize=3)
        ax.set_xlabel(xl)
        ax.set_ylabel("completude (IC 95%)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    if curve_rate:
        from matplotlib.ticker import NullFormatter
        axes[1].set_xscale("log")
        ticks = [1, 2, 4, 8, 16, 32, 64]
        axes[1].set_xticks(ticks, [str(t) for t in ticks])
        axes[1].xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Completude medida por injeção sintética")
    fig.tight_layout()
    atomic_write(path, lambda p: fig.savefig(p, format="png", dpi=120))
    plt.close(fig)
