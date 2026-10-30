"""Orquestra detecção por campo do céu, retomável via ledger."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spherex.config import section_hash
from spherex.infrastructure.checkpoint import atomic_write, atomic_write_json
from spherex.infrastructure.factory import Sources
from spherex.infrastructure.ledger import Ledger
from spherex.pipeline.detection import detect, match_to_truth, remove_static
from spherex.pipeline.injection import inject, make_injection_plan
from spherex.pipeline.linking import link
from spherex.pipeline.preprocess import build_difference
from spherex.pipeline.stamps import handcrafted_features, triplet

log = logging.getLogger(__name__)
DETECT_SECTIONS = ("source", "synthetic", "fields", "preprocess", "detection", "linking", "stamps", "injection")


def detect_hash(cfg: dict) -> str:
    return section_hash(cfg, *DETECT_SECTIONS)


def output_dir(cfg: dict, root: Path, field_id: str) -> Path:
    """Um diretório por (campo, config_hash): reprocessar com outra config nunca sobrescreve."""
    return root / cfg["paths"]["data_processed"] / f"detect_{field_id}__{detect_hash(cfg)}"


def process_field(field: dict, cfg: dict, sources: Sources, root: Path) -> Path:
    """Detecção completa de um campo. Escreve npz (stamps) e depois parquet (tabelas)."""
    c_det, c_link, c_pre = cfg["detection"], cfg["linking"], cfg["preprocess"]
    fid = field["id"]
    ids = sources.exposures.list_exposures(fid, field["ra"], field["dec"], field["radius_deg"])
    exps = [sources.exposures.load(i) for i in ids]
    for e in exps:
        e.image = e.image.copy()  # injeção só em memória: o dado bruto em cache nunca muda

    # injeção sintética (semente por campo, reprodutível)
    inj = cfg["injection"]
    rng = np.random.default_rng([cfg["seed"], sum(map(ord, fid))])
    travel = (exps[-1].mjd - exps[0].mjd) if exps else 0.0
    plan = make_injection_plan(fid, field["ra"], field["dec"], field["radius_deg"] * 0.85,
                               exps[0].mjd if exps else 0.0, inj["n_per_field"], tuple(inj["mag"]),
                               tuple(inj["rate_arcsec_per_day"]), rng, travel_days=travel)
    zp = cfg["synthetic"]["zeropoint"] if cfg["source"] == "synthetic" else cfg.get("zeropoint", 25.0)
    truth_rows, injected_by_exp, known_by_exp = [], {}, {}
    for e in exps:
        injected_by_exp[e.id] = inject(e, plan, zp) if plan else []
        known_by_exp[e.id] = sources.known.known_objects(e)
        for o in injected_by_exp[e.id] + known_by_exp[e.id]:
            truth_rows.append({"field_id": fid, "exposure_id": e.id, "mjd": e.mjd, "name": o.name,
                               "ra": o.ra, "dec": o.dec, "mag": o.mag, "rate": o.rate_arcsec_per_day,
                               "injected": o.injected})

    static = sources.static.sources(field["ra"], field["dec"], field["radius_deg"] * 1.5)
    det_rows, stamps, feats, all_dets, epoch_rows = [], [], [], [], []
    for e in exps:
        pair = build_difference(e, exps, c_pre["band_tolerance_um"], c_pre["background_box"])
        epoch_rows.append({"exposure_id": e.id, "mjd": e.mjd, "wavelength_um": e.wavelength_um,
                           "n_refs": len(pair.references), "skipped_reason": pair.skipped_reason,
                           "noise": pair.noise})
        if pair.skipped_reason:
            log.info("%s pulada: %s", e.id, pair.skipped_reason)
            continue
        dets = detect(pair, c_det["threshold_sigma"], c_det["npixels"])
        dets, _static = remove_static(dets, static, c_det["static_match_arcsec"], field["ra"], field["dec"])
        truth = injected_by_exp[e.id] + known_by_exp[e.id]
        matched = match_to_truth(dets, truth, c_link["match_radius_arcsec"])
        for d in dets:
            st = triplet(pair, d, cfg["stamps"]["size"])
            stamps.append(st)
            feats.append(handcrafted_features(st, d))
            det_rows.append({"id": d.id, "exposure_id": d.exposure_id, "field_id": fid, "mjd": d.mjd,
                             "ra": d.ra, "dec": d.dec, "x": d.x, "y": d.y, "flux": d.flux, "snr": d.snr,
                             "truth": matched.get(d.id), "label": int(d.id in matched),
                             "stamp_idx": len(stamps) - 1})
        all_dets += dets

    tracklets = link(all_dets, field["ra"], field["dec"], c_link["min_detections"],
                     c_link["max_rate_arcsec_per_day"], c_link["match_radius_arcsec"])
    trk_rows = [{"id": t.id, "field_id": fid, "n": t.n, "rate": t.rate_arcsec_per_day,
                 "rate_ra": t.rate_ra_arcsec_per_day, "rate_dec": t.rate_dec_arcsec_per_day,
                 "rms": t.rms_arcsec, "detection_ids": [d.id for d in t.detections],
                 "exposure_ids": sorted({d.exposure_id for d in t.detections})} for t in tracklets]

    out = output_dir(cfg, root, fid)
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["stamps"]["size"]
    arr = np.stack(stamps) if stamps else np.zeros((0, 3, size, size), np.float32)
    farr = np.stack(feats) if feats else np.zeros((0, 5), np.float32)
    atomic_write(out / "stamps.npz", lambda p: _save_npz(p, arr, farr))
    tables = {"detections": det_rows, "tracklets": trk_rows, "truth": truth_rows, "epochs": epoch_rows}
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        atomic_write(out / f"{name}.parquet", lambda p, df=df: df.to_parquet(p, index=False))
    # manifesto por último: só existe se todo o resto foi gravado
    manifest = {"field_id": fid, "config_hash": detect_hash(cfg), "n_detections": len(det_rows),
                "n_tracklets": len(trk_rows), "files": sorted([*tables.keys(), "stamps"])}
    atomic_write_json(out / "_SUCCESS.json", manifest)
    return out / "_SUCCESS.json"


def _save_npz(path: Path, stamps: np.ndarray, feats: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.savez_compressed(f, stamps=stamps, feats=feats)


def load_field_output(out_dir: Path) -> dict:
    out: dict = {}
    for name in ("detections", "tracklets", "truth", "epochs"):
        out[name] = pd.read_parquet(out_dir / f"{name}.parquet")
    with np.load(out_dir / "stamps.npz") as z:
        out["stamps"], out["feats"] = z["stamps"], z["feats"]
    return out


def run_detection(cfg: dict, sources: Sources, root: Path) -> dict[str, int]:
    ledger = Ledger(root / cfg["paths"]["ledger"])
    fields = {f["id"]: f for f in sources.fields}
    stage_cfg = {s: cfg.get(s) for s in DETECT_SECTIONS} | {"seed": cfg["seed"]}
    return ledger.run("detect", list(fields), stage_cfg,
                      lambda fid, _c: process_field(fields[fid], cfg, sources, root))
