"""Ponta a ponta no céu sintético: CLI, retomada, fila de revisão, API e worker."""
import json

import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from spherex.interface.api import create_app
from spherex.interface.cli import app

SMALL = {
    "synthetic": {"n_fields": 3, "shape": [96, 96], "n_stars": 60, "n_movers": 4},
    "injection": {"n_per_field": 8},
    "model": {"epochs": 3, "batch_size": 32},
}


def test_full_pipeline_cli_api_and_worker(tmp_path, small_cfg):
    cfg_path = tmp_path / "small.yaml"
    cfg_path.write_text(yaml.safe_dump(SMALL))
    run = CliRunner()
    common = ["--config", str(cfg_path), "--root", str(tmp_path)]

    r = run.invoke(app, ["detect", *common])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output.strip().splitlines()[-1]) == {"skipped": 0, "done": 3, "failed": 0}
    # segunda execução: nada é refeito
    r = run.invoke(app, ["detect", *common])
    assert json.loads(r.output.strip().splitlines()[-1])["skipped"] == 3

    r = run.invoke(app, ["train", *common])
    assert r.exit_code == 0, r.output
    rep = json.loads(r.output[r.output.index("{"):])
    assert set(rep["train_fields"]).isdisjoint(rep["val_fields"])
    assert rep["baseline_logreg_val_auc"] == rep["baseline_logreg_val_auc"]  # não é NaN

    r = run.invoke(app, ["eval", *common])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.output[r.output.index("{"):])
    assert "NÃO detecta" in summary["disclaimer"]
    assert summary["n_candidates"] > 0

    reports = tmp_path / "artifacts" / "reports"
    assert list(reports.glob("completeness_*.png")) and list(reports.glob("eval_*.json"))

    client = TestClient(create_app(small_cfg, tmp_path))
    assert client.get("/health").json()["ok"]
    queue = client.get("/candidates?mode=queue").json()
    assert queue
    cid = queue[0]["id"]
    detail = client.get(f"/candidates/{cid}").json()
    assert detail["provenance"]["exposure_ids"] and len(detail["stamps"][0]) == 3
    assert client.post(f"/candidates/{cid}/labels", json={"reviewer": "luiz", "label": "real"}).status_code == 201
    assert client.post(f"/candidates/{cid}/labels", json={"reviewer": "", "label": "real"}).status_code == 422
    assert client.post("/candidates/nao-existe/labels", json={"reviewer": "x", "label": "star"}).status_code == 404
    assert client.get(f"/candidates/{cid}").json()["consensus"] == "real"
    assert cid not in [c["id"] for c in client.get("/candidates?mode=queue").json()]
    m = client.get("/metrics").json()
    assert "completeness_by_mag" in m and client.get(m["completeness_png"]).status_code == 200
    assert client.get("/reports/../../etc/passwd").status_code == 404
    assert client.get("/").status_code == 200

    job = client.post("/jobs", json={"kind": "eval"}).json()
    assert job["status"] == "queued"
    r = run.invoke(app, ["worker", *common, "--once"])
    assert r.exit_code == 0, r.output
    assert client.get(f"/jobs/{job['id']}").json()["status"] == "done"


def test_sigterm_during_training_saves_and_resumes(tmp_path):
    """Simula preempção da instância spot: SIGTERM no meio do treino → retomada do checkpoint."""
    import signal
    import subprocess
    import sys
    import time

    cfg = dict(SMALL) | {"model": {"epochs": 400, "batch_size": 32}}
    cfg_path = tmp_path / "long.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    common = ["--config", str(cfg_path), "--root", str(tmp_path)]
    assert CliRunner().invoke(app, ["detect", *common]).exit_code == 0

    proc = subprocess.Popen([sys.executable, "-m", "spherex.interface.cli", "train", *common],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + 120
    manifests = []
    while time.time() < deadline and not manifests:
        manifests = list((tmp_path / "artifacts" / "checkpoints").glob("*/manifest.json"))
        time.sleep(0.2)
    assert manifests, "nenhum checkpoint foi gravado"
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=60)
    assert proc.returncode == 128 + signal.SIGTERM
    manifest = json.loads(manifests[0].read_text())
    latest = max(manifest["checkpoints"], key=lambda c: c["step"])
    assert latest["metrics"].get("preempted") and latest["metrics"].get("partial")
    assert manifest["best"] is None or not manifest["best"]["metrics"].get("partial")
    last_step = latest["step"]

    # estender o treino (mais épocas) retoma o MESMO experimento e refaz a época interrompida
    cfg["model"]["epochs"] = last_step + 3
    cfg_path.write_text(yaml.safe_dump(cfg))
    r = CliRunner().invoke(app, ["train", *common])
    assert r.exit_code == 0, r.output
    rep = json.loads(r.output[r.output.index("{"):])
    assert rep["resumed_from_epoch"] == last_step and rep["epochs_run"] == last_step + 3
    assert rep["checkpoint_dir"].endswith(manifests[0].parent.name)


def test_tracklet_truth_ignores_missing_matches():
    from spherex.application.eval_service import _tracklet_truth
    truth = {"a": "ast_1", "b": float("nan"), "c": "ast_1", "d": None}
    assert _tracklet_truth(["a", "b", "c", "d"], truth) == "ast_1"
    assert _tracklet_truth(["b", "d"], truth) is None
