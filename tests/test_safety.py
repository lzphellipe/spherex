"""Camada de segurança: nada se perde por queda, corrupção, preempção ou erro humano."""
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import pytest

from spherex.infrastructure.checkpoint import CheckpointManager, atomic_write
from spherex.infrastructure.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]


def _pickle_cm(root, keep_last=2):
    return CheckpointManager(root, keep_last=keep_last,
                             save_fn=lambda s, p: p.write_bytes(pickle.dumps(s)),
                             load_fn=lambda p: pickle.loads(p.read_bytes()))


def test_crash_during_save_keeps_previous_checkpoint(tmp_path):
    cm = _pickle_cm(tmp_path)
    cm.save({"w": 1}, 1, {"val_auc": 0.7})
    target = tmp_path / "ckpt_00000001.pt"

    def crash(_p):
        raise RuntimeError("queda de energia")

    with pytest.raises(RuntimeError):
        atomic_write(target, crash)
    assert pickle.loads(target.read_bytes()) == {"w": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupted_latest_falls_back_to_previous(tmp_path):
    cm = _pickle_cm(tmp_path, keep_last=3)
    for step in (1, 2, 3):
        cm.save({"w": step}, step, {"val_auc": 0.5 + step / 10})
    (tmp_path / "ckpt_00000003.pt").write_bytes(b"lixo")
    state, step = cm.load_latest()
    assert step == 2 and state == {"w": 2}


def test_retention_never_deletes_best(tmp_path):
    cm = _pickle_cm(tmp_path, keep_last=2)
    for step, auc in [(1, 0.70), (2, 0.95), (3, 0.80), (4, 0.82)]:
        cm.save({"w": step}, step, {"val_auc": auc})
    names = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert names == ["best.pt", "ckpt_00000003.pt", "ckpt_00000004.pt"]
    assert cm.load_best() == {"w": 2}


def test_ledger_resumes_and_isolates_failures(tmp_path):
    lg = Ledger(tmp_path / "work.db")
    calls = []

    def work(tile, cfg):
        calls.append(tile)
        if tile == "t2" and calls.count("t2") == 1:
            raise RuntimeError("falha transitória")
        out = tmp_path / f"{tile}_{cfg['thr']}.out"
        out.write_text("ok")
        return out

    assert lg.run("detect", ["t1", "t2", "t3"], {"thr": 5}, work) == {"skipped": 0, "done": 2, "failed": 1}
    assert lg.run("detect", ["t1", "t2", "t3"], {"thr": 5}, work) == {"skipped": 2, "done": 1, "failed": 0}
    # config nova: reprocessa sem tocar no resultado antigo
    assert lg.run("detect", ["t1"], {"thr": 4}, work)["done"] == 1
    assert (tmp_path / "t1_5.out").exists() and (tmp_path / "t1_4.out").exists()


def test_ledger_redoes_when_output_file_vanished(tmp_path):
    lg = Ledger(tmp_path / "work.db")

    def work(tile, cfg):
        out = tmp_path / f"{tile}.out"
        out.write_text("ok")
        return out

    lg.run("s", ["a"], {}, work)
    (tmp_path / "a.out").unlink()
    assert lg.run("s", ["a"], {}, work)["done"] == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash indisponível")
def test_guard_script_syntax_and_check_without_backup(tmp_path):
    script = ROOT / "scripts" / "brev_guard.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    (tmp_path / "scripts").mkdir()
    shutil.copy(script, tmp_path / "scripts" / "brev_guard.sh")
    env = os.environ | {"BACKUP_REMOTE": "dummy:remote"}
    r = subprocess.run(["bash", str(tmp_path / "scripts" / "brev_guard.sh"), "check"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "nenhum backup" in r.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash indisponível")
def test_guard_requires_remote(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "brev_guard.sh", tmp_path / "scripts" / "brev_guard.sh")
    env = {k: v for k, v in os.environ.items() if k != "BACKUP_REMOTE"}
    r = subprocess.run(["bash", str(tmp_path / "scripts" / "brev_guard.sh"), "once"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "BACKUP_REMOTE" in r.stderr


def test_raw_cache_is_read_only(tmp_path, small_cfg):
    from spherex.infrastructure.storage import CachedExposureSource
    from spherex.infrastructure.synthetic import SyntheticSky
    sky = SyntheticSky(small_cfg["synthetic"], seed=1)
    src = CachedExposureSource(sky, tmp_path)
    eid = sky.list_exposures(sky.field_specs()[0]["id"])[0]
    src.load(eid)
    path = src.path_for(eid)
    assert path.exists() and oct(path.stat().st_mode)[-3:] == "444"
    again = src.load(eid)
    assert again.source_path == str(path)
