"""Métricas e treino (split sem vazamento, retomada do checkpoint)."""
import numpy as np

from spherex.infrastructure.checkpoint import CheckpointManager
from spherex.pipeline.evaluation import (
    completeness_curve,
    expected_calibration_error,
    magnitude_at_completeness,
    planet_x_detectability,
    precision_at_k,
    wilson_interval,
)
from spherex.pipeline.models import grouped_split, train_cnn


def test_wilson_interval_reference_values():
    lo, hi = wilson_interval(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    assert wilson_interval(0, 0) != wilson_interval(0, 0)  # NaN


def test_completeness_curve_counts():
    vals = np.array([15.1, 15.2, 16.1, 16.3, 16.4])
    rec = np.array([1, 1, 1, 0, 0])
    rows = completeness_curve(vals, rec, np.array([15.0, 16.0, 17.0]))
    assert [r["n"] for r in rows] == [2, 3]
    assert rows[0]["completeness"] == 1.0 and abs(rows[1]["completeness"] - 1 / 3) < 1e-9


def test_magnitude_limit_recovers_true_value():
    rng = np.random.default_rng(0)
    mags = rng.uniform(14, 22, 600)
    p = 1 / (1 + np.exp(2.0 * (mags - 18.5)))
    rec = rng.uniform(size=600) < p
    m = magnitude_at_completeness(mags, rec, seed=1)
    assert abs(m["mag"] - 18.5) < 0.3
    assert m["ci95"][0] < 18.5 < m["ci95"][1]


def test_calibration_and_precision():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=20000)
    labels = (rng.uniform(size=20000) < probs).astype(int)
    ece, _ = expected_calibration_error(probs, labels)
    assert ece < 0.02
    assert precision_at_k(np.array([0.9, 0.8, 0.1]), np.array([True, False, True]), 2) == 0.5


def test_planet_x_limit_is_order_of_magnitude():
    out = planet_x_detectability(20.0, h_mag=0.0)
    assert abs(out["r_max_au"] - 100.0) < 1e-6


def test_grouped_split_has_no_field_leakage():
    groups = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    tr, va = grouped_split(groups, 0.34, seed=0)
    assert set(groups[tr]).isdisjoint(set(groups[va])) and len(tr) + len(va) == 15


def test_training_resumes_from_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 1, (n, 3, 21, 21)).astype(np.float32)
    x[y == 1, 2, 9:12, 9:12] += 8.0   # positivos: fonte na diferença
    tr, va = np.arange(60), np.arange(60, 80)
    cfg = {"model": {"epochs": 2, "batch_size": 16, "lr": 1e-3}}
    ck = CheckpointManager(tmp_path, keep_last=3)
    _, _, epochs, resumed, _ = train_cnn(x, y, tr, va, cfg, ck, seed=0)
    assert epochs == 2 and resumed is None
    cfg["model"]["epochs"] = 4
    _, auc, epochs, resumed, probs = train_cnn(x, y, tr, va, cfg, CheckpointManager(tmp_path, keep_last=3), seed=0)
    assert resumed == 1 and epochs == 4 and len(probs) == 20   # retomou da época 1, rodou 2 e 3
    assert auc > 0.9
