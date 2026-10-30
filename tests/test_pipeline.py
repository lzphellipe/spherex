"""Geometria, casamento de banda, detecção, linking e shift-and-stack."""
import numpy as np

from spherex.domain.models import Detection
from spherex.infrastructure.synthetic import SyntheticSky
from spherex.pipeline.detection import detect, match_to_truth, remove_static
from spherex.pipeline.geometry import (
    make_wcs_header,
    offsets_to_world,
    pixel_to_world,
    reproject,
    tangent_offsets_arcsec,
    world_to_pixel,
)
from spherex.pipeline.injection import inject, make_injection_plan
from spherex.pipeline.linking import link, shift_and_stack
from spherex.pipeline.preprocess import build_difference


def test_wcs_roundtrip():
    h = make_wcs_header(150.0, 2.0, (64, 64), 6.15, rotation_deg=0.3)
    ra, dec = pixel_to_world(h, [10.3, 40.0], [20.7, 5.5])
    x, y = world_to_pixel(h, ra, dec)
    assert np.allclose(x, [10.3, 40.0], atol=1e-6) and np.allclose(y, [20.7, 5.5], atol=1e-6)


def test_tangent_offsets_roundtrip():
    xi, eta = tangent_offsets_arcsec([150.01, 149.99], [2.02, 1.98], 150.0, 2.0)
    ra, dec = offsets_to_world(xi, eta, 150.0, 2.0)
    assert np.allclose(ra, [150.01, 149.99], atol=1e-9) and np.allclose(dec, [2.02, 1.98], atol=1e-9)


def test_reproject_identity_preserves_image():
    h = make_wcs_header(10.0, 0.0, (40, 40), 6.15)
    img = np.random.default_rng(0).normal(size=(40, 40))
    out = reproject(img, h, h, (40, 40))
    assert np.allclose(out[2:-2, 2:-2], img[2:-2, 2:-2], atol=1e-6)


def _sky_exposures(cfg, seed=3):
    sky = SyntheticSky(cfg["synthetic"], seed=seed)
    f = sky.field_specs()[0]
    return sky, f, [sky.load(i) for i in sky.list_exposures(f["id"])]


def test_band_mismatch_creates_spurious_sources(small_cfg):
    """Comparar épocas de bandas diferentes gera mais resíduos que comparar bandas casadas."""
    sky, f, exps = _sky_exposures(small_cfg)
    e0, e_off, e_same = exps[0], exps[2], exps[4]   # 1.10 µm, 1.30 µm, 1.10 µm
    assert abs(e_off.wavelength_um - e0.wavelength_um) > 0.1
    mismatched = build_difference(e0, [e0, e_off], 0.03, 32, enforce_band_match=False)
    matched = build_difference(e0, [e0, e_same], 0.03, 32)
    n_mis = len(detect(mismatched, 5.0, 4))
    n_ok = len(detect(matched, 5.0, 4))
    assert n_mis > 1.5 * n_ok, (n_mis, n_ok)


def test_epoch_without_band_match_is_skipped(small_cfg):
    _, _, exps = _sky_exposures(small_cfg)
    pair = build_difference(exps[2], exps, 0.03, 32)
    assert pair.skipped_reason and "referência" in pair.skipped_reason
    assert detect(pair, 5.0, 4) == []


def test_injected_fast_movers_are_detected_and_linked(small_cfg):
    sky, f, exps = _sky_exposures(small_cfg)
    rng = np.random.default_rng(7)
    plan = make_injection_plan(f["id"], f["ra"], f["dec"], f["radius_deg"] * 0.8, exps[0].mjd, 5,
                               (15.0, 15.5), (15.0, 25.0), rng, travel_days=exps[-1].mjd - exps[0].mjd)
    truth = {e.id: inject(e, plan, 25.0) for e in exps}
    static = sky.sources(0, 0, 0)
    dets = []
    for e in exps:
        pair = build_difference(e, exps, 0.03, 32)
        d, _ = remove_static(detect(pair, 5.0, 4), static, 8.0, f["ra"], f["dec"])
        dets += d
    found = match_to_truth(dets, [o for v in truth.values() for o in v], 8.0)
    per_obj = {}
    for did, name in found.items():
        per_obj.setdefault(name, set()).add(did.rsplit("_d", 1)[0])
    assert sum(len(v) >= 3 for v in per_obj.values()) >= 4

    trs = link(dets, f["ra"], f["dec"], 3, 60.0, 8.0)
    rates = {p["name"]: p["rate"] for p in plan}
    hits = 0
    for t in trs:
        names = [found.get(d.id) for d in t.detections]
        if names and names[0] and all(n == names[0] for n in names):
            assert abs(t.rate_arcsec_per_day - rates[names[0]]) < 3.0
            hits += 1
    assert hits >= 3


def test_linking_rejects_random_scatter():
    rng = np.random.default_rng(0)
    dets = []
    for e, mjd in enumerate([0.0, 1.0, 2.0, 3.0]):
        for i in range(6):
            ra, dec = offsets_to_world(rng.uniform(-300, 300, 1), rng.uniform(-300, 300, 1), 50.0, 0.0)
            dets.append(Detection(f"e{e}_d{i}", f"e{e}", "f", mjd, float(ra[0]), float(dec[0]), 0, 0, 1, 10))
    assert link(dets, 50.0, 0.0, 3, 60.0, 2.0) == []


def test_shift_and_stack_recovers_faint_slow_source():
    """Fonte abaixo do limiar em cada época isolada aparece no empilhamento ao longo da taxa certa."""
    rng = np.random.default_rng(1)
    n, shape, noise, amp = 8, (48, 48), 1.0, 1.4
    mjds = list(range(n))
    vx, vy = 0.5, 0.25
    diffs = []
    for t in mjds:
        img = rng.normal(0, noise, shape)
        x, y = 20 + vx * t, 20 + vy * t
        img[int(round(y)), int(round(x))] += amp * 3
        diffs.append(img)
    single = max(d.max() / noise for d in diffs)
    rates = np.array([(0.0, 0.0), (vx, vy), (-vx, vy), (1.0, 1.0)])
    snr, idx = shift_and_stack(diffs, mjds, rates, noise)
    assert snr[20, 20] > 6 and idx[20, 20] == 1
    assert snr[20, 20] > single
