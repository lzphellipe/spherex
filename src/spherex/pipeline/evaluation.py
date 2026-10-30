"""Métricas honestas: completude com IC, recall de conhecidos, precisão top-K e calibração."""
from __future__ import annotations

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para proporção: correto mesmo com n pequeno ou p perto de 0/1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def completeness_curve(values: np.ndarray, recovered: np.ndarray, edges: np.ndarray) -> list[dict]:
    """Fração recuperada por faixa (de magnitude ou de taxa), com IC 95%."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (values >= lo) & (values < hi)
        n, k = int(sel.sum()), int(recovered[sel].sum())
        lo_ci, hi_ci = wilson_interval(k, n)
        out.append({"lo": float(lo), "hi": float(hi), "n": n, "recovered": k,
                    "completeness": k / n if n else float("nan"), "ci95": [lo_ci, hi_ci]})
    return out


def magnitude_at_completeness(mags: np.ndarray, recovered: np.ndarray, level: float = 0.5,
                              n_boot: int = 200, seed: int = 0) -> dict | None:
    """Magnitude limite (completude = level) por ajuste logístico, com IC 95% por bootstrap.

    Ajustar a curva inteira é mais estável do que pegar a primeira faixa abaixo
    de 50%, que oscila com poucas fontes por faixa.
    """
    mags, recovered = np.asarray(mags, float), np.asarray(recovered, int)
    if len(mags) < 10 or len(set(recovered.tolist())) < 2:
        return None

    def fit(m: np.ndarray, r: np.ndarray) -> float | None:
        from sklearn.linear_model import LogisticRegression
        if len(set(r.tolist())) < 2:
            return None
        clf = LogisticRegression(C=1e3, max_iter=2000).fit(m.reshape(-1, 1), r)
        a, b = float(clf.intercept_[0]), float(clf.coef_[0, 0])
        if b >= 0:  # completude precisa cair com a magnitude
            return None
        return (np.log(level / (1 - level)) - a) / b

    m50 = fit(mags, recovered)
    if m50 is None:
        return None
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(mags), len(mags))
        v = fit(mags[idx], recovered[idx])
        if v is not None and np.isfinite(v):
            boots.append(v)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None]
    return {"level": level, "mag": float(m50), "ci95": ci, "n": int(len(mags))}


def precision_at_k(scores: np.ndarray, is_true: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return float("nan")
    order = np.argsort(-scores)[:k]
    return float(is_true[order].mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """ECE: diferença média ponderada entre confiança declarada e acerto real."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if not sel.any():
            continue
        conf, acc = float(probs[sel].mean()), float(labels[sel].mean())
        ece += sel.mean() * abs(conf - acc)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "confidence": conf, "accuracy": acc})
    return float(ece), rows


def planet_x_detectability(mag_limit: float, h_mag: float = 0.0, albedo_note: bool = True) -> dict:
    """Distância máxima (au) em que um corpo de magnitude absoluta H ficaria acima do limite.

    Luz refletida: m ≈ H + 5 log10(r·Δ); com r ≈ Δ ≫ 1 au, m ≈ H + 10 log10(r).
    Aproximação de ordem de grandeza (ignora fase e térmico; no IR próximo a
    reflexão domina para corpos frios). Serve para explicar o limite, não para afirmar detecção.
    """
    r_max = 10 ** ((mag_limit - h_mag) / 10.0)
    return {"mag_limit": mag_limit, "H": h_mag, "r_max_au": float(r_max),
            "note": "m ≈ H + 10·log10(r) para r≈Δ≫1 au; brilho refletido cai com ~1/r⁴"}
