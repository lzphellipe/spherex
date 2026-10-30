"""CLI: spherex detect | train | eval | demo | serve | worker | status."""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import typer

from spherex.config import load_config

app = typer.Typer(add_completion=False,
                  help="SPHEREx Moving Objects — candidatos para revisão humana, nunca detecções.")
ConfigOpt = typer.Option(None, "--config", "-c", help="YAML que sobrescreve configs/default.yaml")
RootOpt = typer.Option(Path("."), "--root", help="Raiz do projeto (onde ficam data/ e artifacts/)")
SourceOpt = typer.Option(None, "--source", help="synthetic | irsa (sobrescreve a config)")


def _setup(config: Path | None, source: str | None) -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return load_config(config, {"source": source} if source else None)


def _field_ids(cfg: dict, root: Path) -> list[str]:
    from spherex.infrastructure.factory import build_sources
    return [f["id"] for f in build_sources(cfg, root).fields]


@app.command()
def detect(config: Path | None = ConfigOpt, root: Path = RootOpt, source: str | None = SourceOpt) -> None:
    """Pré-processamento, detecção e linking por campo (retomável)."""
    from spherex.application.detect_service import run_detection
    from spherex.infrastructure.factory import build_sources
    cfg = _setup(config, source)
    stats = run_detection(cfg, build_sources(cfg, root), root)
    typer.echo(json.dumps(stats))
    if stats["failed"]:
        raise typer.Exit(code=1)


@app.command()
def train(config: Path | None = ConfigOpt, root: Path = RootOpt, source: str | None = SourceOpt) -> None:
    """Treina a CNN real/bogus (com baseline) e retoma do último checkpoint íntegro se houver."""
    from spherex.application.train_service import run_training
    cfg = _setup(config, source)
    typer.echo(json.dumps(run_training(cfg, root, _field_ids(cfg, root)), indent=2))


@app.command(name="eval")
def evaluate(config: Path | None = ConfigOpt, root: Path = RootOpt, source: str | None = SourceOpt) -> None:
    """Completude, recall de conhecidos, precisão, calibração; publica candidatos na fila de revisão."""
    from spherex.application.eval_service import run_evaluation
    from spherex.application.review_service import ReviewStore
    cfg = _setup(config, source)
    res = run_evaluation(cfg, root, _field_ids(cfg, root))
    ReviewStore(root / cfg["paths"]["review_db"]).upsert_candidates(res["candidates"])
    rep = res["report"]
    summary = {k: rep[k] for k in ("disclaimer", "n_candidates", "precision_at_10", "known_objects",
                                   "injections", "mag_at_50pct_completeness")}
    summary["ece"] = rep["calibration"]["ece"]
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command()
def demo(root: Path = RootOpt) -> None:
    """Roda tudo no céu sintético (offline): detect → train → eval."""
    t = time.time()
    detect(None, root, "synthetic")
    train(None, root, "synthetic")
    evaluate(None, root, "synthetic")
    typer.echo(f"demo concluída em {time.time() - t:.0f}s — relatórios em artifacts/reports/")


@app.command()
def status(config: Path | None = ConfigOpt, root: Path = RootOpt) -> None:
    """Resumo do ledger por estágio."""
    from spherex.infrastructure.ledger import Ledger
    cfg = _setup(config, None)
    typer.echo(json.dumps({"detect": Ledger(root / cfg["paths"]["ledger"]).summary("detect")}))


@app.command()
def serve(config: Path | None = ConfigOpt, root: Path = RootOpt, host: str = "127.0.0.1", port: int = 8000) -> None:
    """API + UI de revisão."""
    import uvicorn

    from spherex.interface.api import create_app
    cfg = _setup(config, None)
    uvicorn.run(create_app(cfg, root), host=host, port=port)


@app.command()
def worker(config: Path | None = ConfigOpt, root: Path = RootOpt, once: bool = False, poll_s: float = 5.0) -> None:
    """Executa os jobs enfileirados pela API (a API nunca processa dentro da requisição)."""
    from spherex.application.review_service import ReviewStore
    cfg = _setup(config, None)
    store = ReviewStore(root / cfg["paths"]["review_db"])
    actions = {"detect": lambda: detect(config, root, None), "train": lambda: train(config, root, None),
               "eval": lambda: evaluate(config, root, None)}
    while True:
        job = store.claim_job()
        if job:
            try:
                actions[job["kind"]]()
                store.finish_job(job["id"], True, {})
            except (Exception, typer.Exit) as e:  # noqa: BLE001 - o worker registra e segue
                store.finish_job(job["id"], False, {"error": repr(e)})
        elif once:
            return
        else:
            time.sleep(poll_s)


if __name__ == "__main__":
    app()
