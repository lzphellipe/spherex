"""API stateless: lê resultados pré-calculados, grava rótulos e enfileira jobs."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from spherex.application.review_service import ReviewStore
from spherex.interface.schemas import CandidateDetail, CandidateSummary, JobIn, JobOut, LabelIn

WEB = Path(__file__).parent / "web"
DISCLAIMER = "Candidatos para revisão humana. Este sistema NÃO detecta nem identifica o Planeta X."


def create_app(cfg: dict, root: Path) -> FastAPI:
    store = ReviewStore(root / cfg["paths"]["review_db"])
    reports = root / cfg["paths"]["reports"]
    app = FastAPI(title="SPHEREx Moving Objects", description=DISCLAIMER, version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "disclaimer": DISCLAIMER}

    @app.get("/candidates", response_model=list[CandidateSummary])
    def candidates(mode: str = "queue", limit: int = 50) -> list[dict]:
        if mode not in ("queue", "top", "all"):
            raise HTTPException(400, "mode deve ser queue, top ou all")
        limit = max(1, min(limit, 500))
        if mode == "top":
            return store.top(limit)
        return store.queue(limit, include_reviewed=(mode == "all"))

    @app.get("/candidates/{cid}", response_model=CandidateDetail)
    def candidate(cid: str) -> dict:
        c = store.get(cid)
        if not c:
            raise HTTPException(404, "candidato não encontrado")
        return c

    @app.post("/candidates/{cid}/labels", status_code=201)
    def label(cid: str, body: LabelIn) -> dict:
        try:
            store.add_label(cid, body.reviewer, body.label)
        except KeyError as e:
            raise HTTPException(404, "candidato não encontrado") from e
        return {"ok": True, "consensus": store.get(cid)["consensus"]}

    @app.get("/metrics")
    def metrics() -> dict:
        files = sorted(reports.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise HTTPException(404, "nenhuma avaliação ainda: rode `spherex eval`")
        rep = json.loads(files[-1].read_text())
        png = files[-1].name.replace("eval_", "completeness_").replace(".json", ".png")
        rep["completeness_png"] = f"/reports/{png}"
        return rep

    @app.get("/reports/{name}")
    def report_file(name: str) -> FileResponse:
        path = (reports / name).resolve()
        if path.parent != reports.resolve() or not path.exists() or path.suffix not in (".png", ".json"):
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/jobs", response_model=JobOut, status_code=202)
    def enqueue(body: JobIn) -> dict:
        return store.job(store.enqueue(body.kind))

    @app.get("/jobs/{job_id}", response_model=JobOut)
    def job(job_id: int) -> dict:
        j = store.job(job_id)
        if not j:
            raise HTTPException(404)
        return j

    return app
