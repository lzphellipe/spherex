"""Fila de revisão humana com aprendizado ativo, consenso e fila de jobs (SQLite, WAL)."""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from spherex.domain.models import Label

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
  id TEXT PRIMARY KEY, field_id TEXT, score REAL, n INTEGER, rate REAL, rms REAL,
  matched_known TEXT, payload TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS labels (
  candidate_id TEXT, reviewer TEXT, label TEXT, created_at TEXT,
  PRIMARY KEY (candidate_id, reviewer));
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, status TEXT, created_at TEXT,
  started_at TEXT, finished_at TEXT, result TEXT);
"""


class ReviewStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- candidatos ----------
    def upsert_candidates(self, cands: list[dict]) -> int:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as c:
            c.executemany(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "score=excluded.score, payload=excluded.payload, updated_at=excluded.updated_at",
                [(x["id"], x["field_id"], x["score"], x["n"], x["rate"], x["rms"], x.get("matched_known"),
                  json.dumps({k: x[k] for k in ("detection_ids", "exposure_ids", "stamps", "provenance")}), now)
                 for x in cands])
        return len(cands)

    def get(self, cid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
            if not row:
                return None
            labels = [dict(r) for r in c.execute("SELECT reviewer, label, created_at FROM labels WHERE candidate_id=?",
                                                 (cid,))]
        d = dict(row)
        d.update(json.loads(d.pop("payload")))
        d["labels"] = labels
        d["consensus"] = consensus([lab["label"] for lab in labels])
        return d

    def queue(self, limit: int = 50, include_reviewed: bool = False) -> list[dict]:
        """Aprendizado ativo: primeiro os candidatos em que o modelo está mais incerto (|p − 0,5| menor)."""
        where = "" if include_reviewed else "WHERE id NOT IN (SELECT DISTINCT candidate_id FROM labels)"
        with self._conn() as c:
            rows = c.execute(f"SELECT id, field_id, score, n, rate, rms, matched_known FROM candidates {where} "
                             "ORDER BY ABS(score - 0.5) ASC, score DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def top(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, field_id, score, n, rate, rms, matched_known FROM candidates "
                             "ORDER BY score DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def add_label(self, cid: str, reviewer: str, label: Label) -> None:
        with self._conn() as c:
            if not c.execute("SELECT 1 FROM candidates WHERE id=?", (cid,)).fetchone():
                raise KeyError(cid)
            c.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?)",
                      (cid, reviewer, label.value, time.strftime("%Y-%m-%dT%H:%M:%S")))

    def labeled(self) -> list[tuple[str, str]]:
        """(candidate_id, rótulo de consenso) para realimentar o treino."""
        with self._conn() as c:
            rows = c.execute("SELECT candidate_id, label FROM labels").fetchall()
        by: dict[str, list[str]] = {}
        for r in rows:
            by.setdefault(r["candidate_id"], []).append(r["label"])
        return [(k, consensus(v)) for k, v in by.items() if consensus(v)]

    # ---------- jobs (a API só enfileira; o worker executa) ----------
    def enqueue(self, kind: str) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO jobs (kind, status, created_at) VALUES (?, 'queued', ?)",
                            (kind, time.strftime("%Y-%m-%dT%H:%M:%S")))
            return int(cur.lastrowid)

    def claim_job(self) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return None
            c.execute("UPDATE jobs SET status='running', started_at=? WHERE id=? AND status='queued'",
                      (time.strftime("%Y-%m-%dT%H:%M:%S"), row["id"]))
            return dict(row)

    def finish_job(self, job_id: int, ok: bool, result: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE jobs SET status=?, finished_at=?, result=? WHERE id=?",
                      ("done" if ok else "failed", time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(result), job_id))

    def job(self, job_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def consensus(labels: list[str], min_votes: int = 1) -> str | None:
    """Maioria simples; empate sem decisão retorna None (vai para mais revisores)."""
    if len(labels) < min_votes:
        return None
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]
