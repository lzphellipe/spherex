"""Adapter AllWISE via TAP do IRSA (fontes estáticas para cross-match)."""
from __future__ import annotations

import csv
import io

import httpx

TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
TABLE = "allwise_p3as_psd"


class AllWiseCatalog:
    def __init__(self, client: httpx.Client | None = None, max_rows: int = 200_000):
        self.client = client or httpx.Client(timeout=300.0)
        self.max_rows = max_rows

    def sources(self, ra: float, dec: float, radius_deg: float) -> list[tuple[float, float]]:
        adql = (f"SELECT TOP {self.max_rows} ra, dec FROM {TABLE} "
                f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1")
        r = self.client.get(TAP_URL, params={"QUERY": adql, "FORMAT": "csv"})
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        return [(float(row["ra"]), float(row["dec"])) for row in reader]
