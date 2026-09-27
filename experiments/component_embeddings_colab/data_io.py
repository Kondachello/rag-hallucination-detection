"""Проверка и загрузка неизменяемого набора из 100 ответов."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FILES = (
    "inputs.no_gold.jsonl", "components.no_gold.jsonl", "labels.csv",
    "folds.csv", "confirmation_features.no_gold.csv",
    "hallugraph_features.no_gold.csv",
)
FEATURES = (
    "entity_log_count", "entity_ungrounded_rate",
    "relation_log_count", "relation_missing", "relation_unsupported_rate",
    "relation_contradicted_rate", "relation_unknown_rate", "relation_not_verified_rate",
    "claim_log_count", "claim_missing", "claim_unsupported_rate",
    "claim_contradicted_rate", "claim_unknown_rate",
)
STATUSES = {
    "entity": ("grounded", "ungrounded"),
    "relation": ("entailed", "unsupported", "contradicted", "unknown", "not_verified"),
    "claim": ("entailed", "unsupported", "contradicted", "unknown"),
}
TYPES = ("entity", "relation", "claim")


def component_key(row: dict) -> str:
    return f"{int(row['source_id'])}:{row['component_id']}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Bundle:
    inputs: list[dict]
    components: list[dict]
    labels: pd.DataFrame
    folds: pd.DataFrame
    confirmation: pd.DataFrame
    hallugraph: pd.DataFrame
    ids: list[int]
    by_id: dict[int, dict]
    by_component: dict[str, dict]
    signature: str

    @property
    def y(self) -> np.ndarray:
        return self.labels.set_index("source_id").loc[self.ids, "hallucination"].to_numpy(dtype=np.int64)

    @property
    def fold_numbers(self) -> np.ndarray:
        return self.folds.set_index("source_id").loc[self.ids, "fold"].to_numpy(dtype=np.int64)

    @property
    def q(self) -> np.ndarray:
        return self.confirmation.set_index("source_id").loc[self.ids, list(FEATURES)].to_numpy(dtype=np.float32)

    @property
    def hallugraph_risk(self) -> np.ndarray:
        df = self.hallugraph.set_index("source_id").loc[self.ids]
        eg = df.EG.to_numpy(dtype=float)
        rp = df.RP_strict.to_numpy(dtype=float)
        no_rel = df.no_relations.to_numpy(dtype=int).astype(bool)
        if np.any(~no_rel & ~np.isfinite(rp)):
            raise ValueError("RP_strict отсутствует при непустом наборе отношений")
        return np.where(no_rel, 1 - eg, 1 - (0.7 * eg + 0.3 * np.nan_to_num(rp)))


def load_bundle(data_dir: Path = DATA) -> Bundle:
    missing = [name for name in FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Нет файлов данных: {missing}")
    files = [data_dir / name for name in FILES]
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in FILES:
            expected = manifest["sources"][name]["sha256"]
            if digest(data_dir / name) != expected:
                raise ValueError(f"Контрольная сумма не совпала: {name}")
    signature = hashlib.sha256("".join(digest(p) for p in files).encode()).hexdigest()
    inputs = read_jsonl(data_dir / FILES[0])
    components = read_jsonl(data_dir / FILES[1])
    labels = pd.read_csv(data_dir / FILES[2])
    folds = pd.read_csv(data_dir / FILES[3])
    confirmation = pd.read_csv(data_dir / FILES[4])
    hallugraph = pd.read_csv(data_dir / FILES[5])
    ids = [int(x["source_id"]) for x in inputs]
    assert len(ids) == len(set(ids)) == 100
    assert len(components) == 2385
    tables = (labels, folds, confirmation, hallugraph)
    for table in tables:
        assert len(table) == 100 and set(table.source_id.astype(int)) == set(ids)
        assert not table.source_id.duplicated().any()
    assert set(folds.fold) == {0, 1, 2, 3, 4}
    assert folds.groupby("fold").size().eq(20).all()
    assert set(labels.hallucination) == {0, 1}
    assert set(FEATURES).issubset(confirmation.columns)
    assert np.isfinite(confirmation[list(FEATURES)].to_numpy(dtype=float)).all()
    by_id = {int(x["source_id"]): x for x in inputs}
    by_component = {}
    counts = {t: 0 for t in TYPES}
    for row in components:
        sid = int(row["source_id"])
        key = component_key(row)
        assert sid in by_id and key not in by_component
        assert row["response_id"] == by_id[sid]["response_id"]
        kind = row["component_type"]
        assert row["confirmation"] in STATUSES[kind]
        assert row["embedding_text"].strip()
        if kind in ("entity", "claim") and row["answer_start"] is not None:
            start, end = int(row["answer_start"]), int(row["answer_end"])
            assert 0 <= start < end <= len(by_id[sid]["answer"])
            if kind == "entity":
                assert by_id[sid]["answer"][start:end].casefold() == row["embedding_text"].casefold()
        counts[kind] += 1
        by_component[key] = row
    assert counts == {"entity": 1009, "relation": 867, "claim": 509}
    for table in tables:
        merged = table.set_index("source_id").loc[ids]
        assert all(merged.response_id.to_numpy() == [by_id[s]["response_id"] for s in ids])
    bundle = Bundle(inputs, components, labels, folds, confirmation,
                    hallugraph, ids, by_id, by_component, signature)
    assert np.isfinite(bundle.hallugraph_risk).all()
    return bundle
