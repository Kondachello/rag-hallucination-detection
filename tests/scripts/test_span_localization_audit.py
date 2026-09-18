from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def module():
    spec = importlib.util.spec_from_file_location("span_localization_audit", ROOT / "scripts" / "build_span_localization_audit.py")
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_find_mentions_preserves_original_offsets() -> None:
    audit = module()
    text = "Ada founded  Acme. ADA founded Acme."
    found = audit.find_mentions(text, "Ada founded Acme")
    assert [(item["start"], item["end"], item["match_status"]) for item in found] == [(0, 17, "normalized_whitespace"), (19, 35, "casefold")]
    assert text[found[0]["start"]:found[0]["end"]] == "Ada founded  Acme"


def test_compress_spans_preserves_units_and_highest_risk() -> None:
    audit = module()
    spans = audit.compress_spans([
        {"start": 0, "end": 3, "text": "Ada", "method": "grapheval", "risk": 0.2, "match_count": 1, "unit_id": "t1", "term_role": "subject", "unit_kind": "triple", "relation": "founded", "flagged": False},
        {"start": 0, "end": 3, "text": "Ada", "method": "grapheval", "risk": 0.9, "match_count": 1, "unit_id": "t2", "term_role": "subject", "unit_kind": "triple", "relation": "led", "flagged": True},
    ])
    assert len(spans) == 1
    assert spans[0]["risk"] == 0.9
    assert spans[0]["flagged"] is True
    assert spans[0]["unit_ids"] == ["t1", "t2"]


def test_interval_metrics_are_not_gold_inputs_to_predictions() -> None:
    audit = module()
    prediction = {"raw_score": 0.7}
    features = audit.response_features(prediction, [{"start": 0, "end": 4, "risk": 0.8}], 1, 10)
    assert features["raw_score"] == 0.7
    assert features["span_coverage"] == 0.4
    assert audit.intersection_length([(0, 4)], [(2, 6)]) == 2
    assert audit.iou([(0, 4)], [(2, 6)]) == 1 / 3


def test_mapping_rate_counts_unique_units_not_text_occurrences() -> None:
    audit = module()
    prediction = {"raw_score": 0.7}
    spans = [
        {"start": 0, "end": 3, "risk": 0.8, "unit_ids": ["t1"]},
        {"start": 10, "end": 13, "risk": 0.8, "unit_ids": ["t1"]},
        {"start": 20, "end": 23, "risk": 0.6, "unit_ids": ["t2"]},
    ]
    features = audit.response_features(prediction, spans, total_units=2, text_length=30)
    assert features["mapped_text_mentions"] == 3
    assert features["mapped_unit_ids"] == 2
    assert features["mapping_rate"] == 1.0
