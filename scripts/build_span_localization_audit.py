#!/usr/bin/env python3
"""Post-seal span localization, response-risk metrics, and local HTML audit.

The script only reads a sealed prediction archive and the separately supplied official
RAGTruth response file.  Gold labels are joined after detector execution; no model,
gateway, KGGen, or detector is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts import RunArchive, sha256_file

METHODS = ("hallugraph", "grapheval")
LABELS = {"hallugraph": "HalluGraph", "grapheval": "GraphEval"}
VERSION = "span-localization-audit-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def portable_path(path: Path) -> str:
    """Avoid leaking a developer-specific absolute path into published artifacts."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def div(a: float, b: float) -> float | None:
    return None if not b else a / b


def interval_union(spans: Iterable[dict[str, Any]]) -> list[tuple[int, int]]:
    intervals = sorted((int(s["start"]), int(s["end"])) for s in spans if int(s["end"]) > int(s["start"]))
    result: list[tuple[int, int]] = []
    for start, end in intervals:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_length(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    a, b = list(left), list(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def iou(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> float | None:
    a, b = list(left), list(right)
    shared = intersection_length(a, b)
    total = interval_length(a) + interval_length(b) - shared
    return div(shared, total)


def find_mentions(text: str, term: str) -> list[dict[str, Any]]:
    """Find literal and whitespace-normalized text mentions without inventing offsets."""
    term = unicodedata.normalize("NFKC", str(term or "")).strip()
    if not term:
        return []
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
        found[(match.start(), match.end())] = {"start": match.start(), "end": match.end(), "match_status": "exact" if match.group() == term else "casefold"}
    pieces = [re.escape(piece) for piece in re.split(r"\s+", term) if piece]
    if len(pieces) > 1:
        pattern = r"\s+".join(pieces)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.setdefault((match.start(), match.end()), {"start": match.start(), "end": match.end(), "match_status": "normalized_whitespace"})
    return [found[key] for key in sorted(found)]


def graph_for_instance(graphs: dict[str, dict[str, Any]], instance: dict[str, Any], role: str) -> dict[str, Any] | None:
    key = instance.get(f"{role}_hash")
    return graphs.get(str(key)) if key else None


def graph_entities(graph: dict[str, Any] | None) -> list[str]:
    return [str(value) for value in (graph or {}).get("entities") or [] if str(value).strip()]


def unit_mentions(method: str, prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Return score-bearing candidate mentions from detector artifacts only."""
    out: list[dict[str, Any]] = []
    components = prediction.get("components") or {}
    if method == "grapheval":
        for triple in components.get("triples") or []:
            risk = triple.get("p_unsupported")
            if risk is None:
                continue
            unit_id = str(triple.get("triple_id", "unknown"))
            for role, term in (("subject", triple.get("raw_subject")), ("object", triple.get("raw_object"))):
                if str(term or "").strip():
                    out.append({"unit_id": unit_id, "unit_kind": "triple", "term_role": role, "term": str(term), "risk": float(risk), "relation": triple.get("raw_relation"), "flagged": bool(triple.get("flagged_at_paper_threshold"))})
        return out
    raw_score = float(prediction.get("raw_score") or 0.0)
    for entity in components.get("ungrounded_entities") or []:
        out.append({"unit_id": f"entity:{entity}", "unit_kind": "ungrounded_entity", "term_role": "entity", "term": str(entity), "risk": 1.0, "flagged": True})
    for relation in components.get("unsupported_relations") or []:
        if not isinstance(relation, (list, tuple)) or len(relation) != 3:
            continue
        subject, predicate, obj = map(str, relation)
        unit_id = f"relation:{subject}|{predicate}|{obj}"
        for role, term in (("subject", subject), ("object", obj)):
            out.append({"unit_id": unit_id, "unit_kind": "unsupported_relation", "term_role": role, "term": term, "risk": 1.0, "response_risk": raw_score, "relation": predicate, "flagged": True})
    return out


def char_risks(length: int, spans: list[dict[str, Any]]) -> list[float]:
    values = [0.0] * length
    for span in spans:
        for index in range(max(0, int(span["start"])), min(length, int(span["end"]))):
            values[index] = max(values[index], float(span["risk"]))
    return values


def compress_spans(raw_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one display interval per text position while retaining all unit provenance."""
    grouped: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for span in raw_spans:
        key = (int(span["start"]), int(span["end"]), str(span["text"]), str(span["method"]))
        current = grouped.get(key)
        if current is None:
            current = {name: value for name, value in span.items() if name not in {"unit_id", "term_role", "relation", "unit_kind"}}
            current["unit_ids"] = []
            current["term_roles"] = []
            current["unit_kinds"] = []
            current["relations"] = []
            grouped[key] = current
        current["risk"] = max(float(current["risk"]), float(span["risk"]))
        current["flagged"] = bool(current.get("flagged")) or bool(span.get("flagged"))
        for field, value in (("unit_ids", span["unit_id"]), ("term_roles", span["term_role"]), ("unit_kinds", span["unit_kind"]), ("relations", span.get("relation"))):
            if value is not None and value not in current[field]:
                current[field].append(value)
        current["match_count"] = max(int(current["match_count"]), int(span["match_count"]))
    return sorted(grouped.values(), key=lambda item: (item["start"], item["end"], -float(item["risk"])))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = int(math.floor(pos)), int(math.ceil(pos))
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def response_features(
    prediction: dict[str, Any],
    spans: list[dict[str, Any]],
    total_units: int,
    text_length: int,
    mapped_units: int | None = None,
) -> dict[str, Any]:
    """Build no-gold response features.

    ``mapping_rate`` is the share of unique graph units with at least one text
    mention.  Earlier versions divided the number of text occurrences by the
    number of units, so repeated mentions could incorrectly produce values > 1.
    """
    risks = sorted((float(s["risk"]) for s in spans), reverse=True)
    chars = char_risks(text_length, spans)
    union = interval_union(spans)
    if mapped_units is None:
        mapped_units = len({unit_id for span in spans for unit_id in span.get("unit_ids", [])})
    return {
        "raw_score": prediction.get("raw_score"),
        "span_max": max(risks) if risks else 0.0,
        "span_top3_mean": sum(risks[:3]) / min(3, len(risks)) if risks else 0.0,
        "span_top5_mean": sum(risks[:5]) / min(5, len(risks)) if risks else 0.0,
        "span_char_mean": sum(chars) / len(chars) if chars else 0.0,
        "span_char_p90": percentile(chars, 0.9) or 0.0,
        "span_risky_char_fraction": sum(value >= 0.5 for value in chars) / len(chars) if chars else 0.0,
        "span_coverage": interval_length(union) / text_length if text_length else 0.0,
        "mapped_text_mentions": len(spans),
        "mapped_unit_ids": mapped_units,
        "total_unit_ids": total_units,
        "mapping_rate": div(mapped_units, total_units),
        "predicted_chars": interval_length(union),
    }


def select_demo_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a small, deterministic set of qualitatively different cases."""
    if limit <= 0 or limit >= len(records):
        return records

    selected: dict[str, dict[str, Any]] = {}

    def add(reason: str, candidates: Iterable[dict[str, Any]], count: int) -> None:
        for record in candidates:
            response_id = str(record["response_id"])
            if response_id in selected:
                continue
            copy = dict(record)
            copy["demo_reason"] = reason
            selected[response_id] = copy
            if len(selected) >= limit or sum(item.get("demo_reason") == reason for item in selected.values()) >= count:
                break

    positives = [record for record in records if record["gold"]]
    negatives = [record for record in records if not record["gold"]]
    metric = lambda record, method, name, default=0.0: record["metrics"][method].get(name) if record["metrics"][method].get(name) is not None else default
    add("GraphEval локализует точнее HalluGraph", sorted(positives, key=lambda r: metric(r, "grapheval", "iou") - metric(r, "hallugraph", "iou"), reverse=True), 3)
    add("HalluGraph локализует точнее GraphEval", sorted(positives, key=lambda r: metric(r, "hallugraph", "iou") - metric(r, "grapheval", "iou"), reverse=True), 3)
    add("Оба метода хорошо попали в gold", sorted(positives, key=lambda r: min(metric(r, "hallugraph", "iou"), metric(r, "grapheval", "iou")), reverse=True), 3)
    add("Gold-спан пропущен обоими методами", [r for r in positives if not r["metrics"]["hallugraph"]["any_hit"] and not r["metrics"]["grapheval"]["any_hit"]], 3)
    add("Высокий риск на корректном ответе", sorted(negatives, key=lambda r: max(float(r["methods"]["hallugraph"].get("raw_score") or 0), float(r["methods"]["grapheval"].get("raw_score") or 0)), reverse=True), 3)
    add("Низкий риск при наличии gold-галлюцинации", sorted(positives, key=lambda r: max(float(r["methods"]["hallugraph"].get("raw_score") or 0), float(r["methods"]["grapheval"].get("raw_score") or 0))), 3)
    add("Дополнительный разнообразный пример", records, limit)
    return list(selected.values())[:limit]


def auroc(rows: list[dict[str, Any]]) -> float | None:
    pairs = [(float(row["score"]), int(row["gold"])) for row in rows if row.get("score") is not None]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(enumerate(pairs), key=lambda value: value[1][0])
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[index][1][0]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(rows: list[dict[str, Any]]) -> float | None:
    pairs = sorted(((float(row["score"]), int(row["gold"])) for row in rows if row.get("score") is not None), reverse=True)
    positives = sum(label for _, label in pairs)
    if not positives:
        return None
    tp = fp = 0
    previous_recall = area = 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row["gold"]) for row in rows]
    decisions = [float(row["score"]) > threshold for row in rows]
    tp = sum(a and b == 1 for a, b in zip(decisions, labels))
    fp = sum(a and b == 0 for a, b in zip(decisions, labels))
    tn = sum(not a and b == 0 for a, b in zip(decisions, labels))
    fn = sum(not a and b == 1 for a, b in zip(decisions, labels))
    precision, recall, specificity = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    brier = sum((float(row["score"]) - int(row["gold"])) ** 2 for row in rows) / len(rows) if rows else None
    bins = [[] for _ in range(10)]
    for row in rows:
        bins[min(9, int(float(row["score"]) * 10))].append(row)
    ece = sum(abs(sum(float(r["score"]) for r in group) / len(group) - sum(int(r["gold"]) for r in group) / len(group)) * len(group) / len(rows) for group in bins if group) if rows else None
    return {"n": len(rows), "threshold": threshold, "AUROC": auroc(rows), "AUPRC": auprc(rows), "precision": precision, "recall": recall, "specificity": specificity, "F1": div(2 * tp, 2 * tp + fp + fn), "balanced_accuracy": div((recall or 0) + (specificity or 0), 2) if recall is not None and specificity is not None else None, "brier": brier, "ece_10": ece, "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def choose_threshold(rows: list[dict[str, Any]]) -> float:
    candidates = sorted({0.0, 1.0, *(float(row["score"]) for row in rows)})
    return max(candidates, key=lambda t: ((threshold_metrics(rows, t)["F1"] or -1), (threshold_metrics(rows, t)["recall"] or -1), -t))


def aggregate_span_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    positive = [row for row in rows if row["gold"]]
    pred_chars = hit_chars = gold_chars = hit_gold_spans = total_gold_spans = 0
    excess_chars = neg_pred_chars = neg_answer_chars = 0
    mapped = units = ambiguous = 0
    ious: list[float] = []
    best_coverages: list[float] = []
    response_hits = 0
    for row in rows:
        record = row["localization"][method]
        p, g = interval_union(record["spans"]), interval_union(row["gold_spans"])
        p_len, g_len, shared = interval_length(p), interval_length(g), intersection_length(p, g)
        pred_chars += p_len; gold_chars += g_len; hit_chars += shared; excess_chars += p_len - shared
        mapped += record["mapped_unit_ids"]; units += record["total_unit_ids"]; ambiguous += record["ambiguous_unit_ids"]
        if not row["gold"]:
            neg_pred_chars += p_len; neg_answer_chars += len(row["response"])
            continue
        if shared:
            response_hits += 1
        value = iou(p, g)
        if value is not None:
            ious.append(value)
        for gold in row["gold_spans"]:
            total_gold_spans += 1
            gold_interval = [(gold["start"], gold["end"])]
            overlaps = [intersection_length([(span["start"], span["end"])], gold_interval) for span in record["spans"]]
            best = max(overlaps, default=0)
            if best:
                hit_gold_spans += 1
            best_coverages.append(best / (gold["end"] - gold["start"]) if gold["end"] > gold["start"] else 0.0)
    precision, recall = div(hit_chars, pred_chars), div(hit_chars, gold_chars)
    return {"n_responses": len(rows), "n_positive_responses": len(positive), "any_hit_rate": div(response_hits, len(positive)), "character_precision": precision, "character_recall": recall, "character_f1": div(2 * (precision or 0) * (recall or 0), (precision or 0) + (recall or 0)) if precision is not None and recall is not None and precision + recall else None, "micro_iou": div(hit_chars, pred_chars + gold_chars - hit_chars), "mean_response_iou_positive": sum(ious) / len(ious) if ious else None, "gold_span_hit_rate": div(hit_gold_spans, total_gold_spans), "mean_gold_span_coverage": sum(best_coverages) / len(best_coverages) if best_coverages else None, "excess_share": div(excess_chars, pred_chars), "excess_to_gold_ratio": div(excess_chars, gold_chars), "negative_response_highlight_rate": div(neg_pred_chars, neg_answer_chars), "mapping_rate": div(mapped, units), "ambiguous_span_rate": div(ambiguous, mapped), "predicted_chars": pred_chars, "gold_chars": gold_chars}


def svg_graph(graph: dict[str, Any] | None, graph_id: str) -> str:
    if not graph:
        return '<p class="muted">Граф отсутствует.</p>'
    nodes = graph_entities(graph)[:55]
    rels = [rel for rel in graph.get("relations") or [] if len(rel) == 3 and rel[0] in nodes and rel[2] in nodes][:100]
    if not nodes:
        return '<p class="muted">В графе нет узлов.</p>'
    positions = {node: (420 + 330 * math.cos(2 * math.pi * i / len(nodes)), 220 + 165 * math.sin(2 * math.pi * i / len(nodes))) for i, node in enumerate(nodes)}
    def esc(value: Any) -> str: return html.escape(str(value))
    edges = ''.join(f'<line x1="{positions[a][0]:.1f}" y1="{positions[a][1]:.1f}" x2="{positions[b][0]:.1f}" y2="{positions[b][1]:.1f}" class="edge"/><text x="{(positions[a][0]+positions[b][0])/2:.1f}" y="{(positions[a][1]+positions[b][1])/2:.1f}" class="edge-label">{esc(r)[:28]}</text>' for a, r, b in rels)
    points = ''.join(f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="17" class="node"/><text x="{x:.1f}" y="{y+3:.1f}" class="node-label">{esc(node)[:18]}</text></g>' for node, (x, y) in positions.items())
    return f'<svg id="{graph_id}" viewBox="0 0 840 440" class="graph" role="img" aria-label="Граф сущностей">{edges}{points}</svg><p class="muted">{len(nodes)} сущностей, {len(rels)} отношений.</p>'


def html_page(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Просмотрщик спанов HalluGraph и GraphEval</title>
<style>
:root{--bg:#f4f6fb;--card:#fff;--ink:#142033;--muted:#667085;--line:#d9e1ee;--blue:#245bdb;--red:#c4322b;--orange:#b85d00;--green:#147747;--purple:#7047b8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,sans-serif}main{max-width:1800px;margin:auto;padding:20px}h1{margin:0;font-size:26px}h2{font-size:18px;margin:0 0 10px}h3{font-size:15px;margin:14px 0 7px}.muted{color:var(--muted)}.notice{margin:12px 0;padding:10px 12px;border-left:4px solid var(--blue);background:#eef4ff;border-radius:6px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;box-shadow:0 1px 2px #1018280b}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0}.number{font-size:24px;font-weight:750}.layout{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px;margin-top:14px}.rail{position:sticky;top:10px;height:calc(100vh - 24px);overflow:auto}.filters{display:grid;gap:8px}input,select,button{font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink)}button{cursor:pointer}button.active{background:var(--blue);color:white;border-color:var(--blue)}.items{margin-top:10px;display:grid;gap:5px}.item{text-align:left;width:100%;padding:9px}.item small{display:block;color:var(--muted);margin-top:2px}.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}.tab-panel{display:none}.tab-panel.active{display:block}.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.metric-table{width:100%;border-collapse:collapse;font-size:12px}.metric-table th,.metric-table td{padding:7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.metric-table th{color:var(--muted);background:#fafcff;position:sticky;top:0}.table-wrap{overflow:auto;max-height:420px;border:1px solid var(--line);border-radius:7px}.text{white-space:pre-wrap;background:#fafcff;border:1px solid var(--line);padding:12px;border-radius:7px;max-height:360px;overflow:auto;font-size:14px}.span-mark{padding:1px 0;border-radius:2px}.is-gold{background:#ffe4e1;border-bottom:3px solid var(--red)}.is-h{background:#cfe0ff}.is-ge{background:#ffe0ad}.is-both{background:#d9c5ff}.is-gold.is-h,.is-gold.is-ge,.is-gold.is-both{border-bottom:3px solid var(--red)}.legend{display:flex;gap:8px;flex-wrap:wrap;font-size:12px}.swatch{padding:3px 7px;border-radius:4px}.graph-shell{overflow:auto;border:1px solid var(--line);border-radius:7px;background:#fbfdff}.graph{display:block;width:840px;max-width:none;height:440px}.node{fill:#e9f0ff;stroke:#245bdb}.node-label{font-size:9px;text-anchor:middle;fill:#142033}.edge{stroke:#9aa8bb;stroke-width:1}.edge-label{font-size:8px;text-anchor:middle;fill:#526176}details{margin-top:10px}details pre{white-space:pre-wrap;max-height:320px;overflow:auto;font-size:11px;background:#f7f9fc;padding:10px;border-radius:6px}.tag{font-size:11px;font-weight:650;padding:2px 6px;border-radius:5px}.yes{background:#fbd8d5;color:#8e1713}.no{background:#dff5e7;color:#075f31}.reason{display:inline-block;background:#ede9fe;color:#5425a8;padding:3px 7px;border-radius:5px;margin:4px 0}.metric-help{margin-top:10px}.metric-help summary,details summary{cursor:pointer;font-weight:650}@media(max-width:950px){main{padding:10px}.layout{grid-template-columns:1fr}.rail{position:static;height:auto;max-height:48vh}.grid2,.grid3{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Просмотрщик спанов HalluGraph и GraphEval</h1><p class="muted" id="subtitle"></p><div class="notice"><b>Как читать:</b> красная нижняя линия — эталонная галлюцинация RAGTruth; синий фон — HalluGraph; оранжевый — GraphEval; фиолетовый — оба метода. Эталон используется только после работы детекторов, для оценки.</div><section class="cards" id="summary"></section>
<details><summary>Общие метрики на отложенной тестовой части</summary><section class="card"><h2>Наличие галлюцинации во всём ответе</h2><div id="response-metrics"></div></section><section class="card"><h2>Точность локализации</h2><div id="span-metrics"></div><details class="metric-help"><summary>Что означают показатели</summary><p><b>Есть попадание</b> — доля галлюцинаторных ответов, где найден хотя бы один общий символ. <b>Точность символов</b> — какая часть подсветки лежит внутри эталона. <b>Полнота символов</b> — какая часть эталонной галлюцинации подсвечена. <b>IoU</b> — пересечение, делённое на объединение. <b>Лишняя подсветка</b> — доля предсказанной подсветки вне эталона.</p></details></section></details>
<div class="layout"><section class="card" id="detail"></section><aside class="card rail"><h2>Показательные примеры</h2><div class="filters"><input id="search" placeholder="ID или текст"><select id="filter"><option value="all">все примеры</option><option value="positive">есть gold-галлюцинация</option><option value="hit">GraphEval пересёк gold</option><option value="miss">GraphEval пропустил gold</option><option value="negative">корректный ответ</option></select><select id="sort"><option value="id">по ID</option><option value="ge">по риску GraphEval</option><option value="h">по риску HalluGraph</option><option value="iou">по GraphEval IoU (использует gold)</option></select></div><p class="muted" id="count"></p><div class="items" id="items"></div></aside></div>
</main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=x=>document.getElementById(x);let selected=null,currentTab='spans';
const f=x=>x==null?'—':Number(x).toFixed(3),pct=x=>x==null?'—':`${(100*Number(x)).toFixed(1)}%`,e=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function table(rows,cols){return `<div class="table-wrap"><table class="metric-table"><thead><tr>${cols.map(c=>`<th>${c[1]}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${e(c[2]?c[2](r):r[c[0]])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function visible(r){let q=$('search').value.toLowerCase(),v=$('filter').value;if(q&&!(`${r.response_id} ${r.query} ${r.response} ${r.demo_reason||''}`).toLowerCase().includes(q))return false;if(v==='positive'&&!r.gold)return false;if(v==='negative'&&r.gold)return false;if(v==='hit'&&!r.metrics.grapheval.any_hit)return false;if(v==='miss'&&(!r.gold||r.metrics.grapheval.any_hit))return false;return true}
function updateHash(){if(!selected)return;history.replaceState(null,'',`#id=${encodeURIComponent(selected.response_id)}&tab=${encodeURIComponent(currentTab)}`)}
function renderItems(){let rows=D.records.filter(visible),s=$('sort').value;rows.sort((a,b)=>s==='id'?String(a.response_id).localeCompare(String(b.response_id),undefined,{numeric:true}):s==='ge'?(b.methods.grapheval.raw_score??-1)-(a.methods.grapheval.raw_score??-1):s==='h'?(b.methods.hallugraph.raw_score??-1)-(a.methods.hallugraph.raw_score??-1):(b.metrics.grapheval.iou??-1)-(a.metrics.grapheval.iou??-1));$('count').textContent=`${rows.length} из ${D.records.length}`;$('items').innerHTML=rows.map(r=>`<button class="item ${selected?.response_id===r.response_id?'active':''}" data-id="${e(r.response_id)}"><b>${e(r.response_id)}</b> ${r.gold?'<span class="tag yes">gold+</span>':'<span class="tag no">gold−</span>'}<small>H ${f(r.methods.hallugraph.raw_score)} · G ${f(r.methods.grapheval.raw_score)} · IoU ${f(r.metrics.grapheval.iou)}</small><small>${e(r.demo_reason||'')}</small></button>`).join('');document.querySelectorAll('.item').forEach(x=>x.onclick=()=>{selected=D.records.find(r=>String(r.response_id)===x.dataset.id);currentTab='spans';renderItems();renderDetail();updateHash()})}
function highlight(text,spans,gold){let cuts=new Set([0,text.length]);[...spans,...gold].forEach(s=>{cuts.add(s.start);cuts.add(s.end)});let a=[...cuts].filter(x=>x>=0&&x<=text.length).sort((x,y)=>x-y),out='';for(let i=0;i<a.length-1;i++){let x=a[i],y=a[i+1],h=spans.some(s=>s.start<y&&s.end>x&&s.method==='hallugraph'),g=spans.some(s=>s.start<y&&s.end>x&&s.method==='grapheval'),z=gold.some(s=>s.start<y&&s.end>x),classes=['span-mark'];if(z)classes.push('is-gold');if(h&&g)classes.push('is-both');else if(h)classes.push('is-h');else if(g)classes.push('is-ge');out+=classes.length>1?`<mark class="${classes.join(' ')}">${e(text.slice(x,y))}</mark>`:e(text.slice(x,y))}return out}
function overlap(span,gold){return gold.reduce((n,g)=>n+Math.max(0,Math.min(span.end,g.end)-Math.max(span.start,g.start)),0)}
function spanRows(r){return ['hallugraph','grapheval'].flatMap(k=>r.localization[k].spans.map(s=>({method:D.labels[k],text:s.text,start:s.start,end:s.end,risk:f(s.risk),overlap:overlap(s,r.gold_spans),units:(s.unit_ids||[]).join(', ')})))}
function graphMarkup(g){if(!g)return '<p class="muted">Граф отсутствует.</p>';let nodes=(g.entities||[]).slice(0,55),rels=(g.relations||[]).filter(x=>nodes.includes(x[0])&&nodes.includes(x[2])).slice(0,100);if(!nodes.length)return '<p class="muted">В графе нет узлов.</p>';let pos={};nodes.forEach((n,i)=>{let a=2*Math.PI*i/nodes.length;pos[n]=[420+330*Math.cos(a),220+165*Math.sin(a)]});let edges=rels.map(([a,r,b])=>`<line x1="${pos[a][0]}" y1="${pos[a][1]}" x2="${pos[b][0]}" y2="${pos[b][1]}" class="edge"/><text x="${(pos[a][0]+pos[b][0])/2}" y="${(pos[a][1]+pos[b][1])/2}" class="edge-label">${e(r).slice(0,28)}</text>`).join('');let points=nodes.map(n=>`<g><circle cx="${pos[n][0]}" cy="${pos[n][1]}" r="17" class="node"/><text x="${pos[n][0]}" y="${pos[n][1]+3}" class="node-label">${e(n).slice(0,18)}</text></g>`).join('');return `<div class="graph-shell"><svg viewBox="0 0 840 440" class="graph">${edges+points}</svg></div><p class="muted">${nodes.length} сущностей, ${rels.length} отношений. Граф можно прокручивать по горизонтали.</p>`}
function methodTab(r,k){let m=r.methods[k],loc=r.localization[k],g=r.graphs;let units=k==='grapheval'?m.components.triples||[]:m.components.unsupported_relations||[];return `<div class="grid2"><div><h3>Результат</h3><p>Риск ответа: <b>${f(m.raw_score)}</b><br>Статус: ${e(m.status)}<br>Локализовано графовых единиц: <b>${loc.mapped_unit_ids} из ${loc.total_unit_ids}</b> (${pct(r.features[k].mapping_rate)})<br>Найдено текстовых упоминаний: ${r.features[k].mapped_text_mentions}</p><details><summary>Компоненты и подозрительные единицы</summary><pre>${e(JSON.stringify({components:m.components,suspicious_units:units},null,2))}</pre></details></div><div><h3>Граф ответа</h3>${graphMarkup(g.response)}<h3>Граф контекста</h3>${graphMarkup(g.context)}</div></div>`}
function activateTab(name){currentTab=name;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===name));document.querySelectorAll('.tab-panel').forEach(x=>x.classList.toggle('active',x.id===name));updateHash()}
function renderDetail(){let r=selected;if(!r)return;let rows=spanRows(r);$('detail').innerHTML=`<h2>Ответ ${e(r.response_id)}</h2><div class="reason">Почему выбран: ${e(r.demo_reason||'пример из полного набора')}</div><p class="muted">Источник ${e(r.source_id)} · выборка ${e(r.split)} · ${e(r.task||'тип задачи не указан')} · gold: ${r.gold?'галлюцинация':'корректный ответ'}</p><div class="tabs"><button class="tab" data-tab="metrics">1. Метрики</button><button class="tab" data-tab="spans">2. Спаны</button><button class="tab" data-tab="hallugraph">3. HalluGraph</button><button class="tab" data-tab="grapheval">4. GraphEval</button></div><div class="tab-panel" id="metrics"><div class="grid3">${['hallugraph','grapheval'].map(k=>`<div class="card"><h3>${D.labels[k]}</h3><p>Риск: <b>${f(r.methods[k].raw_score)}</b><br>IoU: ${f(r.metrics[k].iou)}<br>Точность символов: ${f(r.metrics[k].precision)}<br>Полнота символов: ${f(r.metrics[k].recall)}<br>Лишняя подсветка: ${f(r.metrics[k].excess_share)}</p></div>`).join('')}<div class="card"><h3>Объединение</h3><p>Максимальный риск: <b>${f(r.features.combined.max_method_score)}</b><br>Пересечение подсветок методов: ${f(r.features.combined.span_overlap_iou)}</p></div></div><details><summary>Технические признаки и gold-разметка</summary><pre>${e(JSON.stringify({gold_spans:r.gold_spans,features:r.features},null,2))}</pre></details></div><div class="tab-panel" id="spans"><div class="legend"><span class="swatch is-gold">gold RAGTruth</span><span class="swatch is-h">HalluGraph</span><span class="swatch is-ge">GraphEval</span><span class="swatch is-both">оба метода</span></div><p class="muted">Координаты и метрики считаются только в ответе. Запрос и контекст приведены ниже как справочная информация.</p><h3>Ответ</h3><div class="text">${highlight(r.response,[...r.localization.hallugraph.spans,...r.localization.grapheval.spans],r.gold_spans)}</div><h3>Найденные спаны</h3>${table(rows,[['method','Метод'],['text','Текст'],['start','Начало'],['end','Конец'],['risk','Риск'],['overlap','Общих символов с gold'],['units','Графовая единица']])}<h3>Запрос</h3><div class="text">${e(r.query)}</div><h3>Контекст</h3><div class="text">${e(r.context)}</div><details><summary>Полные технические данные сопоставления</summary><pre>${e(JSON.stringify(r.localization,null,2))}</pre></details></div><div class="tab-panel" id="hallugraph">${methodTab(r,'hallugraph')}</div><div class="tab-panel" id="grapheval">${methodTab(r,'grapheval')}</div>`;document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>activateTab(b.dataset.tab));activateTab(currentTab)}
function init(){$('subtitle').textContent=`Полный расчёт: ${D.summary.n_records} ответов. В этой сборке: ${D.records.length} показательных примеров. Анализ выполнен после запечатывания предсказаний.`;$('summary').innerHTML=`<div class="card"><div class="number">${D.summary.n_records}</div><div>ответов в полном расчёте</div></div><div class="card"><div class="number">${D.records.length}</div><div>примеров в демонстрации</div></div><div class="card"><div class="number">${D.summary.n_test}</div><div>ответов в тестовой части</div></div><div class="card"><div class="number">${D.summary.n_empty_graph}</div><div>пустых графов</div></div>`;const response=D.response_metrics.test.map(r=>({name:r.name,AUROC:f(r.metrics.AUROC),AUPRC:f(r.metrics.AUPRC),F1:f(r.metrics.F1)}));$('response-metrics').innerHTML=table(response,[['name','Методика'],['AUROC','ROC-AUC'],['AUPRC','PR-AUC'],['F1','F1']]);const spans=D.span_metrics.test.map(r=>({name:D.labels[r.method],hit:pct(r.metrics.any_hit_rate),precision:pct(r.metrics.character_precision),recall:pct(r.metrics.character_recall),iou:f(r.metrics.micro_iou),excess:pct(r.metrics.excess_share),map:pct(r.metrics.mapping_rate)}));$('span-metrics').innerHTML=table(spans,[['name','Метод'],['hit','Есть попадание'],['precision','Точность символов'],['recall','Полнота символов'],['iou','IoU'],['excess','Лишняя подсветка'],['map','Локализовано единиц']]);['search','filter','sort'].forEach(id=>$(id).oninput=renderItems);let params=new URLSearchParams(location.hash.slice(1)),wanted=params.get('id');selected=D.records.find(r=>String(r.response_id)===wanted)||D.records[0];currentTab=['metrics','spans','hallugraph','grapheval'].includes(params.get('tab'))?params.get('tab'):'spans';renderItems();renderDetail();updateHash()}
init();</script></body></html>'''
    return template.replace("__DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--viewer-record-limit", type=int, default=0, help="Embed a representative subset in HTML; 0 means all records.")
    parser.add_argument("--viewer-only", action="store_true", help="Write only manifest, dataset metrics and HTML viewer.")
    parser.add_argument("--viewer-file-name", default="span-localization-audit.html", help="HTML file name inside output directory.")
    args = parser.parse_args()
    archive_dir, responses_path, output = args.archive_dir.resolve(), args.responses.resolve(), args.output_dir.resolve()
    if archive_dir in output.parents:
        parser.error("output directory must be outside the sealed archive")
    validation = RunArchive(archive_dir.parent, archive_dir.name).validate()
    if not validation["valid"]:
        raise ValueError(f"invalid sealed archive: {validation['errors']}")
    predictions = read_jsonl(archive_dir / "predictions" / "raw_predictions.jsonl")
    if any(row.get("gold_access_state") != "hidden" for row in predictions):
        raise ValueError("prediction archive contains visible gold")
    instances = {str(row["response_id"]): row for row in read_jsonl(archive_dir / "instances.no_gold.jsonl")}
    gold_by_id = {str(row["id"]): row for row in read_jsonl(responses_path)}
    missing = set(instances) - set(gold_by_id)
    if missing:
        raise ValueError(f"official response file lacks {len(missing)} archive responses")
    graph_index = {str(row["input_sha256"]): row for row in read_jsonl(archive_dir / "shared_graphs" / "graph_index.jsonl")}
    predictions_by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        predictions_by_id[str(row["response_id"])][str(row["method"])] = row
    records: list[dict[str, Any]] = []
    derived_spans: list[dict[str, Any]] = []
    for response_id, instance in sorted(instances.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else pair[0]):
        answer = str(instance.get("response_raw") or "")
        source_gold = gold_by_id[response_id]
        gold_spans = [{"start": int(label["start"]), "end": int(label["end"]), "text": str(label.get("text") or ""), "label_type": label.get("label_type"), "due_to_null": label.get("due_to_null"), "implicit_true": label.get("implicit_true")} for label in source_gold.get("labels") or [] if int(label.get("end", 0)) >= int(label.get("start", 0))]
        graphs = {role: graph_for_instance(graph_index, instance, role) for role in ("context", "query", "response")}
        localized: dict[str, dict[str, Any]] = {}
        method_rows: dict[str, Any] = {}
        features: dict[str, Any] = {}
        metrics_by_method: dict[str, Any] = {}
        for method in METHODS:
            prediction = predictions_by_id[response_id].get(method)
            if prediction is None:
                raise ValueError(f"missing {method} prediction for {response_id}")
            candidates = unit_mentions(method, prediction) if prediction.get("status") == "ok" else []
            raw_spans: list[dict[str, Any]] = []
            for candidate in candidates:
                occurrences = find_mentions(answer, candidate["term"])
                for occurrence in occurrences:
                    span = {**candidate, **occurrence, "response_id": response_id, "method": method, "text": answer[occurrence["start"]:occurrence["end"]], "match_count": len(occurrences)}
                    raw_spans.append(span)
            all_scored_spans = compress_spans(raw_spans)
            spans = [span for span in all_scored_spans if span.get("flagged")]
            total_unit_ids = len({candidate["unit_id"] for candidate in candidates})
            mapped_unit_ids = len({unit_id for span in all_scored_spans for unit_id in span["unit_ids"]})
            ambiguous_unit_ids = len({unit_id for span in all_scored_spans if span["match_count"] > 1 for unit_id in span["unit_ids"]})
            derived_spans.extend(all_scored_spans)
            localized[method] = {"spans": spans, "all_scored_spans": all_scored_spans, "total_unit_ids": total_unit_ids, "mapped_unit_ids": mapped_unit_ids, "ambiguous_unit_ids": ambiguous_unit_ids}
            features[method] = response_features(prediction, all_scored_spans, total_unit_ids, len(answer), mapped_units=mapped_unit_ids)
            p, g = interval_union(spans), interval_union(gold_spans)
            shared = intersection_length(p, g)
            precision, recall = div(shared, interval_length(p)), div(shared, interval_length(g))
            metrics_by_method[method] = {"any_hit": bool(shared), "iou": iou(p, g), "precision": precision, "recall": recall, "excess_share": div(interval_length(p) - shared, interval_length(p))}
            method_rows[method] = {"raw_score": prediction.get("raw_score"), "status": prediction.get("status"), "components": prediction.get("components") or {}, "flagged_unit_ids": prediction.get("flagged_unit_ids") or []}
        h, ge = method_rows["hallugraph"].get("raw_score"), method_rows["grapheval"].get("raw_score")
        hp, gp = interval_union(localized["hallugraph"]["spans"]), interval_union(localized["grapheval"]["spans"])
        agreement_iou = iou(hp, gp) or 0.0
        if h is not None and ge is not None:
            features["combined"] = {"max_method_score": max(float(h), float(ge)), "mean_method_score": (float(h) + float(ge)) / 2, "min_method_score": min(float(h), float(ge)), "agreement_span_score": math.sqrt(max(0.0, float(h) * float(ge))) * agreement_iou, "span_overlap_iou": agreement_iou}
        else:
            features["combined"] = {"max_method_score": None, "mean_method_score": None, "min_method_score": None, "agreement_span_score": None, "span_overlap_iou": agreement_iou}
        graph_spans = [{"start": item["start"], "end": item["end"]} for entity in graph_entities(graphs["response"]) for item in find_mentions(answer, entity)]
        ceiling_shared = intersection_length(interval_union(graph_spans), interval_union(gold_spans))
        record = {"response_id": response_id, "source_id": str(instance.get("source_id")), "split": str(instance.get("split")), "task": (instance.get("metadata") or {}).get("task"), "gold": int(bool(gold_spans)), "gold_spans": gold_spans, "query": str(instance.get("query_raw") or ""), "context": str(instance.get("context_raw") or ""), "response": answer, "methods": method_rows, "localization": localized, "features": features, "metrics": metrics_by_method, "graph_ceiling": {"answer_graph_gold_char_recall": div(ceiling_shared, interval_length(interval_union(gold_spans)))}, "graphs": graphs}
        records.append(record)
    response_metrics = {"train": [], "test": []}
    scalar_defs = [("hallugraph_raw", "HalluGraph raw score", lambda r: r["features"]["hallugraph"]["raw_score"] if r["methods"]["hallugraph"]["status"] == "ok" else None), ("grapheval_raw", "GraphEval raw score", lambda r: r["features"]["grapheval"]["raw_score"] if r["methods"]["grapheval"]["status"] == "ok" else None), ("combined_max", "max(HalluGraph, GraphEval)", lambda r: r["features"]["combined"]["max_method_score"]), ("combined_mean", "mean(HalluGraph, GraphEval)", lambda r: r["features"]["combined"]["mean_method_score"]), ("combined_min", "min(HalluGraph, GraphEval)", lambda r: r["features"]["combined"]["min_method_score"]), ("agreement_span", "agreement × span overlap", lambda r: r["features"]["combined"]["agreement_span_score"]), ("grapheval_span_top3", "GraphEval top-3 span risk", lambda r: r["features"]["grapheval"]["span_top3_mean"] if r["methods"]["grapheval"]["status"] == "ok" else None), ("grapheval_span_density", "GraphEval mean character risk", lambda r: r["features"]["grapheval"]["span_char_mean"] if r["methods"]["grapheval"]["status"] == "ok" else None)]
    for key, name, getter in scalar_defs:
        train = [{"score": getter(r), "gold": r["gold"]} for r in records if r["split"] == "train" and getter(r) is not None]
        test = [{"score": getter(r), "gold": r["gold"]} for r in records if r["split"] == "test" and getter(r) is not None]
        if train:
            threshold = choose_threshold(train)
            response_metrics["train"].append({"key": key, "name": name, "metrics": threshold_metrics(train, threshold)})
            response_metrics["test"].append({"key": key, "name": name, "metrics": threshold_metrics(test, threshold)})
    span_metrics = {split: [{"method": method, "coverage": div(sum(r["methods"][method]["status"] == "ok" for r in records if r["split"] == split), sum(r["split"] == split for r in records)), "metrics": aggregate_span_metrics([r for r in records if r["split"] == split and r["methods"][method]["status"] == "ok"], method)} for method in METHODS] for split in ("train", "test")}
    output.mkdir(parents=True, exist_ok=True)
    viewer_records = select_demo_records(records, args.viewer_record_limit)
    manifest = {"analysis_version": VERSION, "analysis_only": True, "archive_dir": portable_path(archive_dir), "archive_validation": validation, "responses_path": portable_path(responses_path), "responses_sha256": sha256_file(responses_path), "n_records": len(records), "n_viewer_records": len(viewer_records), "gold_join_timing": "post_seal_only", "localization_policy": "literal/casefold/whitespace-normalized entity mentions; relation words are not invented", "known_limit": "HalluGraph unit localization is binary because the archived output has no per-unit continuous risk."}
    compact_records = [{k: v for k, v in record.items() if k != "graphs"} | {"graphs": {role: {"entities": graph_entities(record["graphs"][role]), "relations": (record["graphs"][role] or {}).get("relations") or []} if record["graphs"][role] else None for role in record["graphs"]}} for record in viewer_records]
    payload = {"provenance": {"version": VERSION, "responses_sha256": manifest["responses_sha256"], "analysis_only": True}, "summary": {"n_records": len(records), "n_viewer_records": len(viewer_records), "n_positive": sum(r["gold"] for r in records), "n_test": sum(r["split"] == "test" for r in records), "n_empty_graph": sum(r["methods"]["grapheval"]["status"] != "ok" for r in records)}, "labels": LABELS, "response_metrics": response_metrics, "span_metrics": span_metrics, "records": compact_records}
    (output / "span-localization-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.viewer_only:
        write_jsonl(output / "predicted-spans.jsonl", derived_spans)
        write_jsonl(output / "response-span-metrics.jsonl", [{"response_id": r["response_id"], "source_id": r["source_id"], "split": r["split"], "gold": r["gold"], "metrics": r["metrics"], "features": r["features"], "graph_ceiling": r["graph_ceiling"]} for r in records])
    (output / "dataset-metrics.json").write_text(json.dumps({"response_metrics": response_metrics, "span_metrics": span_metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    viewer_path = output / args.viewer_file_name
    viewer_path.write_text(html_page(payload), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "n_records": len(records), "n_predicted_spans": len(derived_spans), "html": str(viewer_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
