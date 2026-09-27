"""Замороженный BERT: позиции компонентов, проверка длины и кэш векторов."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io import Bundle, ROOT, component_key

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MAX_TOKENS = 512


def triple_parts(text: str) -> dict[str, tuple[int, int] | str]:
    """Разбирает сохранённую направленную тройку и её точные символьные позиции."""
    match = re.fullmatch(r"subject: (.*?); predicate: (.*?); object: (.*)", text, flags=re.DOTALL)
    if not match or any(not match.group(i).strip() for i in (1, 2, 3)):
        raise ValueError(f"Неправильная строка тройки: {text!r}")
    return {
        "subject": match.group(1), "predicate": match.group(2), "object": match.group(3),
        "subject_span": match.span(1), "predicate_span": match.span(2),
        "object_span": match.span(3),
    }


def norm_name(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass
class Job:
    first: str
    second: str | None
    # (вид результата, component_id/source_id, номер текстовой части, start, end)
    targets: list[tuple[str, str | int, int, int | None, int | None]]
    note: str


def make_jobs(bundle: Bundle) -> tuple[list[Job], dict]:
    jobs = []
    per_sid: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in bundle.components:
        per_sid[int(row["source_id"])][row["component_type"]].append(row)
    fallback_answer = 0
    fallback_triple = 0
    linked_triple_mentions = 0
    for sid in bundle.ids:
        answer = bundle.by_id[sid]["answer"]
        rows = per_sid[sid]
        answer_targets = [("answer_cls", sid, 0, None, None)]
        for ent in rows["entity"]:
            if ent["answer_start"] is None:
                fallback_answer += 1
                name = ent["embedding_text"]
                jobs.append(Job(name, None, [("context", component_key(ent), 0, 0, len(name))],
                                f"entity_name_fallback:{ent['component_id']}"))
            else:
                answer_targets.append(("context", component_key(ent), 0,
                                       int(ent["answer_start"]), int(ent["answer_end"])))
        jobs.append(Job(answer, None, answer_targets, f"answer:{sid}"))
        entity_names: dict[str, list[str]] = defaultdict(list)
        for ent in rows["entity"]:
            entity_names[norm_name(ent["embedding_text"])].append(component_key(ent))
        ambiguous = {name: values for name, values in entity_names.items() if len(values) > 1}
        if ambiguous:
            raise ValueError(f"Нельзя однозначно связать конец тройки с сущностью {sid}: {ambiguous}")
        triple_hits: dict[str, int] = defaultdict(int)
        for rel in rows["relation"]:
            cid = component_key(rel)
            triple = rel["embedding_text"]
            parts = triple_parts(triple)
            jobs.append(Job(answer, triple,
                            [("context", cid, 1, *parts["predicate_span"])],
                            f"answer_plus_triple:{cid}"))
            targets = [("triple", cid, 0, *parts["predicate_span"])]
            for name, span_name in (("subject", "subject_span"), ("object", "object_span")):
                for ent_id in entity_names.get(norm_name(str(parts[name])), []):
                    targets.append(("triple_entity", ent_id, 0, *parts[span_name]))
                    triple_hits[ent_id] += 1
                    linked_triple_mentions += 1
            jobs.append(Job(triple, None, targets, f"triple_only:{cid}"))
        for ent in rows["entity"]:
            cid = component_key(ent)
            if triple_hits[cid] == 0:
                fallback_triple += 1
                name = ent["embedding_text"]
                jobs.append(Job(name, None, [("triple_entity", cid, 0, 0, len(name))],
                                f"triple_entity_name_fallback:{cid}"))
        for claim in rows["claim"]:
            cid = component_key(claim)
            jobs.append(Job(claim["embedding_text"], None,
                            [("context", cid, 0, None, None),
                             ("triple", cid, 0, None, None)], f"claim:{cid}"))
    return jobs, {
        "answer_entity_fallbacks": fallback_answer,
        "triple_entity_fallbacks": fallback_triple,
        "triple_entity_linked_mentions": linked_triple_mentions,
        "job_count": len(jobs),
    }


def example_prompts(bundle: Bundle) -> dict[str, str]:
    jobs, _ = make_jobs(bundle)
    pair = next(j for j in jobs if j.second is not None)
    claim = next(j for j in jobs if j.note.startswith("claim:"))
    return {"answer": pair.first, "triple": pair.second or "", "claim": claim.first,
            "relation_predicate": pair.second[pair.targets[0][3]:pair.targets[0][4]]}


def _unit(vector: np.ndarray) -> np.ndarray:
    size = float(np.linalg.norm(vector))
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Получен пустой или некорректный вектор")
    return (vector / size).astype(np.float32)


def encode_jobs(jobs: list[Job], tokenizer, model, *, device: str, batch_size: int = 16,
                progress: bool = True) -> tuple[dict[str, dict], list[dict]]:
    """Поддерживает также тестовые токенизатор и модель без сетевого доступа."""
    import torch
    from tqdm.auto import tqdm

    model.eval()
    model.to(device)
    result: dict[str, dict] = {"context": {}, "triple": {}, "triple_entity": {}, "answer_cls": {}}
    trace = []
    # В пачке либо одиночные тексты, либо пары: sequence_ids тогда однозначен.
    for is_pair in (False, True):
        selected = [j for j in jobs if (j.second is not None) == is_pair]
        iterator = range(0, len(selected), batch_size)
        if progress:
            iterator = tqdm(iterator, desc="Пары" if is_pair else "Одиночные тексты")
        for start in iterator:
            batch = selected[start:start + batch_size]
            first = [j.first for j in batch]
            second = [j.second for j in batch] if is_pair else None
            tokens = tokenizer(first, second, padding=True, truncation=False,
                               return_offsets_mapping=True, return_tensors="pt")
            lengths = tokens["attention_mask"].sum(dim=1).tolist()
            if any(n > MAX_TOKENS for n in lengths):
                bad = [(j.note, n) for j, n in zip(batch, lengths) if n > MAX_TOKENS]
                raise ValueError(f"Вход длиннее {MAX_TOKENS} токенов, усечение запрещено: {bad[:4]}")
            offsets = tokens.pop("offset_mapping").cpu().numpy()
            seq_ids = [tokens.sequence_ids(i) for i in range(len(batch))]
            inputs = {k: v.to(device) for k, v in tokens.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state.detach().float().cpu().numpy()
            for i, job in enumerate(batch):
                valid = int(lengths[i])
                for kind, key, part, left, right in job.targets:
                    if left is None:
                        vec = hidden[i, 0]
                        positions = [0]
                    else:
                        positions = [p for p in range(valid)
                                     if seq_ids[i][p] == part and offsets[i, p, 1] > left
                                     and offsets[i, p, 0] < right]
                        if not positions:
                            raise ValueError(f"Не найдены токены {job.note}, {kind}, {left}:{right}")
                        vec = hidden[i, positions].mean(axis=0)
                    result[kind].setdefault(key, []).append(_unit(vec))
                    trace.append({"job": job.note, "kind": kind, "key": key,
                                  "part": part, "char_span": [left, right],
                                  "token_positions": positions, "tokens": valid,
                                  "text_sha256": hashlib.sha256(
                                      (job.first + "\x00" + (job.second or "")).encode()).hexdigest()})
    return result, trace


def extract_embeddings(bundle: Bundle, *, output_dir: Path | None = None,
                       batch_size: int = 16, force: bool = False) -> dict:
    """Скачивает закреплённую модель только при первом вызове, сохраняет кэш."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_dir = output_dir or ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True, parents=True)
    cache = output_dir / "embeddings.npz"
    meta_path = output_dir / "embeddings_manifest.json"
    trace_path = output_dir / "embedding_trace.jsonl"
    fingerprint = hashlib.sha256((bundle.signature + MODEL_ID + MODEL_REVISION
                                  + "v2-normalized-spans-and-cls").encode()).hexdigest()
    if cache.exists() and meta_path.exists() and trace_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            with np.load(cache) as loaded:
                arrays = {key: loaded[key] for key in ("context", "triple", "answer_cls")}
            expected = ((len(bundle.components), 384), (len(bundle.components), 384),
                        (len(bundle.ids), 384))
            if all(value.shape == shape and np.isfinite(value).all()
                   for value, shape in zip(arrays.values(), expected)):
                return arrays | {"manifest": meta}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise RuntimeError("Для символьных позиций нужен быстрый токенизатор")
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                      trust_remote_code=False, use_safetensors=True)
    jobs, stats = make_jobs(bundle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vectors, trace = encode_jobs(jobs, tokenizer, model, device=device, batch_size=batch_size)
    ids = [component_key(r) for r in bundle.components]
    answer_ids = bundle.ids
    arrays = {
        "context": np.stack([vectors["context"][cid][0] for cid in ids]),
        "triple": np.stack([
            np.mean(vectors["triple_entity"][cid], axis=0) if r["component_type"] == "entity"
            else vectors["triple"][cid][0]
            for cid, r in zip(ids, bundle.components)
        ]),
        "answer_cls": np.stack([vectors["answer_cls"][sid][0] for sid in answer_ids]),
    }
    arrays["triple"] = np.stack([_unit(v) for v in arrays["triple"]])
    if any(a.shape[-1] != 384 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Неожиданная размерность/нечисловой вектор BGE-small")
    np.savez_compressed(cache, **arrays)
    trace_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in trace) + "\n", encoding="utf-8")
    meta = {"fingerprint": fingerprint, "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "data_signature": bundle.signature, "device": device, "batch_size": batch_size,
            "component_order": ids, "response_order": answer_ids,
            "normalization": "L2 per component; triple entity average then L2",
            "encoder": "last_hidden_state span mean or CLS", **stats,
            "max_observed_tokens": max(row["tokens"] for row in trace)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return arrays | {"manifest": meta}
