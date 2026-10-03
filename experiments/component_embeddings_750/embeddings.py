"""Замороженный Qwen3-8B и воспроизводимые тексты компонентов."""

from __future__ import annotations

import gc
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io import Bundle, ROOT, component_key


@dataclass(frozen=True)
class EncoderProfile:
    key: str
    model_id: str
    revision: str
    dimension: int
    pooling: str
    max_tokens: int = 8192
    trust_remote_code: bool = False


ENCODERS = {
    "qwen3_8b": EncoderProfile(
        key="qwen3_8b", model_id="Qwen/Qwen3-Embedding-8B",
        revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        dimension=4096, pooling="last", max_tokens=8192),
    "qwen3_4b": EncoderProfile(
        key="qwen3_4b", model_id="Qwen/Qwen3-Embedding-4B",
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        dimension=2560, pooling="last", max_tokens=8192),
    "gte_large": EncoderProfile(
        key="gte_large", model_id="Alibaba-NLP/gte-large-en-v1.5",
        revision="104333d6af6f97649377c2afbde10a7704870c7b",
        dimension=1024, pooling="cls", max_tokens=8192, trust_remote_code=True),
}
PRIMARY_ENCODER = "qwen3_8b"
BASELINE_ENCODER = "gte_large"
MODEL_ID = ENCODERS[PRIMARY_ENCODER].model_id
MODEL_REVISION = ENCODERS[PRIMARY_ENCODER].revision
MAX_TOKENS = ENCODERS[PRIMARY_ENCODER].max_tokens
PROMPT_VERSION = "component-target-v1-no-reference-context"


def triple_parts(text: str) -> dict[str, str]:
    match = re.fullmatch(r"subject: (.*?); predicate: (.*?); object: (.*)", text,
                         flags=re.DOTALL)
    if not match or any(not match.group(i).strip() for i in (1, 2, 3)):
        raise ValueError(f"Неправильная строка тройки: {text!r}")
    return {"subject": match.group(1), "predicate": match.group(2),
            "object": match.group(3)}


@dataclass(frozen=True)
class TextJob:
    text: str
    kind: str
    key: str | int
    note: str


INSTRUCTIONS = {
    "entity": "Represent the target entity as it is used in the answer. Capture its semantic role and surrounding meaning. Do not assess factual support.",
    "relation": "Represent the target directed relation as it is used in the answer. Preserve the subject, predicate, object and direction. Do not assess factual support.",
    "claim": "Represent the semantic content of the factual claim. Do not assess factual support.",
    "answer": "Represent the semantic content of the complete answer. Do not assess factual support.",
}


def _query(instruction: str, body: str) -> str:
    return f"Instruct: {instruction}\nQuery: {body.strip()}"


def _component_prompt(component: dict, answer: str, *, contextual: bool) -> str:
    kind, value = component["component_type"], component["embedding_text"].strip()
    if kind == "entity":
        body = (f"Answer:\n{answer}\n\nTarget entity:\n{value}" if contextual
                else f"Entity:\n{value}")
    elif kind == "relation":
        parts = triple_parts(value)
        triple = (f"Subject: {parts['subject']}\nPredicate: {parts['predicate']}\n"
                  f"Object: {parts['object']}")
        body = (f"Answer:\n{answer}\n\nTarget directed relation:\n{triple}" if contextual
                else f"Directed relation:\n{triple}")
    elif kind == "claim":
        body = f"Claim:\n{value}"
    else:
        raise ValueError(kind)
    return _query(INSTRUCTIONS[kind], body)


def make_jobs(bundle: Bundle) -> tuple[list[TextJob], dict]:
    """Строит целевые тексты компонентов и ответов без опорного контекста."""
    jobs: list[TextJob] = []
    for component in bundle.components:
        sid = int(component["source_id"])
        answer = bundle.by_id[sid]["answer"]
        key = component_key(component)
        jobs.append(TextJob(_component_prompt(component, answer, contextual=True),
                            "context", key, f"context:{key}"))
    for sid in bundle.ids:
        answer = bundle.by_id[sid]["answer"]
        jobs.append(TextJob(_query(INSTRUCTIONS["answer"], f"Answer:\n{answer}"),
                            "answer_cls", sid, f"answer:{sid}"))
    return jobs, {
        "job_count": len(jobs), "context_jobs": len(bundle.components),
        "isolated_jobs": 0, "answer_jobs": len(bundle.ids),
        "reference_context_in_prompts": False, "prompt_version": PROMPT_VERSION,
    }


def example_prompts(bundle: Bundle) -> dict[str, str]:
    jobs, _ = make_jobs(bundle)
    examples = {}
    for component_type in ("entity", "relation", "claim"):
        component = next(row for row in bundle.components
                         if row["component_type"] == component_type)
        key = component_key(component)
        examples[f"{component_type}_context"] = next(
            job.text for job in jobs if job.kind == "context" and job.key == key)
    examples["answer"] = next(job.text for job in jobs if job.kind == "answer_cls")
    return examples


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Получен пустой или некорректный вектор")
    return (vector / norm).astype(np.float32)


def _tokenize(tokenizer, texts: list[str], profile: EncoderProfile):
    tokens = tokenizer(texts, padding=True, truncation=False, return_tensors="pt",
                       add_special_tokens=True)
    lengths = tokens["attention_mask"].sum(dim=1).tolist()
    if any(length > profile.max_tokens for length in lengths):
        bad = [(i, int(length)) for i, length in enumerate(lengths)
               if length > profile.max_tokens]
        raise ValueError(f"Вход длиннее {profile.max_tokens} токенов; усечение запрещено: {bad[:4]}")
    return tokens, [int(length) for length in lengths]


def _forward_vectors(tokenizer, model, texts: list[str], profile: EncoderProfile,
                     device: str) -> tuple[np.ndarray, list[int]]:
    import torch
    tokens, lengths = _tokenize(tokenizer, texts, profile)
    inputs = {name: value.to(device) for name, value in tokens.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        pooled = hidden[:, -1] if profile.pooling == "last" else hidden[:, 0]
        vectors = pooled.float().cpu().numpy()
    return vectors, lengths


def _batch_candidates(profile: EncoderProfile, device: str) -> list[int]:
    if device == "cpu":
        return [1] if profile.key.startswith("qwen3") else [8]
    return ([16, 8, 4, 2, 1] if profile.key.startswith("qwen3")
            else [128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1])


def _choose_batch(jobs: list[TextJob], tokenizer, model, profile: EncoderProfile,
                  device: str, requested: int | None) -> tuple[int, list[dict]]:
    if requested is not None:
        if requested < 1:
            raise ValueError("batch_size должен быть положительным")
        candidates = [int(requested)]
    else:
        candidates = _batch_candidates(profile, device)
    scanned = tokenizer([job.text for job in jobs], padding=False, truncation=False,
                        add_special_tokens=True, return_length=True)
    length_values = scanned.get("length")
    lengths = ([int(value) for value in length_values] if length_values is not None
               else [len(ids) for ids in scanned["input_ids"]])
    too_long = [(jobs[i].note, length) for i, length in enumerate(lengths)
                if length > profile.max_tokens]
    if too_long:
        raise ValueError(f"Вход длиннее {profile.max_tokens} токенов; усечение запрещено: "
                         f"{too_long[:4]}")
    order = np.argsort(lengths)[::-1]
    longest = [jobs[int(index)] for index in order]
    del scanned
    if device == "cpu":
        return candidates[0], [{"batch_size": candidates[0], "status": "cpu_default"}]
    import torch
    attempts = []
    total = int(torch.cuda.get_device_properties(0).total_memory)
    for candidate in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _forward_vectors(tokenizer, model, [job.text for job in longest[:candidate]],
                             profile, device)
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_reserved())
            attempts.append({"batch_size": candidate, "status": "ok",
                             "peak_reserved_bytes": peak})
            if peak <= 0.90 * total:
                return candidate, attempts
            attempts[-1]["status"] = "over_90_percent"
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            attempts.append({"batch_size": candidate, "status": "out_of_memory"})
        torch.cuda.empty_cache()
    raise RuntimeError("Не удалось подобрать безопасный размер пакета")


def encode_jobs(jobs: list[TextJob], tokenizer, model, *, profile: EncoderProfile,
                device: str, batch_size: int, progress: bool = True):
    from tqdm.auto import tqdm
    model.eval()
    result = {"context": {}, "answer_cls": {}}
    trace = []
    # Одинаковые тексты вычисляются один раз, связь со всеми компонентами сохраняется.
    text_jobs = {}
    for job in jobs:
        text_jobs.setdefault(job.text, []).append(job)
    ordered = sorted([group[0] for group in text_jobs.values()], key=lambda job: len(job.text))
    iterator = range(0, len(ordered), batch_size)
    if progress:
        iterator = tqdm(iterator, desc=f"Векторы {profile.key}")
    for start in iterator:
        batch = ordered[start:start + batch_size]
        vectors, lengths = _forward_vectors(tokenizer, model,
                                            [job.text for job in batch], profile, device)
        for first, vector, length in zip(batch, vectors, lengths):
            normalized = _unit(vector)
            for job in text_jobs[first.text]:
                if job.key in result[job.kind]:
                    raise ValueError(f"Повторный ключ: {job.kind}/{job.key}")
                result[job.kind][job.key] = normalized
                trace.append({"job": job.note, "kind": job.kind, "key": job.key,
                              "tokens": length, "pooling": profile.pooling,
                              "text": job.text,
                              "text_sha256": hashlib.sha256(job.text.encode()).hexdigest()})
    return result, trace


def extract_embeddings(bundle: Bundle, *, output_dir: Path | None = None,
                       encoder: str = PRIMARY_ENCODER, batch_size: int | None = None,
                       force: bool = False) -> dict:
    """Получает и сохраняет векторы одного закреплённого кодировщика."""
    if encoder not in ENCODERS:
        raise ValueError(f"Неизвестный кодировщик: {encoder}")
    profile = ENCODERS[encoder]
    quantization = "int8" if profile.key == "qwen3_8b" else "none"
    output_dir = Path(output_dir or ROOT / "artifacts")
    output_dir.mkdir(exist_ok=True, parents=True)
    cache = output_dir / "embeddings.npz"
    meta_path = output_dir / "embeddings_manifest.json"
    trace_path = output_dir / "embedding_trace.jsonl"
    fingerprint = hashlib.sha256((bundle.signature + profile.model_id + profile.revision
                                  + PROMPT_VERSION + quantization).encode()).hexdigest()
    if cache.exists() and meta_path.exists() and trace_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            with np.load(cache) as loaded:
                arrays = {name: loaded[name] for name in ("context", "answer_cls")}
            expected = ((len(bundle.components), profile.dimension),
                        (len(bundle.ids), profile.dimension))
            if all(array.shape == shape and np.isfinite(array).all()
                   for array, shape in zip(arrays.values(), expected)):
                return arrays | {"manifest": meta}

    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile.model_id, revision=profile.revision,
                                               trust_remote_code=profile.trust_remote_code,
                                               use_fast=True)
    tokenizer.padding_side = "left" if profile.pooling == "last" else "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    load_kwargs = dict(revision=profile.revision, trust_remote_code=profile.trust_remote_code,
                       use_safetensors=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    if profile.key.startswith("qwen3"):
        load_kwargs["attn_implementation"] = "sdpa"
    if profile.key == "qwen3_8b":
        if device != "cuda":
            raise RuntimeError("Для Qwen3-8B выберите T4 GPU в Colab")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = {"": 0}
        model = AutoModel.from_pretrained(profile.model_id, **load_kwargs).eval()
    else:
        model = AutoModel.from_pretrained(profile.model_id, **load_kwargs).to(device).eval()
    dimension = int(model.config.hidden_size)
    if dimension != profile.dimension:
        raise ValueError(f"Ожидалось {profile.dimension} координат, получено {dimension}")
    jobs, stats = make_jobs(bundle)
    selected_batch, attempts = _choose_batch(jobs, tokenizer, model, profile, device,
                                              batch_size)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    vectors, trace = encode_jobs(jobs, tokenizer, model, profile=profile, device=device,
                                 batch_size=selected_batch)
    peak_allocated = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0
    gpu_total = (int(torch.cuda.get_device_properties(0).total_memory)
                 if device == "cuda" else None)
    ids = [component_key(row) for row in bundle.components]
    arrays = {
        "context": np.stack([vectors["context"][key] for key in ids]),
        "answer_cls": np.stack([vectors["answer_cls"][sid] for sid in bundle.ids]),
    }
    if any(array.shape[1] != dimension or not np.isfinite(array).all()
           for array in arrays.values()):
        raise ValueError("Неожиданная размерность или некорректный вектор")
    np.savez_compressed(cache, **arrays)
    trace_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in trace) + "\n",
                          encoding="utf-8")
    meta = {
        "fingerprint": fingerprint, "encoder_key": profile.key,
        "model_id": profile.model_id, "revision": profile.revision,
        "data_signature": bundle.signature, "prompt_version": PROMPT_VERSION,
        "reference_context_in_prompts": False, "device": device,
        "precision": "float16" if device == "cuda" else "float32",
        "weight_quantization": quantization,
        "attention_implementation": "sdpa" if profile.key.startswith("qwen3") else "model_default",
        "pooling": profile.pooling, "embedding_dim": dimension,
        "max_tokens": profile.max_tokens, "batch_size": selected_batch,
        "batch_probe_attempts": attempts, "gpu_total_bytes": gpu_total,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "component_order": ids, "response_order": bundle.ids,
        "normalization": "L2 per vector", "max_observed_tokens": max(x["tokens"] for x in trace),
        **stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"manifest": meta}
