"""Локальная проверка без скачивания кодировщиков и без полного обучения."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_io import load_bundle
from embeddings import ENCODERS, encode_jobs, make_jobs, triple_parts
from experiment import LINEAR_METHODS
from features import feature_matrix, make_fold_view, padded_tensors
from models import NEURAL_METHODS, SetDetector, one_step_smoke


def check_tokenization(bundle, tokenizer_dir: Path):
    """Доступный локальный токенизатор проверяет полный проход нового интерфейса."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                              use_fast=True)
    jobs, _ = make_jobs(bundle)
    selected = jobs[:8]
    enc = tokenizer([job.text for job in selected], truncation=False, padding=True)
    assert all(len(row) <= ENCODERS["gte_large"].max_tokens for row in enc["input_ids"])

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
            base = input_ids.float().unsqueeze(-1)
            frequencies = torch.arange(1, 1025, device=base.device).float()
            output = torch.sin(base / frequencies)
            return type("FakeOutput", (), {"last_hidden_state": output})()

    tokenizer.padding_side = "right"
    first = [next(j for j in jobs if j.kind == kind)
             for kind in ("context", "triple", "answer_cls")]
    result, trace = encode_jobs(first, tokenizer, FakeModel(),
                                profile=ENCODERS["gte_large"], device="cpu",
                                batch_size=2, progress=False)
    assert len(trace) == 3 and all(len(v) == 1 for v in result.values())
    return len(jobs)


def check_shapes(bundle):
    rng = np.random.default_rng(42)
    embeddings = {
        "context": rng.normal(size=(len(bundle.components), 2560)).astype(np.float32),
        "triple": rng.normal(size=(len(bundle.components), 2560)).astype(np.float32),
        "answer_cls": rng.normal(size=(len(bundle.ids), 2560)).astype(np.float32),
    }
    train_idx = np.where(bundle.fold_numbers != 0)[0]
    test_idx = np.where(bundle.fold_numbers == 0)[0]
    view = make_fold_view(bundle, embeddings, train_idx)
    triple = make_fold_view(bundle, embeddings, train_idx, source="triple")
    for method in LINEAR_METHODS:
        base = {"B_triples": "B", "M0_triples": "M0"}.get(method, method)
        matrix = feature_matrix(triple if method.endswith("triples") else view, base)
        assert matrix.shape[0] == 100 and np.isfinite(matrix).all(), method
        assert matrix.shape[1] > 0, method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            detector = make_pipeline(StandardScaler(), LogisticRegression(
                solver="liblinear", max_iter=1, C=1.0))
            detector.fit(matrix[train_idx], bundle.y[train_idx])
        probability = detector.predict_proba(matrix[test_idx])[:, 1]
        assert probability.shape == (20,) and np.isfinite(probability).all(), method
    for method in NEURAL_METHODS:
        model = SetDetector(method)
        batch = padded_tensors(view, test_idx)
        raw = model(batch)
        assert tuple(raw.shape) == (20,) and bool(raw.isfinite().all()), method
        loss = one_step_smoke(method, view, train_idx[:8], bundle.y)
        assert np.isfinite(loss), method
    return len(LINEAR_METHODS), len(NEURAL_METHODS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    bundle = load_bundle()
    jobs, stats = make_jobs(bundle)
    assert len(jobs) == 2 * len(bundle.components) + len(bundle.ids) == 4870
    assert stats["reference_context_in_prompts"] is False
    assert {job.kind for job in jobs} == {"context", "triple", "answer_cls"}
    for relation in (r for r in bundle.components if r["component_type"] == "relation"):
        triple_parts(relation["embedding_text"])
    if args.tokenizer_dir:
        checked = check_tokenization(bundle, args.tokenizer_dir)
        print(f"Токенизация проверена у {checked} входов")
    linear, neural = check_shapes(bundle)
    print(f"OK: 100 ответов, 2385 компонентов, {linear} линейных вариантов, "
          f"{neural} нейросетей по одному шагу; полное обучение не запускалось")


if __name__ == "__main__":
    main()
