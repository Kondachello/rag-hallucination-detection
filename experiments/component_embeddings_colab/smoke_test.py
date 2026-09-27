"""Локальная проверка без скачивания BGE и без полного обучения."""

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
from embeddings import encode_jobs, make_jobs, triple_parts
from experiment import LINEAR_METHODS
from features import feature_matrix, make_fold_view, padded_tensors
from models import NEURAL_METHODS, SetDetector, one_step_smoke


def check_token_positions(bundle, tokenizer_dir: Path):
    """Местный BERT-токенизатор проверяет контракт позиций; Colab повторит с BGE."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                              use_fast=True)
    assert tokenizer.is_fast
    jobs, _ = make_jobs(bundle)
    for pair in (False, True):
        selected = [j for j in jobs if (j.second is not None) == pair]
        for start in range(0, len(selected), 32):
            batch = selected[start:start + 32]
            enc = tokenizer([j.first for j in batch],
                            [j.second for j in batch] if pair else None,
                            truncation=False, padding=True, return_offsets_mapping=True)
            for i, job in enumerate(batch):
                assert len(enc["input_ids"][i]) <= 512
                seq = enc.sequence_ids(i)
                offsets = enc["offset_mapping"][i]
                for _, _, part, left, right in job.targets:
                    if left is None:
                        continue
                    pos = [p for p, (a, b) in enumerate(offsets)
                           if seq[p] == part and b > left and a < right]
                    assert pos, (job.note, left, right)
    # Проверяем и фактический проход функции извлечения на нескольких разных входах.
    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            base = input_ids.float().unsqueeze(-1)
            frequencies = torch.arange(1, 385, device=base.device).float()
            output = torch.sin(base / frequencies)
            return type("FakeOutput", (), {"last_hidden_state": output})()

    first = [next(j for j in jobs if j.note.startswith(prefix))
             for prefix in ("answer:", "answer_plus_triple:", "triple_only:", "claim:")]
    result, trace = encode_jobs(first, tokenizer, FakeModel(), device="cpu", batch_size=2,
                                progress=False)
    assert len(trace) >= 6 and all(v for v in result.values() if v)
    return len(jobs)


def check_shapes(bundle):
    rng = np.random.default_rng(42)
    embeddings = {
        "context": rng.normal(size=(len(bundle.components), 384)).astype(np.float32),
        "triple": rng.normal(size=(len(bundle.components), 384)).astype(np.float32),
        "answer_cls": rng.normal(size=(len(bundle.ids), 384)).astype(np.float32),
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
    assert stats["answer_entity_fallbacks"] == 5
    assert len(jobs) > 2300
    for relation in (r for r in bundle.components if r["component_type"] == "relation"):
        triple_parts(relation["embedding_text"])
    if args.tokenizer_dir:
        checked = check_token_positions(bundle, args.tokenizer_dir)
        print(f"Позиции проверены у {checked} контекстов (локальный BERT-токенизатор)")
    linear, neural = check_shapes(bundle)
    print(f"OK: 100 ответов, 2385 компонентов, {linear} линейных вариантов, "
          f"{neural} нейросетей по одному шагу; полное обучение не запускалось")


if __name__ == "__main__":
    main()
