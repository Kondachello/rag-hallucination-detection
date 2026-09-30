"""Заранее заданная проверка вклада векторов и сравнение кодировщиков."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data_io import Bundle
from experiment import CS, DESCRIPTIONS, _linear
from features import feature_matrix, make_fold_view


CORE_METHODS = ("A", "B", "M0", "E_only", "R_only", "C_only",
                "A+E", "A+R", "A+C", "M14_q_cls")
PRIMARY_REFERENCE = "CONFIRMATION|A"
PRIMARY_CANDIDATE = "qwen3_4b|M0"
COMPARISONS = (
    (PRIMARY_REFERENCE, PRIMARY_CANDIDATE, "primary"),
    (PRIMARY_REFERENCE, "qwen3_4b|B", "secondary"),
    ("gte_large|M0", "qwen3_4b|M0", "secondary"),
    ("qwen3_4b|M0_SHUFFLED", "qwen3_4b|M0", "negative_control"),
    (PRIMARY_REFERENCE, "qwen3_4b|A+E", "type_ablation"),
    (PRIMARY_REFERENCE, "qwen3_4b|A+R", "type_ablation"),
    (PRIMARY_REFERENCE, "qwen3_4b|A+C", "type_ablation"),
)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "F1_at_0.5": f1_score(y, p >= 0.5), "coverage": len(p),
    }


def _paired_interval(y: np.ndarray, reference: np.ndarray, candidate: np.ndarray,
                     repeats: int = 5000) -> dict:
    positive, negative = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(20260930)
    auc_delta, loss_delta = np.empty(repeats), np.empty(repeats)
    for i in range(repeats):
        sample = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
        auc_delta[i] = (roc_auc_score(y[sample], candidate[sample])
                        - roc_auc_score(y[sample], reference[sample]))
        loss_delta[i] = (log_loss(y[sample], np.clip(candidate[sample], 1e-7, 1 - 1e-7),
                                  labels=[0, 1])
                         - log_loss(y[sample], np.clip(reference[sample], 1e-7, 1 - 1e-7),
                                    labels=[0, 1]))
    return {
        "ROC_AUC_difference": roc_auc_score(y, candidate) - roc_auc_score(y, reference),
        "ROC_AUC_95_low": float(np.quantile(auc_delta, 0.025)),
        "ROC_AUC_95_high": float(np.quantile(auc_delta, 0.975)),
        "log_loss_difference": (log_loss(y, np.clip(candidate, 1e-7, 1 - 1e-7), labels=[0, 1])
                                - log_loss(y, np.clip(reference, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "log_loss_95_low": float(np.quantile(loss_delta, 0.025)),
        "log_loss_95_high": float(np.quantile(loss_delta, 0.975)),
    }


def _select_c_from_matrices(
        inner_matrices: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]],
                            y: np.ndarray) -> float:
    losses = []
    for c in CS:
        fold_losses = []
        for matrix, (fit_idx, val_idx) in inner_matrices:
            probability = _linear(c).fit(
                matrix[fit_idx], y[fit_idx]).predict_proba(matrix[val_idx])[:, 1]
            fold_losses.append(log_loss(y[val_idx], probability, labels=[0, 1]))
        losses.append(float(np.mean(fold_losses)))
    return CS[int(np.argmin(losses))]


def _encoder_predictions(bundle: Bundle, embeddings: dict):
    """Считает основные варианты, переиспользуя PCA внутри каждого разбиения."""
    y, folds = bundle.y, bundle.fold_numbers
    predictions = {method: np.full(len(y), np.nan) for method in CORE_METHODS}
    details, m0_cache = [], []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold in sorted(set(folds)):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        outer_view = make_fold_view(bundle, embeddings, train_idx)
        outer_matrices = {
            method: feature_matrix(outer_view, method) for method in CORE_METHODS
        }
        inner_by_method = {method: [] for method in CORE_METHODS}
        for fit_local, val_local in splitter.split(train_idx, y[train_idx]):
            fit_idx, val_idx = train_idx[fit_local], train_idx[val_local]
            inner_view = make_fold_view(bundle, embeddings, fit_idx)
            indices = (fit_idx, val_idx)
            for method in CORE_METHODS:
                inner_by_method[method].append(
                    (feature_matrix(inner_view, method), indices))
        selected = {}
        for method in CORE_METHODS:
            c = _select_c_from_matrices(inner_by_method[method], y)
            matrix = outer_matrices[method]
            model = _linear(c).fit(matrix[train_idx], y[train_idx])
            predictions[method][test_idx] = model.predict_proba(matrix[test_idx])[:, 1]
            selected[method] = c
            details.append({"fold": int(fold), "method": method, "C": c,
                            "feature_count": int(matrix.shape[1])})
        m0_cache.append({"fold": int(fold), "train_idx": train_idx,
                         "test_idx": test_idx, "C": selected["M0"],
                         "matrix": outer_matrices["M0"]})
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise AssertionError("Не для всех ответов есть внеобучающее предсказание")
    return predictions, details, m0_cache


def _shuffled_control(bundle: Bundle, fold_cache: list[dict], repeats: int = 20):
    """Сохраняет размер признаков, но ломает соответствие вектор↔ответ."""
    y = bundle.y
    prediction = np.full(len(y), np.nan)
    details = []
    for cached in fold_cache:
        fold = cached["fold"]
        train_idx, test_idx = cached["train_idx"], cached["test_idx"]
        c, matrix = cached["C"], cached["matrix"]
        # M0 = 13 подтверждений + 24 PCA-координаты + один признак пропущенного span.
        semantic = slice(13, matrix.shape[1] - 1)
        repeated = []
        for repeat in range(repeats):
            rng = np.random.default_rng(9000 + 101 * int(fold) + repeat)
            shuffled = matrix.copy()
            shuffled[train_idx, semantic] = matrix[rng.permutation(train_idx), semantic]
            shuffled[test_idx, semantic] = matrix[rng.permutation(test_idx), semantic]
            model = _linear(c).fit(shuffled[train_idx], y[train_idx])
            repeated.append(model.predict_proba(shuffled[test_idx])[:, 1])
        prediction[test_idx] = np.mean(repeated, axis=0)
        details.append({"fold": int(fold), "C": c, "repeats": repeats,
                        "train_rows": len(train_idx), "test_rows": len(test_idx)})
    if not np.isfinite(prediction).all():
        raise AssertionError("Неполный отрицательный контроль")
    return prediction, details


def run_encoder_comparison(bundle: Bundle, encoder_embeddings: dict[str, dict], *,
                           output_dir: Path, shuffle_repeats: int = 20):
    """Запускает одну основную и заранее помеченные вторичные проверки."""
    required = {"qwen3_4b", "gte_large"}
    if set(encoder_embeddings) != required:
        raise ValueError(f"Нужны кодировщики {sorted(required)}")
    y = bundle.y
    all_predictions: dict[str, np.ndarray] = {}
    descriptions: dict[str, str] = {}
    fold_details = {}
    qwen_m0_cache = None
    for encoder, embeddings in encoder_embeddings.items():
        predictions, details, m0_cache = _encoder_predictions(bundle, embeddings)
        fold_details[encoder] = details
        if encoder == "qwen3_4b":
            qwen_m0_cache = m0_cache
        for method, probability in predictions.items():
            name = f"{encoder}|{method}"
            all_predictions[name] = probability
            descriptions[name] = DESCRIPTIONS[method]
    if not np.allclose(all_predictions["qwen3_4b|A"], all_predictions["gte_large|A"],
                       rtol=0, atol=1e-12):
        raise AssertionError("Контроль только на подтверждённости зависит от кодировщика")
    all_predictions[PRIMARY_REFERENCE] = all_predictions.pop("qwen3_4b|A")
    descriptions[PRIMARY_REFERENCE] = "13 признаков подтверждённости; логистическая регрессия"
    all_predictions.pop("gte_large|A")
    descriptions.pop("qwen3_4b|A", None)
    descriptions.pop("gte_large|A", None)
    if qwen_m0_cache is None:
        raise AssertionError("Не построен кэш основного кодировщика")
    shuffled, shuffle_details = _shuffled_control(
        bundle, qwen_m0_cache, repeats=shuffle_repeats)
    all_predictions["qwen3_4b|M0_SHUFFLED"] = shuffled
    descriptions["qwen3_4b|M0_SHUFFLED"] = (
        "Те же признаки, но векторы случайно переставлены между ответами")
    fold_details["qwen3_4b_shuffled"] = shuffle_details

    metric_rows = []
    for method, probability in all_predictions.items():
        encoder, variant = method.split("|", 1)
        metric_rows.append({"method": method, "encoder": encoder, "variant": variant,
                            "description": descriptions[method], **_metrics(y, probability)})
    metrics = pd.DataFrame(metric_rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    prediction_rows = []
    response_ids = [bundle.by_id[sid]["response_id"] for sid in bundle.ids]
    for method, probability in all_predictions.items():
        prediction_rows.append(pd.DataFrame({
            "source_id": bundle.ids, "response_id": response_ids,
            "fold": bundle.fold_numbers, "gold": y, "method": method,
            "probability": probability,
        }))
    predictions = pd.concat(prediction_rows, ignore_index=True)
    comparison_rows = []
    for reference, candidate, role in COMPARISONS:
        comparison_rows.append({"reference": reference, "candidate": candidate,
                                "role": role,
                                **_paired_interval(y, all_predictions[reference],
                                                   all_predictions[candidate])})
    comparisons = pd.DataFrame(comparison_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "encoder_metrics.csv", index=False)
    predictions.to_csv(output_dir / "encoder_oof_predictions.csv", index=False)
    comparisons.to_csv(output_dir / "encoder_comparisons.csv", index=False)
    (output_dir / "encoder_fold_details.json").write_text(
        json.dumps(fold_details, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "dataset_signature": bundle.signature, "n_answers": len(y),
        "primary_reference": PRIMARY_REFERENCE, "primary_candidate": PRIMARY_CANDIDATE,
        "primary_metric": "ROC_AUC_difference",
        "secondary_metrics": ["log_loss_difference", "PR_AUC", "F1_at_0.5"],
        "outer_folds": sorted(set(int(x) for x in bundle.fold_numbers)),
        "regularization_selection": "3-fold inner CV on outer training rows",
        "pca_selection": "fit separately on each training split",
        "shuffle_repeats": shuffle_repeats,
        "scope": "Primary comparison is confirmatory within this pilot; all grid methods are exploratory.",
    }
    (output_dir / "encoder_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics, predictions, comparisons
