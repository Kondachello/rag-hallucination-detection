"""Расширенный опыт: матрица представлений данных и моделей."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

from data_io import Bundle
from grid_features import DATA_VARIANTS, build_tabular_features
from grid_models import (SET_DATA_VARIANTS, SET_MODELS, TABULAR_MODELS,
                         fit_predict_confirmation_mlp, fit_predict_set,
                         make_tabular_model)


PRIMARY_COMPARISONS = (
    ("D_CONFIRMATION|logreg_l2", "D_CONFIRMATION|torch_mlp"),
    ("D_CONFIRMATION|logreg_l2", "D_CONF_RAW_MEAN|logreg_l2"),
    ("D_CONFIRMATION|logreg_l2", "D_CONF_RAW_MEANMAX|gradient_boosting"),
    ("D_CONFIRMATION|logreg_l2", "D_SET_CONTEXT_Q|set_max"),
    ("D_CONFIRMATION|logreg_l2", "D_SET_CONTEXT_Q_STATUS|set_attention"),
)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Модель вернула некорректные вероятности")
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "F1_at_0.5": f1_score(y, p >= 0.5),
        "coverage": len(p),
    }


def _stratified_auc_difference(y: np.ndarray, reference: np.ndarray, candidate: np.ndarray,
                               *, repeats: int = 3000) -> tuple[float, float, float]:
    observed = roc_auc_score(y, candidate) - roc_auc_score(y, reference)
    positive, negative = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(20260927)
    differences = np.empty(repeats)
    for i in range(repeats):
        sample = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
        differences[i] = (roc_auc_score(y[sample], candidate[sample]) -
                          roc_auc_score(y[sample], reference[sample]))
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(observed), float(low), float(high)


def run_grid_experiment(bundle: Bundle, embeddings: dict, *, output_dir: Path,
                        tabular_variants: tuple[str, ...] = tuple(DATA_VARIANTS),
                        tabular_models: tuple[str, ...] = tuple(TABULAR_MODELS),
                        set_variants: tuple[str, ...] = tuple(SET_DATA_VARIANTS),
                        set_models: tuple[str, ...] = tuple(SET_MODELS),
                        neural_epochs: int = 80):
    """Возвращает метрики, матрицу ROC-AUC, прогнозы и сравнения.

    Настройки моделей фиксированы до просмотра результатов. Каждый прогноз
    строится моделью, не видевшей метку соответствующей внешней части.
    """
    unknown_data = set(tabular_variants) - set(DATA_VARIANTS)
    unknown_models = set(tabular_models) - set(TABULAR_MODELS)
    if unknown_data or unknown_models:
        raise ValueError((unknown_data, unknown_models))
    y, folds, n = bundle.y, bundle.fold_numbers, len(bundle.ids)
    predictions: dict[str, np.ndarray] = {
        f"{data}|{model}": np.full(n, np.nan)
        for data in tabular_variants for model in tabular_models
    }
    predictions.update({
        f"{data}|{model}": np.full(n, np.nan)
        for data in set_variants for model in set_models
    })
    predictions["D_CONFIRMATION|torch_mlp"] = np.full(n, np.nan)
    predictions["BASELINE|frequency"] = np.full(n, np.nan)
    predictions["BASELINE|hallugraph_raw"] = bundle.hallugraph_risk.copy()
    details = []

    for fold in sorted(set(folds)):
        train_idx, test_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
        predictions["BASELINE|frequency"][test_idx] = y[train_idx].mean()
        for data_variant in tabular_variants:
            matrix = build_tabular_features(bundle, embeddings, train_idx, data_variant)
            for model_name in tabular_models:
                method = f"{data_variant}|{model_name}"
                model = make_tabular_model(model_name, 2026 + int(fold))
                model.fit(matrix[train_idx], y[train_idx])
                p = model.predict_proba(matrix[test_idx])[:, 1]
                predictions[method][test_idx] = p
                details.append({"fold": int(fold), "data_variant": data_variant,
                                "model": model_name, "feature_count": matrix.shape[1]})
            print(f"Часть {fold}: {data_variant} x {len(tabular_models)} табличных моделей")

        p, detail = fit_predict_confirmation_mlp(
            bundle, train_idx, test_idx, seed=3026 + int(fold), epochs=neural_epochs)
        predictions["D_CONFIRMATION|torch_mlp"][test_idx] = p
        details.append({"fold": int(fold), "data_variant": "D_CONFIRMATION",
                        "model": "torch_mlp", **detail})

        for data_variant in set_variants:
            for model_name in set_models:
                method = f"{data_variant}|{model_name}"
                p, detail = fit_predict_set(
                    bundle, embeddings, data_variant, model_name, train_idx, test_idx,
                    seed=4026 + int(fold), epochs=neural_epochs)
                predictions[method][test_idx] = p
                details.append({"fold": int(fold), "data_variant": data_variant,
                                "model": model_name, **detail})
            print(f"Часть {fold}: {data_variant} x {len(set_models)} моделей наборов")

    incomplete = [name for name, p in predictions.items() if not np.isfinite(p).all()]
    if incomplete:
        raise AssertionError(f"Нет всех внеобучающих прогнозов: {incomplete}")

    descriptions = {**DATA_VARIANTS, **SET_DATA_VARIANTS,
                    "BASELINE": "Контрольные варианты"}
    model_descriptions = {**TABULAR_MODELS, **SET_MODELS,
                          "torch_mlp": "Сеть 13→16→8→1 только для подтверждённости",
                          "frequency": "Частота класса в обучающей части",
                          "hallugraph_raw": "Исходный строгий риск HalluGraph"}
    rows = []
    for method, p in predictions.items():
        data_variant, model_name = method.split("|", 1)
        rows.append({"method": method, "data_variant": data_variant,
                     "data_description": descriptions[data_variant], "model": model_name,
                     "model_description": model_descriptions[model_name], **_metrics(y, p)})
    metrics = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    matrix = metrics.pivot(index="data_variant", columns="model", values="ROC_AUC")

    long_predictions = []
    response_ids = [bundle.by_id[sid]["response_id"] for sid in bundle.ids]
    for method, p in predictions.items():
        data_variant, model_name = method.split("|", 1)
        long_predictions.append(pd.DataFrame({
            "source_id": bundle.ids, "response_id": response_ids, "fold": folds,
            "gold": y, "data_variant": data_variant, "model": model_name,
            "method": method, "probability": p,
        }))
    prediction_table = pd.concat(long_predictions, ignore_index=True)

    comparisons = []
    for reference, candidate in PRIMARY_COMPARISONS:
        if reference in predictions and candidate in predictions:
            difference, low, high = _stratified_auc_difference(
                y, predictions[reference], predictions[candidate])
            comparisons.append({"reference": reference, "candidate": candidate,
                                "ROC_AUC_difference": difference,
                                "bootstrap_95_low": low, "bootstrap_95_high": high,
                                "predefined": True})
    comparisons = pd.DataFrame(comparisons)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "grid_metrics.csv", index=False)
    matrix.to_csv(output_dir / "roc_auc_matrix.csv")
    prediction_table.to_csv(output_dir / "grid_oof_predictions.csv", index=False)
    comparisons.to_csv(output_dir / "predefined_comparisons.csv", index=False)
    (output_dir / "grid_fold_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "dataset_signature": bundle.signature,
        "n_answers": n,
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "fold_sizes": {str(k): int(v) for k, v in zip(*np.unique(folds, return_counts=True))},
        "n_methods": len(predictions),
        "selection_warning": "Лучшая строка выбрана на тех же 100 ответах и требует новой выборки.",
        "labels_warning": "Метки ответа получены GPT-4o; выборка искусственно сбалансирована 50/50.",
        "configuration": {"tabular_variants": list(tabular_variants),
                          "tabular_models": list(tabular_models),
                          "set_variants": list(set_variants),
                          "set_models": list(set_models),
                          "neural_epochs": neural_epochs},
    }
    (output_dir / "grid_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics, matrix, prediction_table, comparisons
