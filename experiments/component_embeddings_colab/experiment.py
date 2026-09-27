"""Пять заранее заданных частей, внутренний подбор и единая таблица методов."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_io import Bundle, ROOT
from features import feature_matrix, make_fold_view
from models import NEURAL_METHODS, fit_predict_neural

DESCRIPTIONS = {
    "P0": "Частота галлюцинаций в обучении",
    "HG_raw": "Строгий риск HalluGraph, α=0.7",
    "HG_cal": "Тот же риск с калибровкой на обучении",
    "A": "13 агрегированных оценок подтверждённости",
    "B": "Средние векторы E/R/C без подтверждённости",
    "M0": "Подтверждённость + средние E/R/C; линейная модель",
    "B_max": "Покомпонентный максимум E/R/C без подтверждённости",
    "B_triples": "Средние E/R/C; сущности и отношения только из троек",
    "M0_triples": "A + средние из троек вместо контекста ответа",
    "M1": "Раздельный слой после среднего каждого типа",
    "M2": "Общий слой компонентов с кодом типа, затем среднее",
    "M3": "Раздельный слой до среднего компонентов",
    "M4": "Раздельный слой до максимума компонентов",
    "M5": "Раздельный слой, затем среднее и максимум",
    "M6": "Среднее, максимум и разброс; линейная модель",
    "M7": "Средние проблемных компонентов и признаки отсутствия",
    "M8": "Слой до объединения по благоприятным, проблемным и неизвестным",
    "M9": "Общий слой на векторе, исходе проверки и типе",
    "M10": "Заданные веса исходов проверки при усреднении",
    "M11": "Обучаемые веса внимания для компонентов каждого типа",
    "M12": "Риск хотя бы одного проблемного отношения/утверждения",
    "M13": "Средние E/R/C + взаимные сходства и расстояния",
    "M14_q_cls": "Подтверждённость + общий вектор ответа",
    "M14_all": "Подтверждённость + E/R/C + общий вектор ответа",
    "E_only": "Только среднее сущностей",
    "R_only": "Только среднее отношений",
    "C_only": "Только среднее утверждений",
    "A+E": "Подтверждённость + сущности",
    "A+R": "Подтверждённость + отношения",
    "A+C": "Подтверждённость + утверждения",
}
LINEAR_METHODS = tuple(k for k in DESCRIPTIONS if k not in ("P0", "HG_raw", "HG_cal")
                       and k not in NEURAL_METHODS)
ALL_METHODS = tuple(DESCRIPTIONS)
CS = (0.01, 0.1, 1.0, 10.0)


def _source(method: str) -> str:
    return "triple" if method in ("B_triples", "M0_triples") else "context"


def _feature_name(method: str) -> str:
    return {"B_triples": "B", "M0_triples": "M0"}.get(method, method)


def _linear(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear",
                                                               max_iter=1000, random_state=42))


def _select_c(bundle: Bundle, embeddings: dict, method: str,
              train_idx: np.ndarray, y: np.ndarray) -> float:
    splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    data = []
    for fit_local, val_local in splits.split(train_idx, y[train_idx]):
        fit_idx, val_idx = train_idx[fit_local], train_idx[val_local]
        view = make_fold_view(bundle, embeddings, fit_idx, source=_source(method))
        matrix = feature_matrix(view, _feature_name(method))
        data.append((matrix[fit_idx], y[fit_idx], matrix[val_idx], y[val_idx]))
    scores = []
    for c in CS:
        losses = []
        for x_fit, y_fit, x_val, y_val in data:
            p = _linear(c).fit(x_fit, y_fit).predict_proba(x_val)[:, 1]
            losses.append(log_loss(y_val, p, labels=[0, 1]))
        scores.append(float(np.mean(losses)))
    return CS[int(np.argmin(scores))]


def _metrics(y: np.ndarray, p: np.ndarray, *, raw_risk: bool = False) -> dict:
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Некорректные вероятности")
    return {
        "log_loss": np.nan if raw_risk else log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "AUROC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "F1_at_0.5": f1_score(y, p >= 0.5),
        "coverage": int(np.isfinite(p).sum()),
    }


def run_experiment(bundle: Bundle, embeddings: dict, *,
                   methods: tuple[str, ...] = ALL_METHODS,
                   output_dir: Path | None = None,
                   neural_max_epochs: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Обучает все малые модели только в Colab; возвращает внеобучающие прогнозы."""
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Неизвестные методы: {unknown}")
    y, folds = bundle.y, bundle.fold_numbers
    n = len(y)
    predictions = {method: np.full(n, np.nan) for method in methods}
    fold_details = []
    for fold in sorted(set(folds)):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        views = {}
        if any(m not in ("P0", "HG_raw", "HG_cal") for m in methods):
            views["context"] = make_fold_view(bundle, embeddings, train_idx)
        if any(_source(m) == "triple" for m in methods):
            views["triple"] = make_fold_view(bundle, embeddings, train_idx, source="triple")
        for method in methods:
            detail = {"fold": int(fold), "method": method}
            if method == "P0":
                p = np.full(len(test_idx), y[train_idx].mean())
            elif method == "HG_raw":
                p = bundle.hallugraph_risk[test_idx]
            elif method == "HG_cal":
                risk = bundle.hallugraph_risk.reshape(-1, 1)
                model = _linear(1.0).fit(risk[train_idx], y[train_idx])
                p = model.predict_proba(risk[test_idx])[:, 1]
            elif method in NEURAL_METHODS:
                p, neural_detail = fit_predict_neural(
                    method, views["context"], y, train_idx, test_idx, bundle, embeddings,
                    seed=42 + int(fold), max_epochs=neural_max_epochs)
                detail.update(neural_detail)
            else:
                c = _select_c(bundle, embeddings, method, train_idx, y)
                view = views[_source(method)]
                matrix = feature_matrix(view, _feature_name(method))
                model = _linear(c).fit(matrix[train_idx], y[train_idx])
                p = model.predict_proba(matrix[test_idx])[:, 1]
                detail["C"] = c
            predictions[method][test_idx] = p
            fold_details.append(detail)
        print(f"Часть {fold}: рассчитаны {len(methods)} методов, проверено {len(test_idx)} ответов")
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise AssertionError("Не для всех ответов есть внеобучающее предсказание")
    rows = []
    for method, p in predictions.items():
        rows.append({"method": method, "description": DESCRIPTIONS[method],
                     **_metrics(y, p, raw_risk=(method == "HG_raw"))})
    table = pd.DataFrame(rows)
    table["order"] = table.method.map({m: i for i, m in enumerate(ALL_METHODS)})
    table = table.sort_values("order").drop(columns="order").reset_index(drop=True)
    pred_table = pd.DataFrame({"source_id": bundle.ids, "response_id": [bundle.by_id[s]["response_id"]
                                     for s in bundle.ids], "fold": folds, "gold": y,
                               **predictions})
    comparison = {}
    if "M0" in predictions and "A" in predictions:
        pa, pb = np.clip(predictions["A"], 1e-7, 1 - 1e-7), np.clip(predictions["M0"], 1e-7, 1 - 1e-7)
        per_row_a = -(y * np.log(pa) + (1 - y) * np.log1p(-pa))
        per_row_b = -(y * np.log(pb) + (1 - y) * np.log1p(-pb))
        delta = per_row_b - per_row_a
        rng = np.random.default_rng(2026)
        samples = rng.integers(0, n, size=(5000, n))
        interval = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
        comparison = {"M0_minus_A_log_loss": float(delta.mean()),
                      "paired_bootstrap_95": interval.tolist(),
                      "practically_interesting_cutoff": -0.03,
                      "note": "Пилот на 100 ответах, интервал разведочный"}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "method_metrics.csv", index=False)
        pred_table.to_csv(output_dir / "oof_predictions.csv", index=False)
        (output_dir / "fold_details.json").write_text(
            json.dumps(fold_details, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    return table, pred_table, comparison
