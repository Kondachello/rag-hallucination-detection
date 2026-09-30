"""Представления данных для сетки «данные × модель».

Все преобразования, которым нужно обучение (PCA), подгоняются только по
ответам внешней обучающей части. Сырые векторы кодировщика уже L2-нормированы.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_io import Bundle, STATUSES, TYPES
from features import DIM, make_fold_view


DATA_VARIANTS = {
    "D_CONFIRMATION": "Только 13 агрегированных признаков подтверждённости",
    "D_PCA_MEAN": "PCA-8: средние E/R/C без подтверждённости",
    "D_CONF_PCA_MEAN": "13 признаков + PCA-8 средние E/R/C",
    "D_CONF_PCA_STATS": "13 признаков + PCA-8 среднее, максимум, минимум и разброс",
    "D_CONF_PCA_STATUS": "13 признаков + PCA-8 средние по благоприятным/проблемным/неясным",
    "D_RAW_MEAN": "Сырые полноразмерные средние E/R/C без подтверждённости",
    "D_CONF_RAW_MEAN": "13 признаков + сырые средние E/R/C",
    "D_CONF_RAW_MEANMAX": "13 признаков + сырые средние и максимумы E/R/C",
    "D_CONF_RAW_QUANTILES": "13 признаков + сырые квартили 25/50/75% E/R/C",
    "D_CONF_RAW_STATUS": "13 признаков + сырые средние по трём группам подтверждённости",
    "D_CONF_RAW_TOP3": "13 признаков + три самых удалённых от центра компонента каждого типа",
    "D_CONF_CLS": "13 признаков + сырой общий вектор ответа CLS",
    "D_CONF_RAW_MEAN_CLS": "13 признаков + сырые средние E/R/C + CLS ответа",
    "D_CONF_TRIPLE_RAW_MEAN": "13 признаков + сырые средние из отдельных троек",
    "D_CONF_CONTEXT_TRIPLE": "13 признаков + средние из контекста, троек и их разность",
}


@dataclass
class RawView:
    ids: list[int]
    q: np.ndarray
    x: dict[str, list[np.ndarray]]
    status: dict[str, list[np.ndarray]]
    cls: np.ndarray
    missing_span: np.ndarray
    dimension: int


def make_raw_view(bundle: Bundle, embeddings: dict, *, source: str = "context") -> RawView:
    if source not in ("context", "triple"):
        raise ValueError(source)
    raw = np.asarray(embeddings[source], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != len(bundle.components):
        raise ValueError(f"Неверная форма {source}: {raw.shape}")
    dimension = int(raw.shape[1])
    id_to_row = {sid: i for i, sid in enumerate(bundle.ids)}
    grouped = {kind: [[] for _ in bundle.ids] for kind in TYPES}
    for j, component in enumerate(bundle.components):
        grouped[component["component_type"]][id_to_row[int(component["source_id"])]].append(j)
    x, status = {}, {}
    for kind in TYPES:
        x[kind] = [raw[idx].copy() if idx else np.zeros((0, dimension), np.float32)
                   for idx in grouped[kind]]
        choices = STATUSES[kind]
        status[kind] = [np.asarray([choices.index(bundle.components[j]["confirmation"])
                                    for j in idx], dtype=np.int64)
                        for idx in grouped[kind]]
    cls = np.asarray(embeddings["answer_cls"], dtype=np.float32)
    if cls.shape != (len(bundle.ids), dimension):
        raise ValueError(f"Неверная форма answer_cls: {cls.shape}")
    missing_span = np.asarray([
        sum(row["component_type"] == "entity" and row["answer_start"] is None
            for row in bundle.components if int(row["source_id"]) == sid)
        for sid in bundle.ids
    ], dtype=np.float32)[:, None]
    return RawView(bundle.ids, bundle.q.copy(), x, status, cls.copy(), missing_span,
                   dimension)


def confirmation_group(code: int, kind: str) -> int:
    label = STATUSES[kind][int(code)]
    if label in ("grounded", "entailed"):
        return 0
    if label in ("ungrounded", "unsupported", "contradicted"):
        return 1
    return 2


def _pool(arr: np.ndarray, op: str, width: int) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros(width, dtype=np.float32)
    if op == "mean":
        return arr.mean(0)
    if op == "max":
        return arr.max(0)
    if op == "min":
        return arr.min(0)
    if op == "std":
        return arr.std(0)
    raise ValueError(op)


def _matrix(x: dict[str, list[np.ndarray]], ops: tuple[str, ...], width: int) -> np.ndarray:
    chunks = []
    for kind in TYPES:
        for op in ops:
            chunks.append(np.stack([_pool(arr, op, width) for arr in x[kind]]))
    return np.column_stack(chunks).astype(np.float32)


def _counts(x: dict[str, list[np.ndarray]]) -> np.ndarray:
    return np.asarray([[np.log1p(len(x[kind][i])) for kind in TYPES]
                       for i in range(len(x[TYPES[0]]))], dtype=np.float32)


def _status_matrix(view, width: int) -> np.ndarray:
    chunks = []
    for kind in TYPES:
        for group in range(3):
            pooled, counts = [], []
            for arr, codes in zip(view.x[kind], view.status[kind]):
                mask = np.asarray([confirmation_group(c, kind) == group for c in codes], bool)
                selected = arr[mask]
                pooled.append(_pool(selected, "mean", width))
                counts.append(np.log1p(len(selected)))
            chunks.extend([np.stack(pooled), np.asarray(counts, np.float32)[:, None]])
    return np.column_stack(chunks).astype(np.float32)


def _quantiles(view: RawView) -> np.ndarray:
    chunks = []
    for kind in TYPES:
        rows = []
        for arr in view.x[kind]:
            rows.append(np.quantile(arr, (0.25, 0.5, 0.75), axis=0).reshape(-1)
                        if len(arr) else np.zeros(3 * view.dimension, np.float32))
        chunks.append(np.stack(rows))
    return np.column_stack(chunks).astype(np.float32)


def _top3_outliers(view: RawView) -> np.ndarray:
    chunks = []
    for kind in TYPES:
        rows = []
        for arr in view.x[kind]:
            selected = np.zeros((3, view.dimension), np.float32)
            mask = np.zeros(3, np.float32)
            if len(arr):
                distance = np.linalg.norm(arr - arr.mean(0, keepdims=True), axis=1)
                order = np.argsort(-distance, kind="stable")[:3]
                selected[:len(order)] = arr[order]
                mask[:len(order)] = 1
            rows.append(np.r_[selected.reshape(-1), mask])
        chunks.append(np.stack(rows))
    return np.column_stack(chunks).astype(np.float32)


def build_tabular_features(bundle: Bundle, embeddings: dict, train_idx: np.ndarray,
                           variant: str) -> np.ndarray:
    """Строит признаки всех 100 ответов, обучая PCA только на train_idx."""
    if variant not in DATA_VARIANTS:
        raise ValueError(f"Неизвестное представление: {variant}")
    q = bundle.q.astype(np.float32)
    raw = make_raw_view(bundle, embeddings, source="context")
    raw_dim = raw.dimension
    counts = _counts(raw.x)
    if variant == "D_CONFIRMATION":
        matrix = q
    elif variant.startswith("D_PCA") or variant.startswith("D_CONF_PCA"):
        pca = make_fold_view(bundle, embeddings, train_idx, source="context")
        if variant == "D_PCA_MEAN":
            matrix = np.column_stack([_matrix(pca.x, ("mean",), DIM), counts])
        elif variant == "D_CONF_PCA_MEAN":
            matrix = np.column_stack([q, _matrix(pca.x, ("mean",), DIM), raw.missing_span])
        elif variant == "D_CONF_PCA_STATS":
            matrix = np.column_stack([q, _matrix(pca.x, ("mean", "max", "min", "std"), DIM),
                                      counts, raw.missing_span])
        elif variant == "D_CONF_PCA_STATUS":
            matrix = np.column_stack([q, _status_matrix(pca, DIM), raw.missing_span])
        else:
            raise AssertionError(variant)
    elif variant == "D_RAW_MEAN":
        matrix = np.column_stack([_matrix(raw.x, ("mean",), raw_dim), counts])
    elif variant == "D_CONF_RAW_MEAN":
        matrix = np.column_stack([q, _matrix(raw.x, ("mean",), raw_dim), counts,
                                  raw.missing_span])
    elif variant == "D_CONF_RAW_MEANMAX":
        matrix = np.column_stack([q, _matrix(raw.x, ("mean", "max"), raw_dim), counts,
                                  raw.missing_span])
    elif variant == "D_CONF_RAW_QUANTILES":
        matrix = np.column_stack([q, _quantiles(raw), counts, raw.missing_span])
    elif variant == "D_CONF_RAW_STATUS":
        matrix = np.column_stack([q, _status_matrix(raw, raw_dim), raw.missing_span])
    elif variant == "D_CONF_RAW_TOP3":
        matrix = np.column_stack([q, _top3_outliers(raw), counts, raw.missing_span])
    elif variant == "D_CONF_CLS":
        matrix = np.column_stack([q, raw.cls])
    elif variant == "D_CONF_RAW_MEAN_CLS":
        matrix = np.column_stack([q, _matrix(raw.x, ("mean",), raw_dim), raw.cls,
                                  counts, raw.missing_span])
    elif variant == "D_CONF_TRIPLE_RAW_MEAN":
        triple = make_raw_view(bundle, embeddings, source="triple")
        matrix = np.column_stack([q, _matrix(triple.x, ("mean",), triple.dimension),
                                  _counts(triple.x), triple.missing_span])
    elif variant == "D_CONF_CONTEXT_TRIPLE":
        triple = make_raw_view(bundle, embeddings, source="triple")
        if triple.dimension != raw_dim:
            raise ValueError("Размерности контекстных и троечных векторов различаются")
        context_mean = _matrix(raw.x, ("mean",), raw_dim)
        triple_mean = _matrix(triple.x, ("mean",), raw_dim)
        matrix = np.column_stack([q, context_mean, triple_mean, context_mean - triple_mean,
                                  counts, raw.missing_span])
    else:
        raise AssertionError(variant)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] != len(bundle.ids) or not np.isfinite(matrix).all():
        raise AssertionError(f"Некорректные признаки {variant}: {matrix.shape}")
    return matrix
