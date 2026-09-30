"""Сокращение векторов и признаки наборов, подгоняемые только на обучении."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_io import Bundle, STATUSES, TYPES

DIM = 8
STATUS_WIDTH = 5


@dataclass
class FoldView:
    ids: list[int]
    q: np.ndarray
    # тип -> для каждого ответа массив [количество компонентов, 8]
    x: dict[str, list[np.ndarray]]
    status: dict[str, list[np.ndarray]]
    cls: np.ndarray
    missing_span: np.ndarray


def make_fold_view(bundle: Bundle, embeddings: dict, train_indices: np.ndarray,
                   *, source: str = "context") -> FoldView:
    if source not in ("context", "triple"):
        raise ValueError(source)
    ids = bundle.ids
    train_ids = {ids[int(i)] for i in train_indices}
    raw = np.asarray(embeddings[source], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != len(bundle.components) or raw.shape[1] < DIM:
        raise ValueError(f"Неверная форма {source}: {raw.shape}")
    q_scaler = StandardScaler().fit(bundle.q[train_indices])
    q = q_scaler.transform(bundle.q).astype(np.float32)
    grouped: dict[str, list[list[int]]] = {
        t: [[] for _ in ids] for t in TYPES
    }
    id_to_row = {sid: i for i, sid in enumerate(ids)}
    for j, comp in enumerate(bundle.components):
        grouped[comp["component_type"]][id_to_row[int(comp["source_id"])]].append(j)
    x, status = {}, {}
    for kind in TYPES:
        positions = [j for j, row in enumerate(bundle.components)
                     if row["component_type"] == kind]
        fit_positions = [j for j in positions
                         if int(bundle.components[j]["source_id"]) in train_ids]
        pca = PCA(n_components=DIM, svd_solver="randomized", random_state=42).fit(raw[fit_positions])
        reduced = pca.transform(raw[positions]).astype(np.float32)
        values = {j: reduced[k] for k, j in enumerate(positions)}
        x[kind] = [np.stack([values[j] for j in group]).astype(np.float32)
                   if group else np.zeros((0, DIM), dtype=np.float32)
                   for group in grouped[kind]]
        choices = STATUSES[kind]
        status[kind] = [np.asarray([choices.index(bundle.components[j]["confirmation"])
                                    for j in group], dtype=np.int64)
                        for group in grouped[kind]]
    cls_raw = np.asarray(embeddings["answer_cls"], dtype=np.float32)
    if cls_raw.shape != (len(ids), raw.shape[1]):
        raise ValueError(f"Неверная форма answer_cls: {cls_raw.shape}")
    cls = PCA(n_components=DIM, svd_solver="randomized", random_state=42).fit(cls_raw[train_indices]).transform(cls_raw)
    missing_span = np.asarray([
        sum(row["component_type"] == "entity" and row["answer_start"] is None
            for row in bundle.components if int(row["source_id"]) == sid)
        for sid in ids
    ], dtype=np.float32)[:, None]
    return FoldView(ids, q, x, status, cls.astype(np.float32), missing_span)


def _pool(arr: np.ndarray, op: str) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros(DIM, dtype=np.float32)
    if op == "mean":
        return arr.mean(axis=0)
    if op == "max":
        return arr.max(axis=0)
    if op == "std":
        return arr.std(axis=0)
    raise ValueError(op)


def _matrix(view: FoldView, op: str) -> np.ndarray:
    return np.concatenate([
        np.stack([_pool(arr, op) for arr in view.x[kind]]) for kind in TYPES
    ], axis=1)


def _missing(view: FoldView) -> np.ndarray:
    return np.stack([[float(len(view.x[k][i]) == 0) for k in ("relation", "claim")]
                     for i in range(len(view.ids))], dtype=np.float32)


def _group(value: int, kind: str) -> int:
    label = STATUSES[kind][value]
    if label in ("grounded", "entailed"):
        return 0
    if label in ("ungrounded", "unsupported", "contradicted"):
        return 1
    return 2


def feature_matrix(view: FoldView, method: str) -> np.ndarray:
    mean, missing = _matrix(view, "mean"), _missing(view)
    if method == "A":
        return view.q
    if method == "B":
        return np.column_stack([mean, missing, view.missing_span])
    if method == "M0":
        return np.column_stack([view.q, mean, view.missing_span])
    if method == "B_max":
        return np.column_stack([_matrix(view, "max"), missing, view.missing_span])
    if method == "M6":
        return np.column_stack([view.q, mean, _matrix(view, "max"),
                                _matrix(view, "std"), view.missing_span])
    if method == "M7":
        chunks = [view.q, mean, view.missing_span]
        for kind in TYPES:
            groups = []
            for arrays, codes in zip(view.x[kind], view.status[kind]):
                problem = arrays[[_group(int(c), kind) == 1 for c in codes]]
                groups.append(np.r_[_pool(problem, "mean"), float(len(problem) == 0)])
            chunks.append(np.stack(groups))
        return np.column_stack(chunks)
    if method == "M10":
        chunks = [view.q, view.missing_span]
        for kind in TYPES:
            values = []
            for arr, codes in zip(view.x[kind], view.status[kind]):
                if len(arr):
                    weights = np.asarray([1.0 if _group(int(c), kind) == 0
                                          else 3.0 if _group(int(c), kind) == 1
                                          else 1.5 for c in codes])
                    values.append(np.average(arr, axis=0, weights=weights))
                else:
                    values.append(np.zeros(DIM))
            chunks.append(np.stack(values))
        return np.column_stack(chunks)
    if method == "M13":
        mean_parts = [np.stack([_pool(a, "mean") for a in view.x[k]]) for k in TYPES]
        cosines = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            a, b = mean_parts[left], mean_parts[right]
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            cosines.append(np.divide((a * b).sum(axis=1), denom,
                                     out=np.zeros(len(view.ids)), where=denom > 0))
        distances = []
        for k, kind in enumerate(TYPES):
            d = []
            for i, (arr, codes) in enumerate(zip(view.x[kind], view.status[kind])):
                problem = arr[[_group(int(c), kind) == 1 for c in codes]]
                d.append(float(np.linalg.norm(problem - mean_parts[k][i], axis=1).mean())
                         if len(problem) else 0.0)
            distances.append(np.asarray(d))
        return np.column_stack([view.q, mean, view.missing_span, *cosines, *distances])
    if method == "M14_q_cls":
        return np.column_stack([view.q, view.cls])
    if method == "M14_all":
        return np.column_stack([view.q, mean, view.cls, view.missing_span])
    if method in ("E_only", "R_only", "C_only", "A+E", "A+R", "A+C"):
        kind = {"E": "entity", "R": "relation", "C": "claim"}[method[-1] if "+" in method else method[0]]
        single = np.stack([_pool(a, "mean") for a in view.x[kind]])
        extra = np.asarray([float(len(a) == 0) for a in view.x[kind]])[:, None]
        return np.column_stack([view.q, single, extra]) if method.startswith("A+") else np.column_stack([single, extra])
    raise ValueError(f"Неизвестный метод: {method}")


def padded_tensors(view: FoldView, indices: np.ndarray, device: str = "cpu"):
    """Фиксированные тензоры для сети; маски отделяют пустоту от нулевого вектора."""
    import torch
    xs, masks, statuses, groups = [], [], [], []
    for kind in TYPES:
        width = max(1, max(len(view.x[kind][int(i)]) for i in indices))
        arr = np.zeros((len(indices), width, DIM), dtype=np.float32)
        mask = np.zeros((len(indices), width), dtype=bool)
        onehot = np.zeros((len(indices), width, STATUS_WIDTH), dtype=np.float32)
        group = np.zeros((len(indices), width, 3), dtype=np.float32)
        for pos, index in enumerate(indices):
            values, codes = view.x[kind][int(index)], view.status[kind][int(index)]
            n = len(values)
            arr[pos, :n] = values
            mask[pos, :n] = True
            for j, code in enumerate(codes):
                onehot[pos, j, int(code)] = 1.0
                group[pos, j, _group(int(code), kind)] = 1.0
        xs.append(torch.tensor(arr, device=device))
        masks.append(torch.tensor(mask, device=device))
        statuses.append(torch.tensor(onehot, device=device))
        groups.append(torch.tensor(group, device=device))
    q = torch.tensor(view.q[indices], dtype=torch.float32, device=device)
    return q, xs, masks, statuses, groups
