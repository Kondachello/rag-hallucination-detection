"""Семейства моделей для расширенной сетки."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from data_io import TYPES
from grid_features import RawView, make_raw_view


TABULAR_MODELS = {
    "logreg_l2": "Логистическая регрессия L2",
    "logreg_elastic": "Логистическая регрессия с разреживанием Elastic Net",
    "random_forest": "Случайный лес",
    "extra_trees": "Крайне случайные деревья",
    "gradient_boosting": "Градиентный бустинг по гистограммам",
    "knn_cosine": "11 ближайших соседей по косинусному расстоянию",
    "gaussian_nb": "Наивный байесовский классификатор",
    "mlp_tabular": "Малая полносвязная сеть для готового вектора признаков",
}

SET_DATA_VARIANTS = {
    "D_SET_CONTEXT": "Сырые наборы E/R/C из контекста, без подтверждённости",
    "D_SET_CONTEXT_Q": "Сырые наборы E/R/C из контекста + 13 признаков",
    "D_SET_CONTEXT_Q_STATUS": "Сырые наборы + 13 признаков + исход каждого компонента",
    "D_SET_TRIPLE_Q_STATUS": "Сырые наборы из троек + 13 признаков + исходы",
}

SET_MODELS = {
    "set_mean": "Отдельный слой D→32, затем среднее",
    "set_max": "Отдельный слой D→32, затем максимум",
    "set_meanmax": "Отдельный слой D→32, затем среднее и максимум",
    "set_attention": "Обучаемые веса компонентов после слоя D→32",
}


def make_tabular_model(name: str, seed: int):
    if name == "logreg_l2":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, solver="liblinear", max_iter=2000, random_state=seed))
    if name == "logreg_elastic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, penalty="elasticnet", l1_ratio=0.5, solver="saga",
            max_iter=5000, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_features="sqrt", min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200, max_features="sqrt", min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=seed)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=7,
            min_samples_leaf=8, l2_regularization=5.0, random_state=seed)
    if name == "knn_cosine":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=11, weights="distance", metric="cosine", algorithm="brute"))
    if name == "gaussian_nb":
        return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-2))
    if name == "mlp_tabular":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="relu", alpha=0.1,
            learning_rate_init=0.003, max_iter=500, early_stopping=True,
            validation_fraction=0.2, n_iter_no_change=30, random_state=seed))
    raise ValueError(f"Неизвестная модель: {name}")


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(1).values
    return torch.where(mask.any(1, keepdim=True), values, torch.zeros_like(values))


class RawSetDetector(nn.Module):
    def __init__(self, mode: str, *, embedding_dim: int, use_q: bool, use_status: bool):
        super().__init__()
        if mode not in SET_MODELS:
            raise ValueError(mode)
        self.mode, self.use_q, self.use_status = mode, use_q, use_status
        input_width = embedding_dim + (5 if use_status else 0)
        self.projections = nn.ModuleList([nn.Linear(input_width, 32) for _ in TYPES])
        self.attention = nn.ModuleList([nn.Linear(32, 1) for _ in TYPES])
        pooled_width = 64 if mode == "set_meanmax" else 32
        self.final = nn.Sequential(
            nn.Linear((13 if use_q else 0) + len(TYPES) * pooled_width, 32),
            nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1))

    def forward(self, batch) -> torch.Tensor:
        q, xs, masks, statuses = batch
        pieces = []
        for t in range(len(TYPES)):
            inputs = torch.cat([xs[t], statuses[t]], -1) if self.use_status else xs[t]
            h = torch.relu(self.projections[t](inputs))
            if self.mode == "set_mean":
                pieces.append(_masked_mean(h, masks[t]))
            elif self.mode == "set_max":
                pieces.append(_masked_max(h, masks[t]))
            elif self.mode == "set_meanmax":
                pieces.extend([_masked_mean(h, masks[t]), _masked_max(h, masks[t])])
            else:
                score = self.attention[t](h).squeeze(-1).masked_fill(~masks[t], -1e9)
                weight = torch.softmax(score, 1) * masks[t]
                weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-9)
                pieces.append((h * weight.unsqueeze(-1)).sum(1))
        all_parts = ([q] if self.use_q else []) + pieces
        return self.final(torch.cat(all_parts, 1)).squeeze(-1)


class ConfirmationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(13, 16), nn.ReLU(), nn.Dropout(0.15),
                                    nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def _q_stats(q: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = q[train_idx].mean(0)
    scale = q[train_idx].std(0)
    return mean, np.where(scale > 1e-7, scale, 1.0)


def _set_batch(view: RawView, indices: np.ndarray, q_mean: np.ndarray,
               q_scale: np.ndarray, device: str):
    xs, masks, statuses = [], [], []
    for kind in TYPES:
        width = max(1, max(len(view.x[kind][int(i)]) for i in indices))
        values = np.zeros((len(indices), width, view.dimension), np.float32)
        mask = np.zeros((len(indices), width), bool)
        onehot = np.zeros((len(indices), width, 5), np.float32)
        for row, index in enumerate(indices):
            arr, codes = view.x[kind][int(index)], view.status[kind][int(index)]
            n = len(arr)
            values[row, :n], mask[row, :n] = arr, True
            for j, code in enumerate(codes):
                onehot[row, j, int(code)] = 1
        xs.append(torch.tensor(values, device=device))
        masks.append(torch.tensor(mask, device=device))
        statuses.append(torch.tensor(onehot, device=device))
    q = (view.q[indices] - q_mean) / q_scale
    return torch.tensor(q, dtype=torch.float32, device=device), xs, masks, statuses


def fit_predict_set(bundle, embeddings: dict, data_variant: str, model_name: str,
                    train_idx: np.ndarray, test_idx: np.ndarray, *, seed: int,
                    epochs: int = 80) -> tuple[np.ndarray, dict]:
    if data_variant not in SET_DATA_VARIANTS or model_name not in SET_MODELS:
        raise ValueError((data_variant, model_name))
    source = "triple" if data_variant == "D_SET_TRIPLE_Q_STATUS" else "context"
    use_q = data_variant != "D_SET_CONTEXT"
    use_status = data_variant.endswith("STATUS")
    view = make_raw_view(bundle, embeddings, source=source)
    q_mean, q_scale = _q_stats(view.q, train_idx)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RawSetDetector(model_name, embedding_dim=view.dimension,
                           use_q=use_q, use_status=use_status).to(device)
    train_batch = _set_batch(view, train_idx, q_mean, q_scale, device)
    target = torch.tensor(bundle.y[train_idx], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.03)
    model.train()
    last_loss = np.nan
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(train_batch), target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(_set_batch(view, test_idx, q_mean, q_scale, device)))
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return probability.cpu().numpy(), {"epochs": epochs, "parameters": parameters,
                                       "train_loss": last_loss, "device": device}


def fit_predict_confirmation_mlp(bundle, train_idx: np.ndarray, test_idx: np.ndarray,
                                 *, seed: int, epochs: int = 80) -> tuple[np.ndarray, dict]:
    q = bundle.q.astype(np.float32)
    mean, scale = _q_stats(q, train_idx)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.tensor((q[train_idx] - mean) / scale, device=device)
    x_test = torch.tensor((q[test_idx] - mean) / scale, device=device)
    y_train = torch.tensor(bundle.y[train_idx], dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    model = ConfirmationMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.03)
    model.train()
    last_loss = np.nan
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(x_test)).cpu().numpy()
    return probability, {"epochs": epochs,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "train_loss": last_loss, "device": device}
