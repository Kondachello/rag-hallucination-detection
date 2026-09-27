"""Малые сети для наборов компонентов. Кодировщик BERT не обучается."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from features import DIM, FoldView, make_fold_view, padded_tensors

NEURAL_METHODS = ("M1", "M2", "M3", "M4", "M5", "M8", "M9", "M11", "M12")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values
    return torch.where(mask.any(dim=1, keepdim=True), values, torch.zeros_like(values))


class SetDetector(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        if method not in NEURAL_METHODS:
            raise ValueError(method)
        self.method = method
        self.type_layers = nn.ModuleList([nn.Linear(DIM, 4) for _ in range(3)])
        self.shared = nn.Linear(DIM + 3, 4)
        self.status_shared = nn.Linear(DIM + 5 + 3, 4)
        self.attention = nn.ModuleList([nn.Linear(DIM + 5, 1) for _ in range(3)])
        self.risk_heads = nn.ModuleList([nn.Linear(DIM + 5, 1) for _ in range(2)])
        size = {
            "M1": 25, "M2": 25, "M3": 25, "M4": 25,
            "M5": 37, "M8": 49, "M9": 37, "M11": 37, "M12": 23,
        }[method]
        self.final = nn.Linear(size, 1)

    def forward(self, batch) -> torch.Tensor:
        q, xs, masks, statuses, groups = batch
        pieces = []
        if self.method == "M1":
            for t in range(3):
                pieces.append(torch.relu(self.type_layers[t](masked_mean(xs[t], masks[t]))))
        elif self.method == "M2":
            for t in range(3):
                type_code = torch.zeros((*xs[t].shape[:2], 3), device=q.device)
                type_code[..., t] = 1
                h = torch.relu(self.shared(torch.cat([xs[t], type_code], dim=-1)))
                pieces.append(masked_mean(h, masks[t]))
        elif self.method in ("M3", "M4", "M5", "M8"):
            for t in range(3):
                h = torch.relu(self.type_layers[t](xs[t]))
                if self.method == "M8":
                    for g in range(3):
                        pieces.append(masked_mean(h, masks[t] & groups[t][..., g].bool()))
                elif self.method == "M4":
                    pieces.append(masked_max(h, masks[t]))
                elif self.method == "M5":
                    pieces.extend([masked_mean(h, masks[t]), masked_max(h, masks[t])])
                else:
                    pieces.append(masked_mean(h, masks[t]))
        elif self.method == "M9":
            for t in range(3):
                type_code = torch.zeros((*xs[t].shape[:2], 3), device=q.device)
                type_code[..., t] = 1
                h = torch.relu(self.status_shared(torch.cat([xs[t], statuses[t], type_code], -1)))
                pieces.extend([masked_mean(h, masks[t]), masked_max(h, masks[t])])
        elif self.method == "M11":
            for t in range(3):
                scores = self.attention[t](torch.cat([xs[t], statuses[t]], -1)).squeeze(-1)
                scores = scores.masked_fill(~masks[t], -1e9)
                weights = torch.softmax(scores, dim=1) * masks[t]
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
                pieces.append((xs[t] * weights.unsqueeze(-1)).sum(dim=1))
        elif self.method == "M12":
            pieces.append(masked_mean(xs[0], masks[0]))
            for t in (1, 2):
                raw = self.risk_heads[t - 1](torch.cat([xs[t], statuses[t]], -1)).squeeze(-1)
                p = torch.sigmoid(raw).clamp(1e-5, 1 - 1e-5)
                log_survival = (torch.log1p(-p) * masks[t]).sum(dim=1)
                pieces.append((-torch.expm1(log_survival)).unsqueeze(-1))
        return self.final(torch.cat([q, *pieces], dim=1)).squeeze(-1)


def _fit_steps(model: SetDetector, batch, y, epochs: int, *, learning_rate: float = 0.01):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    target = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(batch), target)
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def one_step_smoke(method: str, view: FoldView, indices: np.ndarray, y: np.ndarray) -> float:
    torch.manual_seed(42)
    model = SetDetector(method)
    batch = padded_tensors(view, indices)
    return _fit_steps(model, batch, y[indices], epochs=1)


def fit_predict_neural(method: str, view: FoldView, y: np.ndarray,
                       train_idx: np.ndarray, test_idx: np.ndarray,
                       bundle, embeddings: dict,
                       *, seed: int = 42, max_epochs: int = 120,
                       patience: int = 15) -> tuple[np.ndarray, dict]:
    """Выбор числа эпох по внутренней части, затем обучение заново на всех 80."""
    local_train, local_val = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                                          random_state=seed).split(train_idx, y[train_idx]))
    fit_idx, val_idx = train_idx[local_train], train_idx[local_val]
    inner_view = make_fold_view(bundle, embeddings, fit_idx)
    torch.manual_seed(seed)
    candidate = SetDetector(method)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.01, weight_decay=0.01)
    fit_batch = padded_tensors(inner_view, fit_idx)
    val_batch = padded_tensors(inner_view, val_idx)
    fit_y = torch.tensor(y[fit_idx], dtype=torch.float32)
    val_y = torch.tensor(y[val_idx], dtype=torch.float32)
    best, best_epoch, best_state = float("inf"), 1, None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        candidate.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(candidate(fit_batch), fit_y)
        loss.backward()
        optimizer.step()
        candidate.eval()
        with torch.inference_mode():
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(candidate(val_batch), val_y))
        if val_loss < best - 1e-5:
            best, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(candidate.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    torch.manual_seed(seed)
    final = SetDetector(method)
    train_batch = padded_tensors(view, train_idx)
    _fit_steps(final, train_batch, y[train_idx], best_epoch)
    final.eval()
    with torch.inference_mode():
        pred = torch.sigmoid(final(padded_tensors(view, test_idx))).cpu().numpy()
    return pred, {"best_epoch": best_epoch, "validation_log_loss": best}
