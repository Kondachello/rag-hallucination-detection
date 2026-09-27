"""Быстрая проверка второй версии без полного пятичастного опыта."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_bundle
from grid_features import DATA_VARIANTS, build_tabular_features
from grid_models import (SET_DATA_VARIANTS, SET_MODELS, TABULAR_MODELS,
                         fit_predict_confirmation_mlp, fit_predict_set,
                         make_tabular_model)


def run_smoke(bundle, embeddings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = np.where(bundle.fold_numbers != 0)[0]
    test_idx = np.where(bundle.fold_numbers == 0)[0]
    shapes = []
    for variant in DATA_VARIANTS:
        matrix = build_tabular_features(bundle, embeddings, train_idx, variant)
        shapes.append({"data_variant": variant, "rows": matrix.shape[0],
                       "columns": matrix.shape[1], "finite": bool(np.isfinite(matrix).all())})

    # Проверяем интерфейс каждого табличного семейства на компактных 13 признаках.
    q = build_tabular_features(bundle, embeddings, train_idx, "D_CONFIRMATION")
    model_rows = []
    for name in TABULAR_MODELS:
        model = make_tabular_model(name, seed=42)
        model.fit(q[train_idx], bundle.y[train_idx])
        p = model.predict_proba(q[test_idx])[:, 1]
        model_rows.append({"data_variant": "D_CONFIRMATION", "model": name,
                           "n_predictions": len(p), "finite": bool(np.isfinite(p).all())})

    p, detail = fit_predict_confirmation_mlp(
        bundle, train_idx, test_idx, seed=42, epochs=1)
    model_rows.append({"data_variant": "D_CONFIRMATION", "model": "torch_mlp",
                       "n_predictions": len(p), "finite": bool(np.isfinite(p).all()), **detail})
    for data_variant in SET_DATA_VARIANTS:
        for model_name in SET_MODELS:
            p, detail = fit_predict_set(bundle, embeddings, data_variant, model_name,
                                        train_idx, test_idx, seed=42, epochs=1)
            model_rows.append({"data_variant": data_variant, "model": model_name,
                               "n_predictions": len(p),
                               "finite": bool(np.isfinite(p).all()), **detail})
    shapes = pd.DataFrame(shapes)
    models = pd.DataFrame(model_rows)
    if not shapes.finite.all() or not models.finite.all():
        raise AssertionError("Проверка обнаружила неконечные числа")
    return shapes, models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Путь к embeddings.npz из первого запуска")
    args = parser.parse_args()
    bundle = load_bundle()
    with np.load(args.embeddings) as archive:
        embeddings = {name: archive[name] for name in archive.files}
    shapes, models = run_smoke(bundle, embeddings)
    print(shapes.to_string(index=False))
    print(models.to_string(index=False))
    print(f"OK: {len(shapes)} представлений, {len(models)} проверок моделей")


if __name__ == "__main__":
    main()
