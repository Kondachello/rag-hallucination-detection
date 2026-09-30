"""Проверяемый отчёт и единый архив расширенного опыта."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

from embeddings import ENCODERS, MODEL_ID, MODEL_REVISION


RESULT_FILES = (
    "grid_metrics.csv", "roc_auc_matrix.csv", "grid_oof_predictions.csv",
    "predefined_comparisons.csv", "grid_fold_details.json", "grid_audit.json",
    "embeddings_manifest.json", "embedding_trace.jsonl", "embeddings.npz",
)
ENCODER_FILES = (
    "encoder_metrics.csv", "encoder_oof_predictions.csv", "encoder_comparisons.csv",
    "encoder_fold_details.json", "encoder_audit.json",
)
BASELINE_FILES = (
    "baseline_gte/embeddings_manifest.json", "baseline_gte/embedding_trace.jsonl",
    "baseline_gte/embeddings.npz",
)
DATA_FILES = (
    "inputs.no_gold.jsonl", "components.no_gold.jsonl", "labels.csv",
    "folds.csv", "confirmation_features.no_gold.csv",
    "hallugraph_features.no_gold.csv", "manifest.json",
)


def _table(frame: pd.DataFrame, columns: list[str] | None = None, n: int | None = None) -> str:
    if columns is not None:
        frame = frame[columns]
    if n is not None:
        frame = frame.head(n)
    frame = frame.copy()
    for name in frame.select_dtypes(include="number"):
        if pd.api.types.is_float_dtype(frame[name]):
            frame[name] = frame[name].map(lambda value: f"{value:.4f}" if pd.notna(value) else "—")
    headers = "| " + " | ".join(frame.columns) + " |"
    ruler = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ")
                         for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([headers, ruler, *rows])


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_audit_package(output_dir: Path, *, source_root: Path | None = None) -> Path:
    """Проверяет полный прогон, пишет два отчёта и упаковывает все исходные результаты."""
    output_dir = Path(output_dir)
    source_root = Path(source_root) if source_root else Path(__file__).resolve().parent
    missing = [name for name in RESULT_FILES if not (output_dir / name).is_file()]
    missing += [name for name in ENCODER_FILES if not (output_dir / name).is_file()]
    missing += [name for name in BASELINE_FILES if not (output_dir / name).is_file()]
    missing += [f"data/{name}" for name in DATA_FILES if not (source_root / "data" / name).is_file()]
    if missing:
        raise FileNotFoundError("Неполный опыт, нельзя создать архив: " + ", ".join(missing))

    metrics = pd.read_csv(output_dir / "grid_metrics.csv")
    predictions = pd.read_csv(output_dir / "grid_oof_predictions.csv")
    comparisons = pd.read_csv(output_dir / "predefined_comparisons.csv")
    encoder_metrics = pd.read_csv(output_dir / "encoder_metrics.csv")
    encoder_predictions = pd.read_csv(output_dir / "encoder_oof_predictions.csv")
    encoder_comparisons = pd.read_csv(output_dir / "encoder_comparisons.csv")
    audit = json.loads((output_dir / "grid_audit.json").read_text(encoding="utf-8"))
    emb_meta = json.loads((output_dir / "embeddings_manifest.json").read_text(encoding="utf-8"))
    gte_meta = json.loads((output_dir / "baseline_gte" / "embeddings_manifest.json").read_text(encoding="utf-8"))
    if emb_meta.get("data_signature") != audit["dataset_signature"]:
        raise ValueError("Векторы и обучение относятся к разным версиям данных")
    if emb_meta.get("model_id") != MODEL_ID or emb_meta.get("revision") != MODEL_REVISION:
        raise ValueError("Модель векторов не совпадает с закреплённой версией")
    gte_profile = ENCODERS["gte_large"]
    if (gte_meta.get("data_signature") != audit["dataset_signature"]
            or gte_meta.get("model_id") != gte_profile.model_id
            or gte_meta.get("revision") != gte_profile.revision):
        raise ValueError("Контрольные GTE-векторы не совпадают с данными или закреплённой моделью")
    if (gte_meta.get("component_order") != emb_meta.get("component_order")
            or gte_meta.get("response_order") != emb_meta.get("response_order")):
        raise ValueError("Порядок ответов или компонентов различается между кодировщиками")
    expected_methods = int(audit["n_methods"])
    expected_answers = int(audit["n_answers"])
    if len(metrics) != expected_methods or metrics.method.nunique() != expected_methods:
        raise ValueError("Число методов в таблице не совпадает с журналом опыта")
    if len(predictions) != expected_methods * expected_answers:
        raise ValueError("Неполная таблица внеобучающих прогнозов")
    if predictions.duplicated(["method", "source_id"]).any():
        raise ValueError("Повторный прогноз для одного ответа и метода")
    counts = predictions.groupby("method").source_id.nunique()
    if not (counts == expected_answers).all():
        raise ValueError("Не все методы покрывают все ответы")
    if not np.isfinite(predictions.probability.to_numpy(dtype=float)).all():
        raise ValueError("Прогнозы содержат неконечные числа")
    if not predictions.probability.between(0, 1).all():
        raise ValueError("Прогнозы вне интервала [0, 1]")
    if not predictions.groupby("source_id").gold.nunique().eq(1).all():
        raise ValueError("Метка одного ответа различается между методами")
    expected_labels = pd.read_csv(source_root / "data" / "labels.csv").set_index("source_id")["hallucination"]
    actual_labels = predictions.drop_duplicates("source_id").set_index("source_id")["gold"]
    if (set(expected_labels.index) != set(actual_labels.index)
            or not np.array_equal(expected_labels.sort_index().to_numpy(),
                                  actual_labels.sort_index().to_numpy())):
        raise ValueError("Метки прогнозов не совпадают с приложенными данными")
    for vector_path, vector_meta in (
        (output_dir / "embeddings.npz", emb_meta),
        (output_dir / "baseline_gte" / "embeddings.npz", gte_meta),
    ):
        with np.load(vector_path) as vectors:
            dimension = int(vector_meta["embedding_dim"])
            for name, count in (("context", len(vector_meta["component_order"])),
                                ("triple", len(vector_meta["component_order"])),
                                ("answer_cls", expected_answers)):
                array = vectors[name]
                if array.shape != (count, dimension) or not np.isfinite(array).all():
                    raise ValueError(f"Неверный массив векторов {vector_path.name}: {name}")
    for row in metrics.to_dict("records"):
        group = predictions[predictions.method == row["method"]]
        y = group.gold.to_numpy(dtype=int)
        p = group.probability.to_numpy(dtype=float)
        recalculated = {"ROC_AUC": roc_auc_score(y, p),
                        "PR_AUC": average_precision_score(y, p),
                        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
                        "F1_at_0.5": f1_score(y, p >= 0.5)}
        for metric, value in recalculated.items():
            if not np.isclose(row[metric], value, rtol=0, atol=1e-6):
                raise ValueError(f"Метрика {metric} метода {row['method']} не совпадает с прогнозами")
    if encoder_predictions.duplicated(["method", "source_id"]).any():
        raise ValueError("Повторный прогноз в сравнении кодировщиков")
    if not encoder_predictions.groupby("method").source_id.nunique().eq(expected_answers).all():
        raise ValueError("Неполные прогнозы в сравнении кодировщиков")
    for row in encoder_metrics.to_dict("records"):
        group = encoder_predictions[encoder_predictions.method == row["method"]]
        y = group.gold.to_numpy(dtype=int)
        p = group.probability.to_numpy(dtype=float)
        if not np.isclose(row["ROC_AUC"], roc_auc_score(y, p), rtol=0, atol=1e-6):
            raise ValueError(f"ROC-AUC кодировщика не совпал: {row['method']}")

    fold_rows = []
    for (method, fold), group in predictions.groupby(["method", "fold"], sort=True):
        y = group.gold.to_numpy(dtype=int)
        p = group.probability.to_numpy(dtype=float)
        fold_rows.append({"method": method, "fold": int(fold), "n": len(group),
                          "ROC_AUC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
                          "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])})
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    top = metrics.sort_values("ROC_AUC", ascending=False)
    baselines = metrics[metrics.method.isin(["D_CONFIRMATION|logreg_l2",
                                            "D_CONFIRMATION|torch_mlp",
                                            "BASELINE|hallugraph_raw",
                                            "BASELINE|frequency"])].sort_values("method")
    best = str(top.iloc[0].method)
    hard = predictions[predictions.method == best].copy()
    hard["error"] = np.where(hard.gold == 1, 1 - hard.probability, hard.probability)
    hard = hard.nlargest(10, "error")
    chosen = [best, "D_CONFIRMATION|logreg_l2", "BASELINE|hallugraph_raw"]
    fold_summary = fold_metrics[fold_metrics.method.isin(chosen)].sort_values(["method", "fold"])
    commit = _git_commit(source_root)
    run_manifest = {
        "schema": "component-grid-audit-v2", "git_commit": commit,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "encoders": {"primary": emb_meta, "baseline": gte_meta},
        "python": sys.version.split()[0], "n_answers": expected_answers,
        "n_methods": expected_methods, "dataset_signature": audit["dataset_signature"],
        "embeddings_manifest": emb_meta,
        "files": {},
    }

    sections = [
        "# Отчёт по опыту 2.2: компоненты и детекция галлюцинаций",
        f"**Ответов:** {expected_answers}; **методов:** {expected_methods}; "
        f"**модель векторов:** `{MODEL_ID}` (`{MODEL_REVISION}`); "
        f"**коммит кода:** `{commit or 'не определён'}`.",
        "Главная метрика — ROC-AUC (площадь под кривой различения классов). "
        "Каждому ответу присвоен прогноз моделью, которая не обучалась на метке этого ответа.",
        "## Основная проверка гипотезы и сравнение кодировщиков",
        _table(encoder_comparisons),
        "Строка `primary` сравнивает одну и ту же логистическую модель на 13 признаках "
        "с той же моделью после добавления Qwen3-векторов. GTE-large использует те же "
        "тексты компонентов. `negative_control` сравнивает настоящие Qwen3-векторы с "
        "векторами, случайно переставленными между ответами.",
        _table(encoder_metrics, ["method", "description", "ROC_AUC", "PR_AUC",
                                 "log_loss", "F1_at_0.5", "coverage"]),
        "Все остальные 139 сочетаний ниже являются разведочным перебором.",
        "## Лучшие 20 методов по ROC-AUC",
        _table(top, ["method", "data_description", "model_description", "ROC_AUC",
                     "PR_AUC", "log_loss", "F1_at_0.5", "coverage"], 20),
        "## Контрольные методы",
        _table(baselines, ["method", "ROC_AUC", "PR_AUC", "log_loss", "F1_at_0.5", "coverage"]),
        "Риск HalluGraph не является калиброванной вероятностью; его "
        "логарифмическую ошибку нельзя интерпретировать как качество калибровки.",
        "## Заранее выбранные парные сравнения",
        _table(comparisons) if not comparisons.empty else "В сокращённом проверочном прогоне сравнений нет.",
        "Интервалы выше разведочные. Выбор лучшего среди множества методов по той же "
        "сотне ответов смещает оценку его качества вверх.",
        "## Результаты по каждой проверочной части",
        _table(fold_summary, ["method", "fold", "n", "ROC_AUC", "log_loss"]),
        "Полные показатели каждого метода в каждой части находятся в `fold_metrics.csv`.",
        "## 10 наиболее уверенных ошибок лучшего метода",
        _table(hard, ["source_id", "response_id", "fold", "gold", "probability", "error"]),
        "`gold=1` означает галлюцинацию. `probability` — прогноз вероятности этого класса; "
        "`error` показывает величину ошибки относительно истинной метки.",
        "## Как повторить и проверить выводы",
        "- `grid_metrics.csv` — все методы и итоговые метрики; `roc_auc_matrix.csv` — матрица.",
        "- `grid_oof_predictions.csv` — прогноз каждого метода для каждого ответа; "
        "по нему можно пересчитать все метрики.",
        "- `fold_metrics.csv` и `grid_fold_details.json` — разбивка и настройки по частям.",
        "- `predefined_comparisons.csv` — заранее выбранные парные разницы и интервалы.",
        "- `encoder_metrics.csv`, `encoder_oof_predictions.csv` и "
        "`encoder_comparisons.csv` — основная проверка, сравнение Qwen3/GTE и контроль "
        "с переставленными векторами.",
        "- `embeddings.npz`, `embedding_trace.jsonl`, `embeddings_manifest.json` — "
        "векторы, точные входы, их длины и сведения о модели.",
        "- `data/` — точная копия входного набора; `run_manifest.json` — версия кода, "
        "модели и контрольные суммы файлов.",
        "## Ограничения",
        "В выборке 100 искусственно сбалансированных ответов с целевыми метками GPT-4o. "
        "Это разведочный опыт без независимой внешней выборки. "
        "Небольшой выигрыш или лучшая клетка матрицы не доказывают улучшение на новых данных.",
    ]
    report = "\n\n".join(sections) + "\n"
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    # HTML открывается локально без установленных библиотек и интернета.
    html_sections = [
        "<!doctype html><html lang='ru'><meta charset='utf-8'><title>Отчёт по опыту 2.2</title>",
        "<style>body{font:16px system-ui;max-width:1180px;margin:2rem auto;padding:0 1rem;"
        "line-height:1.5;color:#17212b}table{border-collapse:collapse;display:block;overflow:auto}"
        "th,td{padding:.35rem .6rem;border:1px solid #ccd4dd;text-align:left}"
        "th{background:#eaf1f8}tr:nth-child(even){background:#f8fafc}code{background:#eef2f6}</style>",
        "<h1>Отчёт по опыту 2.2</h1>",
        f"<p>Ответов: {expected_answers}; методов: {expected_methods}; "
        f"модель: <code>{html.escape(MODEL_ID)}</code>; "
        f"коммит: <code>{html.escape(commit or 'не определён')}</code>.</p>",
    ]
    for title, frame in [
        ("Основные сравнения", encoder_comparisons),
        ("Метрики кодировщиков", encoder_metrics),
        ("Лучшие 20 методов", top.head(20)), ("Контрольные методы", baselines),
        ("Заранее выбранные сравнения", comparisons),
        ("Метрики по частям", fold_summary), ("Уверенные ошибки", hard),
    ]:
        html_sections.append(f"<h2>{html.escape(title)}</h2>")
        html_sections.append(frame.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}"))
    html_sections.append("<p><strong>Ограничение:</strong> 100 ответов, метки GPT-4o, "
                         "много проверенных гипотез, независимой проверки нет. "
                         "Полные таблицы и исходные прогнозы лежат рядом в архиве.</p></html>")
    (output_dir / "REPORT.html").write_text("\n".join(html_sections), encoding="utf-8")

    paths = {name: output_dir / name for name in (*RESULT_FILES, *ENCODER_FILES,
                                                   *BASELINE_FILES,
                                                   "fold_metrics.csv", "REPORT.md", "REPORT.html")}
    paths.update({f"data/{name}": source_root / "data" / name for name in DATA_FILES})
    for name, path in paths.items():
        run_manifest["files"][name] = {"bytes": path.stat().st_size,
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_path = output_dir / "component_qwen3_4b_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for name, path in (*paths.items(), ("run_manifest.json", manifest_path)):
            archive.write(path, arcname=name)
    with zipfile.ZipFile(archive_path) as archive:
        damaged = archive.testzip()
        if damaged:
            raise IOError(f"Повреждён файл в архиве: {damaged}")
        if set(archive.namelist()) != set(paths) | {"run_manifest.json"}:
            raise IOError("Состав архива не совпал с манифестом")
    return archive_path
