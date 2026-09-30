"""Собирает Colab-ноутбук расширенной сетки без выполнения ячеек."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Эксперимент 2.1: матрица «данные × модель»

**Одна кнопка:** выберите GPU, затем «Среда выполнения → Выполнить все».

Ноутбук получает векторы замороженной GTE-large и проверяет 15 способов
превратить компоненты в один вектор ответа, восемь классических моделей,
четыре сети для переменного набора сырых компонентов и отдельную сеть только
на 13 признаках подтверждённости. Главная метрика — ROC-AUC, то есть площадь
под ROC-кривой.

Все варианты используют одинаковые пять внешних частей. Метка проверяемого
ответа не участвует ни в получении векторов, ни в построении его признаков.
    """),
    md(r"""
## Что показал первый запуск на BGE-small

Первый запуск технически корректен: 30 методов дали по 100 внеобучающих
прогнозов, векторы конечны и нормированы, а сохранённые метрики полностью
пересчитываются из прогнозов.

| Метод | ROC-AUC | Логарифмическая ошибка |
|---|---:|---:|
| Только 13 признаков (`A`) | 0.6324 | 0.7101 |
| 13 признаков + средние векторы, логистическая регрессия (`M0`) | 0.5724 | 0.7459 |
| Слой для каждого компонента + максимум (`M4`) | **0.6512** | **0.6573** |
| Общий слой с исходом проверки (`M9`) | 0.6500 | 0.6597 |
| HalluGraph | 0.5982 | — |

Улучшение `M4` относительно `A` небольшое, а архитектуры различаются. Поэтому
первый запуск не позволяет приписать улучшение именно векторам. Ниже добавлен
обязательный контроль: та же малая сеть только на 13 признаках.
    """),
    md(r"""
## 1. Подготовка Colab

Первая ячейка клонирует опубликованную ветку и устанавливает зависимости.
PyTorch уже входит в среду Colab.
    """),
    code(r"""
from pathlib import Path
import os, subprocess, sys

REPO_URL = "https://github.com/Kondachello/rag-hallucination-detection.git"
BRANCH = "span_by_kolya"
SUBDIR = Path("experiments/component_embeddings_colab")

if (Path.cwd() / "grid_experiment.py").is_file():
    EXP_ROOT = Path.cwd().resolve()
else:
    checkout = Path("/content/rag_hallucination_detection")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        REPO_URL, str(checkout)], check=True)
    EXP_ROOT = checkout / SUBDIR
if not (EXP_ROOT / "grid_experiment.py").is_file():
    raise FileNotFoundError(f"Не найдена новая версия опыта: {EXP_ROOT}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(EXP_ROOT / "requirements.txt")], check=True)
os.chdir(EXP_ROOT)
sys.path.insert(0, str(EXP_ROOT))
print("Папка опыта:", EXP_ROOT)
    """),
    code(r"""
import json
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown
from data_io import load_bundle

bundle = load_bundle()
print("Ответов:", len(bundle.ids), "компонентов:", len(bundle.components))
print("Классы:", dict(zip(*np.unique(bundle.y, return_counts=True))))
print("Размеры пяти частей:", dict(zip(*np.unique(bundle.fold_numbers, return_counts=True))))
print("GPU доступен:", torch.cuda.is_available())
    """),
    md(r"""
## 2. Получение исходных векторов

Используется закреплённая `Alibaba-NLP/gte-large-en-v1.5`: 409 млн параметров,
1024 координаты, предел 8192 токена. Для сущности берутся
токены её позиции в ответе; для отношения — токены предиката при совместной
подаче ответа и направленной тройки; утверждение кодируется отдельно.
Дополнительно строятся варианты только из троек и общий CLS-вектор ответа.
Модель GTE-large не дообучается.
    """),
    code(r"""
from embeddings import MODEL_ID, MODEL_REVISION, example_prompts, extract_embeddings

print("Кодировщик:", MODEL_ID, MODEL_REVISION)
display(pd.Series(example_prompts(bundle), name="Пример входа").to_frame())
ARTIFACTS = EXP_ROOT / "artifacts_grid_v2"
embeddings = extract_embeddings(bundle, output_dir=ARTIFACTS)
display(pd.DataFrame([
    {"массив": name, "форма": str(embeddings[name].shape),
     "конечные числа": bool(np.isfinite(embeddings[name]).all())}
    for name in ("context", "triple", "answer_cls")
]))
meta = embeddings["manifest"]
print("Точность:", meta["precision"], "размер пакета:", meta["batch_size"])
if meta["gpu_total_bytes"]:
    print("Пик зарезервированной GPU-памяти:",
          f"{meta['gpu_peak_reserved_bytes'] / 2**30:.2f} / {meta['gpu_total_bytes'] / 2**30:.2f} ГБ")
    """),
    md(r"""
## 3. Зафиксированные гипотезы

«Сырые» означает исходные 1024 координаты GTE-large без PCA. Для табличных моделей
переменное число компонентов превращается в средние, максимумы, квартили,
группы по подтверждённости или три наиболее удалённых от центра компонента.
Сети для наборов получают каждый 1024-мерный компонент отдельно и сами
обучают слой 1024→32 перед объединением.

Параметры моделей фиксированы в коде до просмотра новых результатов. Это
уменьшает риск подогнать решение под маленькую сотню ответов.
    """),
    code(r"""
from grid_features import DATA_VARIANTS
from grid_models import TABULAR_MODELS, SET_DATA_VARIANTS, SET_MODELS

display(pd.DataFrame(DATA_VARIANTS.items(), columns=["Вариант данных", "Описание"]))
display(pd.DataFrame(TABULAR_MODELS.items(), columns=["Табличная модель", "Описание"]))
display(pd.DataFrame(SET_DATA_VARIANTS.items(), columns=["Набор компонентов", "Описание"]))
display(pd.DataFrame(SET_MODELS.items(), columns=["Модель набора", "Описание"]))
    """),
    md(r"""
## 4. Быстрая техническая проверка

Сначала строятся все 15 представлений. Каждое классическое семейство обучается
на одной внешней обучающей части, а каждая сеть выполняет ровно одну эпоху.
Эти веса и прогнозы дальше не используются.
    """),
    code(r"""
from grid_smoke_test import run_smoke

shape_check, model_check = run_smoke(bundle, embeddings)
display(shape_check)
display(model_check[["data_variant", "model", "n_predictions", "finite"]])
assert shape_check.finite.all() and model_check.finite.all()
print("Техническая проверка пройдена")
    """),
    md(r"""
## 5. Полный пятичастный запуск

Будет рассчитано 139 сочетаний и контрольных вариантов. Для каждой клетки
строятся 100 прогнозов: по 20 ответов из каждой внешней части. Полный запуск
занимает заметно больше времени, чем первая версия, прежде всего из-за лесов
и сетей для сырых наборов.
    """),
    code(r"""
from grid_experiment import run_grid_experiment

RUN_FULL = True
NEURAL_EPOCHS = 80
if RUN_FULL:
    metrics, auc_matrix, predictions, comparisons = run_grid_experiment(
        bundle, embeddings, output_dir=ARTIFACTS, neural_epochs=NEURAL_EPOCHS)
    print("Готово. Методов:", len(metrics), "внеобучающих прогнозов:", len(predictions))
    """),
    md(r"""
## 6. Матрица ROC-AUC

Строка — способ представить данные, столбец — модель. Пустая клетка означает,
что сочетание неприменимо. Смотреть следует не только на максимум: при 100
ответах близкие значения легко меняются от конкретной выборки.
    """),
    code(r"""
if RUN_FULL:
    display(auc_matrix.style.format("{:.3f}", na_rep="—").background_gradient(
        cmap="RdYlGn", axis=None, vmin=0.4, vmax=max(0.7, float(np.nanmax(auc_matrix)))))
    display(metrics.head(25).style.format({"ROC_AUC": "{:.3f}", "PR_AUC": "{:.3f}",
                                           "log_loss": "{:.3f}", "F1_at_0.5": "{:.3f}"}))
    """),
    code(r"""
if RUN_FULL:
    import matplotlib.pyplot as plt
    top = metrics.head(25).sort_values("ROC_AUC")
    ax = top.plot.barh(x="method", y="ROC_AUC", legend=False, figsize=(11, 10),
                       color="#2563eb", title="25 лучших сочетаний по ROC-AUC")
    ax.axvline(0.6324, color="#dc2626", linestyle="--", label="A из первого запуска")
    ax.set_xlim(0.25, max(0.75, float(top.ROC_AUC.max()) + 0.03))
    ax.legend(); plt.tight_layout(); plt.show()
    """),
    md(r"""
## 7. Заранее выбранные сравнения

Ниже не перебирается «лучший против базового». Сравниваются пять заранее
записанных гипотез: влияние глубины без векторов, сырых средних, бустинга,
максимума по сырым компонентам и обучаемых весов с исходами проверки.
Интервалы разведочные и не заменяют независимую проверочную выборку.
    """),
    code(r"""
if RUN_FULL:
    display(comparisons.style.format({"ROC_AUC_difference": "{:+.3f}",
                                      "bootstrap_95_low": "{:+.3f}",
                                      "bootstrap_95_high": "{:+.3f}"}))
    """),
    md(r"""
## 8. Архив для следующего аудита

Последняя ячейка создаёт `component_grid_v2_results.zip` и начинает его
скачивание. Именно этот один ZIP нужно прислать для следующего аудита. В нём
есть метрики, вся матрица, 100 внеобучающих прогнозов каждого метода,
настройки по частям, доверительные интервалы и манифест векторов.
    """),
    code(r"""
if RUN_FULL:
    import zipfile
    result_names = [
        "grid_metrics.csv", "roc_auc_matrix.csv", "grid_oof_predictions.csv",
        "predefined_comparisons.csv", "grid_fold_details.json", "grid_audit.json",
        "embeddings_manifest.json",
    ]
    zip_path = ARTIFACTS / "component_grid_v2_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in result_names:
            path = ARTIFACTS / name
            if path.exists():
                archive.write(path, arcname=name)
    print("Архив для аудита:", zip_path, "размер:", zip_path.stat().st_size, "байт")
    try:
        from google.colab import files
        files.download(str(zip_path))
    except ImportError:
        from IPython.display import FileLink
        display(FileLink(str(zip_path)))
    """),
    md(r"""
### Ограничения вывода

Даже если одна клетка заметно выиграет, это остаётся разведочным результатом:
в наборе только 100 искусственно сбалансированных ответов, целевые метки
получены GPT-4o, а лучшая строка выбирается из большого числа гипотез на той
же сотне. Надёжное подтверждение потребует заранее выбранной модели и новой
размеченной выборки.
    """),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "run_grid_experiment.ipynb", "provenance": [],
              "gpuType": "T4", "accelerator": "GPU"},
}
nbf.validate(notebook)
nbf.write(notebook, ROOT / "run_grid_experiment.ipynb")
print(ROOT / "run_grid_experiment.ipynb")
