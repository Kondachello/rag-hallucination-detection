"""Собирает основной Colab-ноутбук без выполнения тяжёлых ячеек."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


cells = [
    md(r"""
# Эксперимент 2.2: проверка вклада эмбеддингов компонентов

Выберите среду **T4 GPU**, затем нажмите **«Среда выполнения → Выполнить все»**.
Последняя ячейка проверит результаты, соберёт полный архив и запустит скачивание.

Основной кодировщик — закреплённый `Qwen/Qwen3-Embedding-4B`: 4,02 млрд
параметров, 2560 координат. Контрольный кодировщик — прежний GTE-large.
Оба получают одинаковые тексты сущностей, направленных отношений, утверждений
и ответов. Модели заморожены и не видят целевые метки.
    """),
    md(r"""
## Что показал предыдущий запуск GTE-large

Предыдущий архив технически цел: 139 методов, по 100 внеобучающих прогнозов.
Лучшим оказался градиентный бустинг только на 13 признаках подтверждённости:
ROC-AUC `0.7122`. Добавление PCA-средних GTE-векторов к тому же бустингу дало
`0.6864`; для логистической регрессии — `0.6488` против `0.6696` без векторов.
Сырые средние векторы ухудшили логистическую регрессию на `0.1832` ROC-AUC,
разведочный 95% интервал разницы `[-0.3224; -0.0424]`.

Поэтому новый запуск разделён на два уровня:

1. **Основная проверка:** одна логистическая модель на 13 признаках против той
   же модели с Qwen3-векторами. Регуляризация выбирается только внутри обучения.
2. **Разведочный перебор:** прежняя матрица способов агрегации и моделей.

Дополнительно сравниваются Qwen3 и GTE на одинаковых входах, отдельные типы
компонентов и контроль со случайно переставленными между ответами векторами.
    """),
    md("## 1. Подготовка среды"),
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
    else:
        subprocess.run(["git", "-C", str(checkout), "pull", "--ff-only", "origin", BRANCH],
                       check=True)
    EXP_ROOT = checkout / SUBDIR
if not (EXP_ROOT / "encoder_comparison.py").is_file():
    raise FileNotFoundError(f"Не найдена новая версия опыта: {EXP_ROOT}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(EXP_ROOT / "requirements.txt")], check=True)
os.chdir(EXP_ROOT)
sys.path.insert(0, str(EXP_ROOT))
print("Папка опыта:", EXP_ROOT)
    """),
    code(r"""
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown
from data_io import load_bundle

bundle = load_bundle()
print("Ответов:", len(bundle.ids), "компонентов:", len(bundle.components))
print("Классы:", dict(zip(*np.unique(bundle.y, return_counts=True))))
print("Размеры пяти частей:", dict(zip(*np.unique(bundle.fold_numbers, return_counts=True))))
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет")
if not torch.cuda.is_available():
    raise RuntimeError("Выберите T4 GPU: Среда выполнения → Сменить среду выполнения. Затем снова нажмите «Выполнить все».")
    """),
    md(r"""
## 2. Тексты, которые получают кодировщики

Опорные документы намеренно не добавляются в эти тексты. Иначе эмбеддинг
смешивал бы смысл компонента с его подтверждённостью, и мы не смогли бы
измерить пользу семантических векторов как отдельной группы признаков.

- сущность: ответ и явно указанная целевая сущность;
- отношение: ответ и полная направленная тройка «субъект — предикат — объект»;
- утверждение: отдельный текст утверждения;
- изолированный вариант: только сущность, тройка или утверждение;
- общий вектор: полный ответ.

Qwen3 использует вектор последнего токена по инструкции авторов модели;
GTE-large использует первый служебный токен. Все векторы нормируются.
    """),
    code(r"""
from embeddings import (BASELINE_ENCODER, ENCODERS, PRIMARY_ENCODER,
                        example_prompts, extract_embeddings)

display(pd.Series({k: v[:1200] for k, v in example_prompts(bundle).items()},
                  name="Начало точного входа").to_frame())
for key, profile in ENCODERS.items():
    print(key, profile.model_id, profile.revision, profile.dimension, "координат")
    """),
    md(r"""
## 3. Получение GTE-large и Qwen3-Embedding-4B

Сначала рассчитывается контрольный GTE-large, затем он выгружается из
видеопамяти и рассчитывается Qwen3. Размер пакета выбирается пробным проходом
на самых длинных входах. Код не допускает усечение и сохраняет пик памяти,
версию модели, хеш каждого текста и точный порядок векторов.
    """),
    code(r"""
ARTIFACTS = EXP_ROOT / "artifacts_qwen3_4b_v3"
gte_embeddings = extract_embeddings(
    bundle, encoder=BASELINE_ENCODER, output_dir=ARTIFACTS / "baseline_gte")
embeddings = extract_embeddings(
    bundle, encoder=PRIMARY_ENCODER, output_dir=ARTIFACTS)

summary = []
for key, values in ((BASELINE_ENCODER, gte_embeddings), (PRIMARY_ENCODER, embeddings)):
    meta = values["manifest"]
    summary.append({
        "кодировщик": key, "размерность": meta["embedding_dim"],
        "пакет": meta["batch_size"], "макс. токенов во входе": meta["max_observed_tokens"],
        "пик GPU, ГБ": meta["gpu_peak_reserved_bytes"] / 2**30,
        "доступно GPU, ГБ": meta["gpu_total_bytes"] / 2**30,
    })
display(pd.DataFrame(summary).style.format({"пик GPU, ГБ": "{:.2f}",
                                             "доступно GPU, ГБ": "{:.2f}"}))
for name in ("context", "triple", "answer_cls"):
    assert embeddings[name].shape[-1] == 2560 and np.isfinite(embeddings[name]).all()
print("Все Qwen3-векторы получены и проверены")
    """),
    md(r"""
## 4. Быстрая техническая проверка

Все представления строятся на Qwen3-векторах. Каждое классическое семейство
обучается на одной части, а каждая малая сеть выполняет одну эпоху. Эти
проверочные веса дальше не используются.
    """),
    code(r"""
from grid_smoke_test import run_smoke

shape_check, model_check = run_smoke(bundle, embeddings)
display(shape_check)
display(model_check[["data_variant", "model", "n_predictions", "finite"]])
assert shape_check.finite.all() and model_check.finite.all()
print("Размерности и один шаг всех семейств проверены")
    """),
    md(r"""
## 5. Основная проверка и сравнение кодировщиков

Основная строка сравнивает `CONFIRMATION|A` и `qwen3_4b|M0`. В обеих строках
одинаковые пять внешних частей, логистическая регрессия и внутренний выбор
регуляризации. Единственное содержательное различие — добавленные средние
Qwen3-векторы сущностей, отношений и утверждений после PCA, обученной только
на соответствующей обучающей части.
    """),
    code(r"""
from encoder_comparison import run_encoder_comparison

encoder_metrics, encoder_predictions, encoder_comparisons = run_encoder_comparison(
    bundle, {PRIMARY_ENCODER: embeddings, BASELINE_ENCODER: gte_embeddings},
    output_dir=ARTIFACTS, shuffle_repeats=20)
display(encoder_comparisons.style.format({
    "ROC_AUC_difference": "{:+.3f}", "ROC_AUC_95_low": "{:+.3f}",
    "ROC_AUC_95_high": "{:+.3f}", "log_loss_difference": "{:+.3f}",
    "log_loss_95_low": "{:+.3f}", "log_loss_95_high": "{:+.3f}"}))
display(encoder_metrics.style.format({"ROC_AUC": "{:.3f}", "PR_AUC": "{:.3f}",
                                      "log_loss": "{:.3f}", "F1_at_0.5": "{:.3f}"}))
    """),
    md(r"""
## 6. Разведочная матрица «данные × модель»

После основной проверки запускаются 139 сочетаний на Qwen3-векторах. Эта
матрица помогает выбрать идеи для следующей выборки, но её лучшая клетка не
считается независимым подтверждением качества.
    """),
    code(r"""
from grid_features import DATA_VARIANTS
from grid_models import TABULAR_MODELS, SET_DATA_VARIANTS, SET_MODELS
from grid_experiment import run_grid_experiment

display(pd.DataFrame(DATA_VARIANTS.items(), columns=["Данные", "Описание"]))
display(pd.DataFrame(TABULAR_MODELS.items(), columns=["Модель", "Описание"]))
RUN_FULL = True
NEURAL_EPOCHS = 80
if RUN_FULL:
    metrics, auc_matrix, predictions, comparisons = run_grid_experiment(
        bundle, embeddings, output_dir=ARTIFACTS, neural_epochs=NEURAL_EPOCHS)
    print("Методов:", len(metrics), "внеобучающих прогнозов:", len(predictions))
    """),
    md("## 7. Результаты разведочной матрицы"),
    code(r"""
if RUN_FULL:
    display(auc_matrix.style.format("{:.3f}", na_rep="—").background_gradient(
        cmap="RdYlGn", axis=None, vmin=0.4, vmax=max(0.75, float(np.nanmax(auc_matrix)))))
    display(metrics.head(25).style.format({"ROC_AUC": "{:.3f}", "PR_AUC": "{:.3f}",
                                           "log_loss": "{:.3f}", "F1_at_0.5": "{:.3f}"}))
    display(comparisons.style.format({"ROC_AUC_difference": "{:+.3f}",
                                      "bootstrap_95_low": "{:+.3f}",
                                      "bootstrap_95_high": "{:+.3f}"}))
    """),
    code(r"""
if RUN_FULL:
    import matplotlib.pyplot as plt
    top = metrics.head(25).sort_values("ROC_AUC")
    ax = top.plot.barh(x="method", y="ROC_AUC", legend=False, figsize=(11, 10),
                       color="#2563eb", title="25 лучших разведочных сочетаний")
    ax.axvline(0.7122, color="#dc2626", linestyle="--",
               label="лучший результат прошлого запуска")
    ax.set_xlim(0.25, max(0.78, float(top.ROC_AUC.max()) + 0.03))
    ax.legend(); plt.tight_layout(); plt.show()
    """),
    md(r"""
## 8. Проверка, отчёт и автоматическое скачивание

Архив включает оба набора векторов, основную проверку, все 139 разведочных
строк, прогнозы каждого ответа, разбиение по частям, входные данные, русский
отчёт в Markdown и HTML и контрольные суммы. При неполном результате архив
не создаётся.
    """),
    code(r"""
if RUN_FULL:
    from grid_report import write_audit_package
    from IPython.display import FileLink

    zip_path = write_audit_package(ARTIFACTS, source_root=EXP_ROOT)
    print("Архив проверен:", zip_path, f"({zip_path.stat().st_size / 2**20:.1f} МиБ)")
    display(Markdown((ARTIFACTS / "REPORT.md").read_text(encoding="utf-8")))
    display(FileLink(str(zip_path)))
    try:
        from google.colab import files
        files.download(str(zip_path))
        print("Скачивание запрошено. Если браузер заблокировал его, нажмите ссылку выше.")
    except Exception as exc:
        print("Автоматическое скачивание не удалось:", exc)
        print("Скачайте архив по ссылке выше или через панель файлов Colab.")
    """),
    md(r"""
### Ограничение

Даже основная строка остаётся пилотом: 100 искусственно сбалансированных
ответов, метки GPT-4o и нет независимой внешней выборки. Результат отвечает,
есть ли сигнал на этой сотне при заранее заданном сравнении. Для общего вывода
понадобится замороженная конфигурация и новые ответы с независимой разметкой.
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
