"""Воспроизводимая сборка основного ноутбука без выполнения его ячеек."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def md(value: str):
    return nbf.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbf.v4.new_code_cell(value.strip())


cells = [
    md(r"""
# Детектор галлюцинаций по векторным представлениям компонентов

**Опыт 2 · 100 ответов Llama 3.1 · пять заранее заданных частей**

Здесь сущности (E), отношения (R) и атомарные утверждения (C) превращаются в
числовые векторы замороженным BGE-small. Затем мы сравниваем 30 способов
объединить переменное число компонентов с 13 признаками подтверждённости.
Цель — предсказать наличие галлюцинации **во всём ответе**.

> Главная проверка: помогают ли средние E/R/C сверх одних оценок
> подтверждённости (`M0` против `A`)? Остальные архитектуры — разведочные
> гипотезы. Полная таблица появится в конце после выполнения ячеек.

Данные и код лежат рядом с этим ноутбуком в одной папке. Целевые метки
GPT-4o отделены от текстов и не используются при получении векторов.
    """),
    md(r"""
## 1. Подготовка Colab

Выберите среду с GPU в меню «Среда выполнения → Сменить среду выполнения».
Укажите ниже свою ветку GitHub. Первая ячейка клонирует репозиторий и
устанавливает небольшие библиотеки. PyTorch берётся из среды Colab.
    """),
    code(r"""
from pathlib import Path
import os, subprocess, sys

REPO_URL = "https://github.com/Kondachello/rag-hallucination-detection.git"
BRANCH = "span_by_kolya"  # измените, если папка опубликована в другой ветке
SUBDIR = Path("experiments/component_embeddings_colab")

if (Path.cwd() / "data_io.py").is_file():
    EXP_ROOT = Path.cwd().resolve()
elif (Path.cwd() / SUBDIR / "data_io.py").is_file():
    EXP_ROOT = (Path.cwd() / SUBDIR).resolve()
else:
    checkout = Path("/content/hallu_smiles_component_exp")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                        REPO_URL, str(checkout)], check=True)
    EXP_ROOT = checkout / SUBDIR
    if not (EXP_ROOT / "data_io.py").is_file():
        raise FileNotFoundError(f"Папка опыта не найдена: {EXP_ROOT}; проверьте ветку")

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
import sklearn
import transformers
from IPython.display import display, Markdown
from data_io import load_bundle, FEATURES, TYPES

bundle = load_bundle()
print("Версии:", {"torch": torch.__version__, "transformers": transformers.__version__,
                 "numpy": np.__version__, "sklearn": sklearn.__version__})
print("GPU доступен:", torch.cuda.is_available())
print("Ответов:", len(bundle.ids), "Компонентов:", len(bundle.components))
display(pd.DataFrame({
    "Тип": ["Сущность", "Отношение", "Утверждение"],
    "Количество": [sum(c["component_type"] == kind for c in bundle.components) for kind in TYPES],
}))
display(pd.crosstab(bundle.fold_numbers, bundle.y, rownames=["Часть"],
                    colnames=["Метка галлюцинации"]))
    """),
    md(r"""
## 2. Как формируются входы BERT

Выбран **[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)**,
закреплённая ревизия указана в `embeddings.py`. BGE-small — компактный
англоязычный кодировщик семейства BERT, дающий 384 числа на токен.

| Компонент | Текст на входе | Как берём вектор |
|---|---|---|
| E | Полный исходный ответ | Среднее по токенам **её упоминания**, по символьным границам |
| R | Пара: ответ и направленная тройка | Среднее по токенам **предиката во второй части** |
| C | Только текст атомарного утверждения | Первый служебный токен `[CLS]` |

Для сравнения `B_triples` также кодируем отношения без ответа и берём
сущности из концов троек. Если точного совпадения нет, кодируем имя отдельно.
Контекст источника и вопрос здесь не добавляются: их влияние уже есть в
автоматических проверках подтверждённости. Усечение запрещено.
    """),
    code(r"""
from embeddings import MODEL_ID, MODEL_REVISION, make_jobs, example_prompts

jobs, prompt_stats = make_jobs(bundle)
examples = example_prompts(bundle)
print("Модель:", MODEL_ID, "ревизия:", MODEL_REVISION)
print("Число подач модели:", len(jobs), "Запасные способы:", prompt_stats)
display(pd.DataFrame([
    {"Компонент": "E", "Текст": examples["answer"][:300] + "…",
     "Позиция вектора": "символьный интервал сущности в полном ответе"},
    {"Компонент": "R", "Текст": examples["triple"],
     "Позиция вектора": f"предикат: {examples['relation_predicate']!r}"},
    {"Компонент": "C", "Текст": examples["claim"],
     "Позиция вектора": "[CLS] отдельного утверждения"},
]))
    """),
    md(r"""
## 3. Получение векторов и контроль позиций

Первый запуск скачает модель из Hugging Face. Веса останутся неизменными.
Каждый текст проверяется на предел 512 токенов, а каждый символьный интервал
на наличие соответствующих токенов. При ошибке вычисление останавливается.
Готовые векторы и подробная трассировка сохраняются в `artifacts/` и
повторно используются при запуске в той же среде.
    """),
    code(r"""
from embeddings import extract_embeddings

embeddings = extract_embeddings(bundle, output_dir=EXP_ROOT / "artifacts", batch_size=16)
meta = embeddings["manifest"]
display(pd.DataFrame([
    {"Набор": name, "Размерность": str(embeddings[name].shape),
     "Конечные числа": bool(np.isfinite(embeddings[name]).all())}
    for name in ("context", "triple", "answer_cls")
]))
print("Устройство:", meta["device"], "максимум токенов:", meta["max_observed_tokens"])
print("Сущностей без позиции в ответе:", meta["answer_entity_fallbacks"])
print("Сущностей без совпавшего конца тройки:", meta["triple_entity_fallbacks"])
    """),
    code(r"""
import json

trace_path = EXP_ROOT / "artifacts" / "embedding_trace.jsonl"
with trace_path.open(encoding="utf-8") as handle:
    trace = [json.loads(next(handle)) for _ in range(5)]
display(pd.DataFrame(trace)[["job", "kind", "key", "part", "char_span",
                             "token_positions", "tokens"]])
    """),
    md(r"""
## 4. Проверка размерностей и один шаг маленьких моделей

На каждой внешней части метод главных компонент переводит векторы E/R/C
из 384 в 8 координат. Он обучается **только на 80 обучающих ответах**.
Это сокращение не использует целевые метки. Пробный один шаг ниже проверяет
формы входов всех девяти нейросетевых вариантов; его веса не используются
в основном опыте.
    """),
    code(r"""
from features import make_fold_view, feature_matrix
from models import NEURAL_METHODS, one_step_smoke

train_example = np.where(bundle.fold_numbers != 0)[0]
view_example = make_fold_view(bundle, embeddings, train_example)
print("A:", feature_matrix(view_example, "A").shape,
      "B:", feature_matrix(view_example, "B").shape,
      "A+B:", feature_matrix(view_example, "M0").shape)
step_losses = {method: one_step_smoke(method, view_example, train_example[:8], bundle.y)
               for method in NEURAL_METHODS}
display(pd.DataFrame({"Модель": list(step_losses),
                      "Потеря после одного шага (только проверка кода)": list(step_losses.values())}))
    """),
    md(r"""
## 5. Основной опыт: одинаковые пять частей для всех методов

Для линейных моделей сила регуляризации выбирается во внутреннем
трёхчастном разделении, включая повторную подгонку сокращения размерности.
Для малых нейросетей по внутренней проверке выбирается число эпох, затем
модель заново обучается на всех 80 ответах. На оставшихся 20 строится один
внеобучающий прогноз. Так повторяется пять раз.

**Варианты:** `P0` — частота класса; `HG` — проверяемый строгий HalluGraph;
`A` — только подтверждённость; `B` — только средние векторы; `M0` — их сумма;
`M1`–`M14` — слои, объединение среднего/максимума, группы исходов,
обучаемые веса и общий вектор ответа; дополнительно проверяются
вариант троек и вклад каждого типа.

Порог для F1 заранее фиксирован как 0.5. Главная метрика — средняя
логарифмическая потеря вероятностей. Запуск всех вариантов может занять
заметное время, особенно первый расчёт векторов.
    """),
    code(r"""
from experiment import ALL_METHODS, run_experiment

RUN_ALL = True
if RUN_ALL:
    metrics, predictions, comparison = run_experiment(
        bundle, embeddings, methods=ALL_METHODS,
        output_dir=EXP_ROOT / "artifacts", neural_max_epochs=120)
    print("Получено внеобучающих прогнозов:", len(predictions),
          "для каждого из", len(ALL_METHODS), "методов")
else:
    print("Включите RUN_ALL для полного расчёта таблицы")
    """),
    md(r"""
## 6. Итоговая таблица

Таблица содержит **все** методы, краткое описание, логарифмическую потерю
(меньше лучше), площадь под ROC-кривой, площадь под кривой точность–полнота,
F1 при пороге 0.5 и число проверенных ответов. `HG_raw` — исходная шкала
риска, поэтому для неё логарифмическая потеря не указывается;
`HG_cal` — тот же риск после обучения калибровки только на 80 ответах.
    """),
    code(r"""
if RUN_ALL:
    display(metrics.style.format({"log_loss": "{:.3f}", "AUROC": "{:.3f}",
                                  "PR_AUC": "{:.3f}", "F1_at_0.5": "{:.3f}",
                                  "coverage": "{:d}"}, na_rep="—"))
    print("Основное сравнение M0 минус A:")
    display(pd.DataFrame([comparison]))
    print("Файлы результатов:")
    for path in sorted((EXP_ROOT / "artifacts").iterdir()):
        print(" •", path.name)
    """),
    code(r"""
if RUN_ALL:
    import matplotlib.pyplot as plt
    shown = metrics.dropna(subset=["log_loss"]).sort_values("log_loss").head(15)
    ax = shown.set_index("method").log_loss.sort_values().plot.barh(
        figsize=(9, 7), color="#2563eb", title="15 лучших вариантов по логарифмической потере")
    ax.set_xlabel("Средняя логарифмическая потеря на 100 внеобучающих прогнозах")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
### Как читать результат

Начните с разницы `M0−A`: отрицательная означает, что векторы улучшили
главную метрику. Интервал строится попарно по тем же 100 ответам и носит
разведочный характер. Остальные архитектуры предложены как гипотезы;
выбор «лучшей» строки по той же сотне не является независимой проверкой.
Метки GPT-4o и искусственный баланс 50/50 ограничивают перенос вывода на
реальный поток. Вариант HalluGraph здесь соответствует локальной строгой
формуле и должен быть сопоставлен с отчётом, когда он появится.
    """),
]

# Большая таблица остаётся последней исполняемой ячейкой ноутбука.
cells[-3:] = [cells[-2], cells[-1], cells[-3]]
notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "run_experiment.ipynb", "provenance": []},
}
nbf.validate(notebook)
nbf.write(notebook, ROOT / "run_experiment.ipynb")
print(ROOT / "run_experiment.ipynb")
