"""Создаёт Colab-ноутбук без выполнения ячеек."""
from pathlib import Path
import nbformat as nbf

md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[md('''# Эмбеддинги компонентов: 750 ответов

Выберите **T4 GPU**, затем **Среда выполнения → Выполнить все**.
Ноутбук сам установит зависимости, получит эмбеддинги, выполнит 32 метода,
покажет метрики с разбросом и скачает `component_750_results.zip`.

Основной кодировщик — **Qwen3-Embedding-8B**, 4096 координат. Для T4 веса
хранятся в 8-битном формате; вычисления используют FP16. Фактический расход
видеопамяти записывается. Модель заморожена, целевые метки в неё не передаются.
'''),md('''## 1. Подготовка

Архив содержит исходные ответы Llama и автоматические метки GPT-4o.
В новом наборе **494 ответа без галлюцинаций и 256 с галлюцинациями**.
Искусственная балансировка не применяется. Исправленные ответы не используются.
'''),code('''from pathlib import Path
import subprocess, sys, os
REPO = 'https://github.com/Kondachello/rag-hallucination-detection.git'
BRANCH = 'span_by_kolya'
if (Path.cwd()/'compact_experiment.py').is_file() and not Path.cwd().is_relative_to(Path('/content/component_750_repo')):
    EXP_ROOT = Path.cwd()
else:
    checkout = Path('/content/component_750_repo')
    if not checkout.exists():
        subprocess.run(['git','clone','--depth','1','--branch',BRANCH,REPO,str(checkout)],check=True)
    else:
        subprocess.run(['git','-C',str(checkout),'pull','--ff-only','origin',BRANCH],check=True)
    EXP_ROOT = checkout/'experiments/component_embeddings_750'
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(EXP_ROOT/'requirements.txt')],check=True)
os.chdir(EXP_ROOT)
sys.path.insert(0,str(EXP_ROOT))
import torch, numpy as np, pandas as pd
from IPython.display import display, Markdown, FileLink
from data_io import load_bundle, FEATURES
bundle = load_bundle()
if torch.cuda.is_available():
    print('GPU:',torch.cuda.get_device_name(0), 'память:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2),'ГиБ')
else:
    print('Используется CPU: готовые эмбеддинги уже сохранены в репозитории.')
print('Ответы:',len(bundle.ids),'компоненты:',len(bundle.components))
print('PyTorch:',torch.__version__)
display(pd.crosstab(bundle.partitions,bundle.y,rownames=['Часть'],colnames=['Галлюцинация']))
display(pd.Series(FEATURES,name='Доступные признаки').to_frame())
'''),md('''## 2. Что именно проверяется

Оставлены 32 метода из прежних 139 — около 23%. Основные модели:
логистическая регрессия, бустинг, случайный лес и ближайшие соседи.
Данные: признаки без векторов, средние всех компонентов, группы по исходу
проверки, каждый тип отдельно и общий вектор ответа. Есть контроль без
подтверждённости и контроль с переставленными векторами.

В архиве **нет проверки отношений по контексту**. Их тройки используются
для получения векторов, но оценки помечены `not_verified` и не включаются
в признаки подтверждённости. Поэтому доступно 9 признаков вместо прежних 13.
'''),code('''from compact_experiment import METHODS, VARIANTS, MODELS
display(pd.DataFrame(VARIANTS.items(),columns=['Представление','Описание']))
display(pd.DataFrame(MODELS.items(),columns=['Модель','Описание']))
print('Заранее выбранных методов:',len(METHODS))
assert len(METHODS)==32
'''),md('''## 3. Получение эмбеддингов

Сущность: ответ и явно указанная целевая сущность. Отношение: ответ и полная
направленная тройка. Утверждение: его отдельный текст. Общий вектор: полный ответ.
Опорный контекст и целевые метки в эти входы не включаются.
Берётся вектор последнего токена, затем нормируется до единичной длины.

Размер пакета автоматически проверяется на самых длинных входах. Одинаковые
тексты вычисляются один раз. Повторный запуск использует готовый кэш векторов.
'''),code('''from embeddings import extract_embeddings, example_prompts, MODEL_ID, MODEL_REVISION
ARTIFACTS = EXP_ROOT/'artifacts_750_qwen8b'
from embedding_cache import restore_cache
restore_cache(bundle, ARTIFACTS)
print(MODEL_ID, MODEL_REVISION)
display(pd.Series(example_prompts(bundle),name='Точные примеры входов').to_frame())
embeddings = extract_embeddings(bundle,output_dir=ARTIFACTS)
meta = embeddings['manifest']
print('Пакет:',meta['batch_size'],'пик:',round(meta['gpu_peak_reserved_bytes']/2**30,2),'ГиБ')
for key in ('context','answer_cls'):
    print(key,embeddings[key].shape)
    assert embeddings[key].shape[1]==4096 and np.isfinite(embeddings[key]).all()
'''),md('''## 4. Как оценивается качество

629 ответов используются для разработки: 3 повтора кросс-валидации по 5 частей.
Одинаковые входы и копии ответов входят в одну группу и не пересекают границу
обучения и проверки. Стандартизация и PCA обучаются только на обучении.
Сначала усредняются компоненты каждого ответа, затем обучается PCA: длинные
ответы не получают больший вес из-за количества компонентов.

У логистической регрессии регуляризация выбирается внутри обучения на 3
групповых частях. Остальные модели имеют заранее фиксированные параметры.
Для каждого метода сохраняются среднее и стандартное отклонение по 15 частям,
а также отдельное стандартное отклонение итоговых метрик между 3 повторами.

121 ответ отложен отдельно; среди них нет прежней сотни и её групп.
Основная пара: признаки + логистическая регрессия против той же модели
с добавлением средних E/R/C-векторов. В конце все заранее выбранные методы
оцениваются на отложенной части. Выбирать по ней нового победителя и считать
его оценку независимой нельзя. 95% интервалы разницы строятся по целым группам.
'''),code('''from compact_experiment import run_experiment
cv_metrics, holdout_metrics = run_experiment(bundle,embeddings,ARTIFACTS,repeats=3)
print('Повторная оценка и отложенная проверка завершены.')
'''),md('''## 5. Результаты и разброс

ROC-AUC выше — лучше; логарифмическая ошибка ниже — лучше. Обозначение
«±» показывает стандартное отклонение между проверочными частями.
Оно не является доверительным интервалом и не делится на корень из 15:
части и повторы используют пересекающиеся обучающие наборы.
'''),code('''display(cv_metrics[['method','ROC_AUC_repeat_mean','ROC_AUC_fold_mean','ROC_AUC_fold_std','ROC_AUC_repeat_std','PR_AUC_repeat_mean','log_loss_repeat_mean']].style.format(precision=4))
display(holdout_metrics.style.format(precision=4))
display(pd.read_csv(ARTIFACTS/'holdout_comparisons.csv').style.format(precision=4))
matrix = cv_metrics.assign(variant=cv_metrics.method.str.split('|').str[0],model=cv_metrics.method.str.split('|').str[1]).pivot(index='variant',columns='model',values='ROC_AUC_repeat_mean')
display(matrix.style.format('{:.3f}',na_rep='—').background_gradient(cmap='RdYlGn',axis=None))
'''),md('''## 6. Отчёт и скачивание

Архив содержит эмбеддинги, точные тексты входов, все прогнозы, разброс метрик,
отложенную проверку, данные, код и русский отчёт. Перед упаковкой метрики
пересчитываются из прогнозов. При неполном запуске упаковка завершится ошибкой.
'''),code('''from report import package
archive = package(ARTIFACTS)
display(Markdown((ARTIFACTS/'REPORT.md').read_text(encoding='utf-8')))
print('Полный архив:',archive,round(archive.stat().st_size/2**20,1),'МиБ')
display(FileLink(str(archive)))
try:
    from google.colab import files
    files.download(str(archive))
except Exception as error:
    print('Скачайте файл через панель файлов Colab:',error)
''')]
notebook=nbf.v4.new_notebook(cells=cells)
notebook.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'run_experiment.ipynb','provenance':[],'gpuType':'T4'},'accelerator':'GPU'}
nbf.validate(notebook)
nbf.write(notebook,Path(__file__).parent/'run_experiment.ipynb')
