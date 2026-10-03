"""Сборка русского Colab-ноутбука исключительно на сохранённых векторах."""
from pathlib import Path
import nbformat as nbf

md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[md('''# Малые MLP над эмбеддингами компонентов

**Среда выполнения → Выполнить все.** Можно использовать T4 или CPU.
Кодировщик Qwen3-8B не загружается: его готовые эмбеддинги уже в GitHub.
Повторять 51-минутное получение векторов не нужно.

Ноутбук проверит 31 метод, покажет таблицы и скачает
`component_750_mlp_results.zip`. При повторном запуске завершённые обучения
будут восстановлены из промежуточных файлов.
'''),md('''## 1. Подготовка и данные

750 исходных ответов, 256 с галлюцинациями. Метки автоматические, от GPT-4o.
629 ответов для групповой кросс-валидации, 121 для дополнительной проверки.
Прежние ответы уже входят в набор. Одинаковые входы и копии ответа образуют
группы и не пересекают границы обучения и проверки.
'''),code('''from pathlib import Path
import os, sys, subprocess, json, hashlib
checkout=Path('/content/component_750_mlp_repo')
if (Path.cwd()/'mlp_experiment.py').is_file() and not Path.cwd().is_relative_to(checkout):
    EXP_ROOT=Path.cwd()
else:
    if not checkout.exists():
        subprocess.run(['git','clone','--depth','1','--branch','span_by_kolya',
                        'https://github.com/Kondachello/rag-hallucination-detection.git',str(checkout)],check=True)
    else:
        subprocess.run(['git','-C',str(checkout),'pull','--ff-only','origin','span_by_kolya'],check=True)
    EXP_ROOT=checkout/'experiments/component_embeddings_750'
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(EXP_ROOT/'mlp_requirements.txt')],check=True)
os.chdir(EXP_ROOT)
sys.path.insert(0,str(EXP_ROOT))
import numpy as np, pandas as pd, torch
from IPython.display import display, Markdown, FileLink
from data_io import load_bundle, FEATURES
bundle=load_bundle()
device='cuda' if torch.cuda.is_available() else 'cpu'
print('Обучение MLP:',device,'PyTorch:',torch.__version__)
display(pd.crosstab(bundle.partitions,bundle.y,rownames=['Часть'],colnames=['Галлюцинация']))
display(pd.Series(FEATURES,name='9 доступных признаков').to_frame())
'''),md('''## 2. Точный кэш, без кодировщика

Сущности получены из ответа с указателем целевой сущности; отношения —
из ответа с полной направленной тройкой; утверждения — из отдельного текста.
Есть также вектор целого ответа. Опорный контекст и целевые метки не подавались.

Кэш сохранён без изменения чисел. Проверяются контрольные суммы, версия
кодировщика, порядок компонентов и подпись набора.
'''),code('''from embedding_cache import restore_cache
from data_io import component_key
CACHE=EXP_ROOT/'artifacts_750_qwen8b'
ARTIFACTS=EXP_ROOT/'artifacts_mlp'
restore_cache(bundle,CACHE)
meta=json.loads((CACHE/'embeddings_manifest.json').read_text(encoding='utf-8'))
assert meta['data_signature']==bundle.signature
assert meta['component_order']==[component_key(c) for c in bundle.components]
assert meta['response_order']==bundle.ids
with np.load(CACHE/'embeddings.npz',allow_pickle=False) as arrays:
    embeddings={name:arrays[name] for name in ('context','answer_cls')}
embeddings['manifest']=meta
for name,shape in [('context',(16868,4096)),('answer_cls',(750,4096))]:
    assert embeddings[name].shape==shape and np.isfinite(embeddings[name]).all()
    print(name,embeddings[name].shape)
print('Готовые эмбеддинги загружены. Веса Qwen не используются.')
'''),md('''## 3. Архитектуры и контроли

Основной FFN: **PCA128 → расширение до 256 → ReLU → сужение до 32**.
Последний линейный слой получает 9 признаков и четыре выхода по 32:
**137 → sigmoid → вероятность галлюцинации**. Он и FFN обучаются совместно.
Для общего блока с указателем типа добавляется обучаемый вектор из 8 координат.

Комбинации: общий блок / общий с типом / четыре блока;
FFN до или после агрегации; среднее или максимум. Дополнительно:
выход 16, GELU вместо ReLU, отключение признаков, исходные 4096 координат,
только компоненты, только ответ, линейное преобразование и MLP только на признаках.
Для исходных 4096 координат блок **4096 → 256 → 32** не расширяется:
это контроль отказа от PCA, без десятков миллионов параметров.

Линейный контроль LR_Q_ALL получает те же четыре PCA128-вектора, что FFN.
LR_Q_MEAN и LR_Q_ANSWER здесь тоже используют PCA128: их новые цифры нельзя
напрямую считать повторением прежних моделей с PCA8.
'''),code('''from mlp_models import CONFIGS, BASELINES, METHODS, Detector, parameter_count
from mlp_experiment import TRAINING
table=[]
for config in CONFIGS:
    model=Detector(config,4096 if config.space=='raw' else 128)
    table.append({'Метод':config.name,'Описание':config.description(),
                  'Параметры':parameter_count(model)})
display(pd.DataFrame(table))
print('Линейные контроли:',BASELINES)
print('Всего:',len(METHODS),'методов; нейросетей:',len(CONFIGS))
print('Настройки обучения:',TRAINING)
assert len(METHODS)==31
'''),md('''## 4. Как обучаем и проверяем

**5 частей × 3 повтора**, те же внешние группы, что в прошлом опыте.
Для каждой сети — два начальных состояния; итоговая вероятность усредняется.
Стандартизация и общая PCA обучаются только на обучающих ответах.
В PCA каждый непустой тип каждого ответа даёт один средний вектор:
ответы с большим числом компонентов не доминируют в преобразовании.

Внутри обучения выделяется одна групповая часть для выбора числа эпох/C.
После выбора сеть создаётся заново и учится на всей обучающей части.
Внешняя проверочная часть не влияет на остановку. Максимум 100 эпох,
регуляризация весов, отключение 20% нейронов, ограничение градиента.
Пустые группы после FFN обнуляются; при максимуме дополнение не считается компонентом.

**Ограничение:** 121 отложенный ответ уже был просмотрен. Все новые гипотезы
разведочные; подтверждать выбранного победителя нужно на новых данных.
'''),code('''from mlp_experiment import run_experiment
cv_metrics,holdout_metrics=run_experiment(bundle,embeddings,ARTIFACTS,
                                        repeats=3,seeds=(11,29),device=device)
print('Все модели обучены и прогнозы сохранены.')
'''),md('''## 5. Большая таблица результатов

Показываем отдельно среднее ROC-AUC трёх повторов, стандартное отклонение
по 15 частям, между повторами и от начального состояния сети.
Стандартное отклонение — разброс, а не доверительный интервал.
Для основных пар на отложенной части интервалы строятся по целым группам.
'''),code('''display(cv_metrics[['method','description','ROC_AUC_repeat_mean','ROC_AUC_fold_std',
                    'ROC_AUC_repeat_std','ROC_AUC_initialization_std_mean',
                    'PR_AUC_repeat_mean','log_loss_repeat_mean']].style.format(precision=4))
display(holdout_metrics.style.format(precision=4))
display(pd.read_csv(ARTIFACTS/'holdout_comparisons.csv',float_precision='round_trip').style.format(precision=4))
'''),md('''## 6. Проверка, отчёт и автоматическое скачивание

Архив содержит прогнозы всех частей и начальных состояний, метрики,
разбиения, историю обучения, число параметров, веса финальных моделей,
преобразования, код и данные. Большой кэш не дублируется: он уже сохранён
в GitHub, в архиве есть его точная ссылка через контрольные суммы.

Пришлите **весь ZIP** для аудита. Если браузер блокирует автоматическое
скачивание, используйте ссылку или панель файлов Colab.
'''),code('''from mlp_report import package
import zipfile, traceback
try:
    archive=package(ARTIFACTS)
    display(Markdown((ARTIFACTS/'REPORT.md').read_text(encoding='utf-8')))
except Exception:
    # Сохраняем результаты даже при ошибке отчёта; не выдаём их за проверенные.
    note={'metrics_verified':False,'error':traceback.format_exc()}
    (ARTIFACTS/'AUDIT_STATUS.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
    print(note['error'])
    archive=ARTIFACTS/'component_750_mlp_results_unverified.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in ARTIFACTS.rglob('*'):
            if p.is_file() and p.suffix not in ('.zip','.tmp'):
                z.write(p,str(p.relative_to(ARTIFACTS)))
        for p in (EXP_ROOT/'data').iterdir():
            if p.is_file(): z.write(p,'data/'+p.name)
        for p in EXP_ROOT.iterdir():
            if p.is_file() and p.suffix in ('.py','.ipynb'): z.write(p,'code/'+p.name)
print('Архив:',archive,'размер:',round(archive.stat().st_size/2**20,1),'МиБ')
display(FileLink(str(archive)))
try:
    from google.colab import files
    files.download(str(archive))
except Exception as error:
    print('Скачайте через панель файлов:',archive.name,'Причина:',error)
''')]
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},
                              'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':'run_mlp_experiment.ipynb','provenance':[],'gpuType':'T4'}})
nbf.validate(notebook)
nbf.write(notebook,Path(__file__).parent/'run_mlp_experiment.ipynb')
