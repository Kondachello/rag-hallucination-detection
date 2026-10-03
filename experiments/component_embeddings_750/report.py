"""Пересчёт метрик и самодостаточный архив для аудита."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from compact_experiment import metrics, METHODS, PRIMARY
from data_io import ROOT, DATA, load_bundle
from embeddings import MODEL_ID, MODEL_REVISION
from sklearn.metrics import log_loss

def read_csv(path):
    # Сохраняем точные float64 и равенство прогнозов при чтении CSV.
    return pd.read_csv(path, float_precision='round_trip')

def matches_metric(computed, saved, metric, method, group):
    if np.isclose(computed,saved,atol=1e-8,rtol=0):
        return True
    # В первом запуске только этот контроль возвращал исходный float32.
    # Подтверждаем его старую log_loss точно, не ослабляя остальные проверки.
    if metric=='log_loss' and method=='BASELINE|entity_risk':
        p=group.probability.to_numpy(np.float32)
        legacy=log_loss(group.gold.to_numpy(),np.clip(p,1e-7,1-1e-7),labels=[0,1])
        return np.isclose(legacy,saved,atol=1e-8,rtol=0)
    return False

def package(output_dir, *, expected_methods=METHODS):
    out=Path(output_dir); b=load_bundle()
    manifest=json.loads((out/'experiment_manifest.json').read_text(encoding='utf-8'))
    em=json.loads((out/'embeddings_manifest.json').read_text(encoding='utf-8'))
    if em['data_signature']!=b.signature or manifest['dataset_signature']!=b.signature:
        raise ValueError('Несовместимые данные и результаты')
    if em.get('model_id')!=MODEL_ID or em.get('revision')!=MODEL_REVISION or em.get('embedding_dim')!=4096 or em.get('weight_quantization')!='int8':
        raise ValueError('Результаты относятся к другому кодировщику')
    if set(manifest['methods'])!=set(expected_methods): raise ValueError('Неполный набор методов')
    with np.load(out/'embeddings.npz') as arrays:
        for name,n in [('context',len(b.components)),('answer_cls',len(b.ids))]:
            if arrays[name].shape!=(n,em['embedding_dim']) or not np.isfinite(arrays[name]).all():
                raise ValueError('Неполные эмбеддинги')
    if em['component_order']!=[f"{int(c['source_id'])}:{c['component_id']}" for c in b.components] or em['response_order']!=b.ids:
        raise ValueError('Неверный порядок векторов')
    cv=read_csv(out/'cv_metrics.csv'); fm=read_csv(out/'cv_fold_metrics.csv')
    rm=read_csv(out/'cv_repeat_metrics.csv'); hm=read_csv(out/'holdout_metrics.csv')
    oof=read_csv(out/'cv_predictions.csv'); hp=read_csv(out/'holdout_predictions.csv')
    parts={'development':set(np.array(b.ids)[b.partitions=='development']),
           'holdout':set(np.array(b.ids)[b.partitions=='holdout'])}
    label_map=dict(zip(b.ids,b.y))
    for scope,pred in [('development',oof),('holdout',hp)]:
        if set(pred.method)!=set(expected_methods): raise ValueError('Не все методы рассчитаны')
        if any(label_map[int(r.source_id)]!=r.gold for r in pred.itertuples()): raise ValueError('Неверные метки')
        columns=['repeat','method'] if scope=='development' else ['method']
        for key,group in pred.groupby(columns):
            if set(group.source_id)!=parts[scope] or group.source_id.duplicated().any():
                raise ValueError('Пропущенные или повторные прогнозы')
            computed=metrics(group.gold.to_numpy(),group.probability.to_numpy())
            if scope=='development':
                repeat,method=key; row=rm[(rm.repeat==repeat)&(rm.method==method)].iloc[0]
            else:
                method=key[0]; row=hm[hm.method==method].iloc[0]
            for metric in ('ROC_AUC','PR_AUC','log_loss','F1'):
                if not matches_metric(computed[metric],row[metric],metric,method,group):
                    raise ValueError(f'Метрики не совпали: {scope}, {key}, {metric}: {computed[metric]} / {row[metric]}')
    if set(oof['repeat'])!=set(range(manifest['repeats'])): raise ValueError('Не все повторы готовы')
    for (repeat,fold,method),group in oof.groupby(['repeat','fold','method']):
        values=metrics(group.gold.to_numpy(),group.probability.to_numpy())
        saved=fm[(fm.repeat==repeat)&(fm.fold==fold)&(fm.method==method)]
        if len(saved)!=1: raise ValueError('Пропущена метрика части')
        for metric in ('ROC_AUC','PR_AUC','log_loss','F1'):
            if not matches_metric(values[metric],saved.iloc[0][metric],metric,method,group):
                raise ValueError('Метрика части не совпала с прогнозами')
    for row in cv.to_dict('records'):
        f=fm[fm.method==row['method']]; r=rm[rm.method==row['method']]
        if len(f)!=manifest['repeats']*5 or len(r)!=manifest['repeats']: raise ValueError('Неполные разбиения')
        for metric in ('ROC_AUC','PR_AUC','log_loss','F1'):
            for suffix,value in [('fold_mean',f[metric].mean()),('fold_std',f[metric].std(ddof=1)),('repeat_mean',r[metric].mean())]:
                if not np.isclose(row[metric+'_'+suffix],value,atol=1e-8,rtol=0): raise ValueError('Неверная сводная метрика')
    comparisons=read_csv(out/'holdout_comparisons.csv')
    table=cv[['method','description','ROC_AUC_repeat_mean','ROC_AUC_fold_mean','ROC_AUC_fold_std','ROC_AUC_repeat_std','PR_AUC_repeat_mean','log_loss_repeat_mean']].copy()
    table=table.merge(hm[['method','ROC_AUC','PR_AUC','log_loss','F1']],on='method',suffixes=('','_holdout'))
    table.to_csv(out/'final_table.csv',index=False)
    display=table.copy()
    for metric in ('ROC_AUC','PR_AUC','log_loss'):
        display[metric+'_CV']=display[metric+'_repeat_mean'].map(lambda v:f'{v:.4f}')
    display['ROC_AUC_части']=display.apply(lambda r:f"{r.ROC_AUC_fold_mean:.4f} ± {r.ROC_AUC_fold_std:.4f}",axis=1)
    display=display[['method','description','ROC_AUC_CV','ROC_AUC_части','ROC_AUC','PR_AUC','log_loss','F1']]
    html=display.to_html(index=False,float_format=lambda v:f'{v:.4f}',escape=True)
    notes=(f"750 исходных ответов; {manifest['development']} для разработки, {manifest['holdout']} для отложенной проверки. "
        f"{manifest['repeats']} повтора групповой кросс-валидации по 5 частей. "
        "Копии ответов и одинаковые входы находятся в одной группе. Прежние 100 ответов исключены из отложенной части. "
        "Стандартное отклонение по частям показывает разброс и не является доверительным интервалом. "
        "Отдельное стандартное отклонение между повторами показывает чувствительность к разбиению. "
        "Парные интервалы на отложенной части получены пересэмплированием целых групп. "
        "PCA обучается на средних векторах обучающих ответов; C логистической регрессии выбирается внутренними 3 частями. "
        "Проверок отношений нет: все они not_verified. Используются 9 доступных признаков. "
        "Entity risk — доля неподтверждённых сущностей; это не полная формула HalluGraph. "
        "Метки исходных ответов выставлены GPT-4o. Методика извлечения компонентов отличается от прошлого пилота, "
        "поэтому разницу метрик между наборами нельзя приписать только размеру выборки. "
        "Отложенные результаты всех методов — заранее заданные вторичные проверки; выбор лучшего по ним потребует новой выборки.")
    notes+=" Исходная log_loss контроля entity_risk проверена также в float32 для совместимости с первым запуском; новые запуски используют float64."
    report='# Отчёт: эмбеддинги компонентов на 750 ответах\n\n'+notes+'\n\n## Главная проверка\n\n'
    report+=comparisons.to_csv(index=False)+'\n## Полная таблица\n\n'
    report+='| '+' | '.join(display.columns)+' |\n|'+'|'.join('---' for _ in display.columns)+'|\n'
    for row in display.itertuples(index=False,name=None): report+='| '+' | '.join(str(v).replace('|',' / ') for v in row)+' |\n'
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    (out/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;padding:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4rem}</style><h1>Эмбеддинги компонентов: 750 ответов</h1><p>'+notes+'</p><h2>Основные сравнения</h2>'+comparisons.to_html(index=False)+'<h2>Все методы</h2>'+html,encoding='utf-8')
    required=('cv_metrics.csv','cv_fold_metrics.csv','cv_repeat_metrics.csv','cv_predictions.csv',
              'holdout_metrics.csv','holdout_predictions.csv','holdout_comparisons.csv','fold_details.json',
              'experiment_manifest.json','embeddings_manifest.json','embedding_trace.jsonl','embeddings.npz',
              'final_table.csv','REPORT.md','REPORT.html')
    paths={name:out/name for name in required}
    paths.update({'data/'+p.name:p for p in DATA.iterdir() if p.is_file()})
    paths.update({'code/'+p.name:p for p in ROOT.iterdir() if p.suffix in ('.py','.ipynb') or p.name in ('requirements.txt','README.md')})
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    run=dict(git_commit=commit,files={name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for name,p in paths.items()})
    (out/'run_manifest.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    paths['run_manifest.json']=out/'run_manifest.json'
    archive=out/'component_750_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for name,path in paths.items(): z.write(path,arcname=name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist())!=set(paths): raise ValueError('Архив повреждён')
    return archive
