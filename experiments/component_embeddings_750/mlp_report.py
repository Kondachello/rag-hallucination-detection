"""Строгая проверка готовых прогнозов MLP и архив без повторения большого кэша."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from compact_experiment import metrics
from data_io import ROOT, DATA, load_bundle
from mlp_models import CONFIGS
from embedding_cache import sha

def read(name,out):
    return pd.read_csv(out/name,float_precision='round_trip')

def compare(computed,row,keys):
    for key in keys:
        if not np.isclose(computed[key],row[key],atol=1e-8,rtol=0):
            raise ValueError(f'Метрика не совпала: {key}; {computed[key]} / {row[key]}')

def package(output_dir,bundle=None):
    out=Path(output_dir); b=bundle or load_bundle()
    manifest=json.loads((out/'experiment_manifest.json').read_text(encoding='utf-8'))
    em=json.loads((out/'embeddings_manifest.json').read_text(encoding='utf-8'))
    if manifest['data_signature']!=b.signature or em['data_signature']!=b.signature:
        raise ValueError('Несовместимые данные и результаты')
    cache=json.loads((ROOT/'embedding_cache/manifest.json').read_text())
    if em['fingerprint']!=cache['fingerprint']:
        raise ValueError('Другой кэш эмбеддингов')
    methods=set(manifest['methods']); repeats=manifest['repeats']
    dev=set(np.array(b.ids)[b.partitions=='development']); hold=set(np.array(b.ids)[b.partitions=='holdout'])
    gold=dict(zip(b.ids,b.y)); keys=('ROC_AUC','PR_AUC','log_loss','F1')
    oof=read('cv_predictions.csv',out); hp=read('holdout_predictions.csv',out)
    fm=read('cv_fold_metrics.csv',out); rm=read('cv_repeat_metrics.csv',out)
    hm=read('holdout_metrics.csv',out); cv=read('cv_metrics.csv',out)
    sp=read('cv_seed_predictions.csv',out); sm=read('cv_seed_metrics.csv',out)
    hs=read('holdout_seed_predictions.csv',out)
    for table,columns in [(cv,['method']),(hm,['method']),(rm,['repeat','method']),
                          (fm,['repeat','fold','method']),(sm,['repeat','method','seed'])]:
        if set(table.method)!=methods or table.duplicated(columns).any():
            raise ValueError('Пропущенные/повторные метрики')
    if set(oof['repeat'])!=set(range(repeats)) or set(sp['repeat'])!=set(range(repeats)):
        raise ValueError('Отсутствуют повторы')
    for pred in (oof,hp,sp,hs):
        if set(pred.method)!=methods:
            raise ValueError('Неполный набор методов')
        if any(gold[int(row.source_id)]!=row.gold for row in pred.itertuples()):
            raise ValueError('Неверная целевая метка')
    for (repeat,method),group in oof.groupby(['repeat','method']):
        if set(group.source_id)!=dev or group.source_id.duplicated().any():
            raise ValueError('Неполные/повторные прогнозы')
        saved=rm[(rm.repeat==repeat)&(rm.method==method)]
        if len(saved)!=1: raise ValueError('Нет метрики повтора')
        compare(metrics(group.gold.to_numpy(),group.probability.to_numpy()),saved.iloc[0],keys)
    for (repeat,fold,method),group in oof.groupby(['repeat','fold','method']):
        saved=fm[(fm.repeat==repeat)&(fm.fold==fold)&(fm.method==method)]
        if len(saved)!=1: raise ValueError('Нет метрики части')
        compare(metrics(group.gold.to_numpy(),group.probability.to_numpy()),saved.iloc[0],keys)
    for (repeat,method),group in oof.groupby(['repeat','method']):
        if set(group.fold)!=set(range(5)): raise ValueError('Пропущена часть')
    for method,group in hp.groupby('method'):
        if set(group.source_id)!=hold or group.source_id.duplicated().any():
            raise ValueError('Неполная отложенная проверка')
        compare(metrics(group.gold.to_numpy(),group.probability.to_numpy()),hm[hm.method==method].iloc[0],keys)
    for (repeat,method,seed),group in sp.groupby(['repeat','method','seed']):
        if set(group.source_id)!=dev or group.source_id.duplicated().any():
            raise ValueError('Неполный прогноз инициализации')
        saved=sm[(sm.repeat==repeat)&(sm.method==method)&(sm.seed==seed)]
        compare(metrics(group.gold.to_numpy(),group.probability.to_numpy()),saved.iloc[0],keys)
    expected_seeds={m:({0} if m.startswith('LR_') else set(manifest['seeds'])) for m in methods}
    for (repeat,method),group in sp.groupby(['repeat','method']):
        if set(group.seed)!=expected_seeds[method]: raise ValueError('Нет начального состояния')
    for method,group in hs.groupby('method'):
        if set(group.seed)!=expected_seeds[method]: raise ValueError('Нет отложенного начального состояния')
        for _,one in group.groupby('seed'):
            if set(one.source_id)!=hold or one.source_id.duplicated().any(): raise ValueError('Неполный прогноз')
    for ensemble,seeds_pred,grouping in [(oof,sp,['repeat','fold','method','source_id']),
                                         (hp,hs,['method','source_id'])]:
        averaged=seeds_pred.groupby(grouping).probability.mean().sort_index()
        actual=ensemble.set_index(grouping).probability.sort_index()
        if not averaged.index.equals(actual.index) or not np.allclose(averaged,actual,atol=1e-8,rtol=0):
            raise ValueError('Неверное усреднение начальных состояний')
    for row in cv.to_dict('records'):
        f=fm[fm.method==row['method']]; r=rm[rm.method==row['method']]; s=sm[sm.method==row['method']]
        if len(f)!=repeats*5 or len(r)!=repeats: raise ValueError('Неполные части')
        for metric in keys:
            values={metric+'_fold_mean':f[metric].mean(),metric+'_fold_std':f[metric].std(ddof=1),
                    metric+'_repeat_mean':r[metric].mean(),metric+'_repeat_std':r[metric].std(ddof=1) if repeats>1 else 0.,
                    metric+'_initialization_std_mean':s.groupby('repeat')[metric].std(ddof=1).fillna(0).mean()}
            compare(values,row,values)
    details=json.loads((out/'fold_details.json').read_text(encoding='utf-8'))
    lookup=b.splits.set_index('source_id')['group'].to_dict()
    def groups(ids): return {lookup[i] for i in ids}
    for records in [*[entry['settings'] for entry in details['cv']],details['holdout']]:
        for record in records:
            if groups(record['train_ids']) & groups(record['test_ids']): raise ValueError('Утечка между частями')
            if groups(record['inner_fit_ids']) & groups(record['inner_validation_ids']): raise ValueError('Утечка внутри обучения')
            if set(record['inner_fit_ids'])|set(record['inner_validation_ids'])!=set(record['train_ids']): raise ValueError('Неверное внутреннее разбиение')
    table=cv.merge(hm,on='method',suffixes=('','_holdout'))
    table.to_csv(out/'final_table.csv',index=False,float_format='%.17g')
    comparisons=read('holdout_comparisons.csv',out) if (out/'holdout_comparisons.csv').stat().st_size>1 else pd.DataFrame()
    display=table[['method','description','ROC_AUC_repeat_mean','ROC_AUC_fold_std','ROC_AUC_repeat_std',
                   'ROC_AUC_initialization_std_mean','ROC_AUC','PR_AUC','log_loss','F1']]
    notes=('Эмбеддинги Qwen3-8B взяты из точного кэша, кодировщик не запускался. '
           '9 доступных признаков; подтверждённости отношений нет. '
           'Сеть и линейная голова обучаются совместно. Общая PCA и стандартизация обучаются только на обучении. '
           'Эпохи/C выбираются на внутренней групповой части, затем выполняется повторное обучение на всей обучающей части. '
           'Среднее вероятностей двух начальных состояний; отдельно показан разброс по частям, повторам и инициализациям. '
           'Сравнения разведочные: отложенные 121 ответ уже были просмотрены в прошлом опыте. '
           'Для подтверждения выбранного победителя нужны новые ответы.')
    report='# Опыт с MLP/FFN\n\n'+notes+'\n\n## Таблица\n\n'
    report+='| '+' | '.join(display.columns)+' |\n|'+'|'.join('---' for _ in display.columns)+'|\n'
    for row in display.itertuples(index=False,name=None):
        report+='| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v).replace('|','/') for v in row)+' |\n'
    report+='\n## Отложенные парные сравнения\n\n'+comparisons.to_csv(index=False)
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    (out/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;padding:2rem}td,th{padding:.4rem;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>MLP/FFN</h1><p>'+notes+'</p>'+display.to_html(index=False,float_format=lambda v:f'{v:.4f}')+'<h2>Парные сравнения</h2>'+comparisons.to_html(index=False),encoding='utf-8')
    paths={}
    for path in out.rglob('*'):
        if path.is_file() and path.suffix not in ('.zip','.tmp') and path.name!='run_manifest.json':
            paths[str(path.relative_to(out))]=path
    for path in DATA.iterdir():
        if path.is_file(): paths['data/'+path.name]=path
    for path in ROOT.iterdir():
        if path.is_file() and (path.suffix in ('.py','.ipynb') or path.name in ('mlp_requirements.txt','MLP_DESIGN.md')):
            paths['code/'+path.name]=path
    paths['cache_reference.json']=ROOT/'embedding_cache/manifest.json'
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    run=dict(git_commit=commit,metrics_verified=True,files={name:dict(sha256=sha(path),bytes=path.stat().st_size) for name,path in paths.items()})
    (out/'run_manifest.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    paths['run_manifest.json']=out/'run_manifest.json'
    archive=out/'component_750_mlp_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for name,path in paths.items(): z.write(path,name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist())!=set(paths): raise ValueError('Повреждён ZIP')
    return archive
