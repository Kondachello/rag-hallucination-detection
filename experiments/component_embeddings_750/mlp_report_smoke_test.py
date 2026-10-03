"""Полный отчёт на искусственных прогнозах; модели не обучаются."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from data_io import load_bundle, ROOT
from compact_experiment import group_splits,metrics
from mlp_models import METHODS,BASELINES
from mlp_experiment import summarise
from mlp_report import package

def check():
    b=load_bundle(); rng=np.random.default_rng(4)
    dev=np.where(b.partitions=='development')[0]; hold=np.where(b.partitions=='holdout')[0]
    rows=[]; fold_rows=[]; seed_rows=[]; details=[]
    for fold,(train,test) in enumerate(group_splits(b,dev,42)):
        fit,val=next(group_splits(b,train,59,n=5))
        details.append(dict(repeat=0,fold=fold,settings=[dict(train_ids=[b.ids[i] for i in train],test_ids=[b.ids[i] for i in test],
                         inner_fit_ids=[b.ids[i] for i in fit],inner_validation_ids=[b.ids[i] for i in val])]))
        for method in METHODS:
            probs=[]
            for seed in ((0,) if method in BASELINES else (11,29)):
                p=rng.uniform(.1,.9,len(test)); probs.append(p)
                seed_rows.extend(dict(repeat=0,fold=fold,method=method,seed=seed,source_id=b.ids[i],gold=int(b.y[i]),probability=float(v)) for i,v in zip(test,p))
            p=np.mean(probs,axis=0)
            fold_rows.append(dict(repeat=0,fold=fold,method=method,**metrics(b.y[test],p)))
            rows.extend(dict(repeat=0,fold=fold,method=method,source_id=b.ids[i],gold=int(b.y[i]),probability=float(v)) for i,v in zip(test,p))
    oof,fm,rm,sp,sm,cv=summarise(rows,fold_rows,seed_rows,METHODS)
    hm=[]; hp=[]; hs=[]
    for method in METHODS:
        probs=[]
        for seed in ((0,) if method in BASELINES else (11,29)):
            p=rng.uniform(.1,.9,len(hold)); probs.append(p)
            hs.extend(dict(method=method,seed=seed,source_id=b.ids[i],gold=int(b.y[i]),probability=float(v)) for i,v in zip(hold,p))
        p=np.mean(probs,axis=0)
        hm.append(dict(method=method,**metrics(b.y[hold],p)))
        hp.extend(dict(method=method,source_id=b.ids[i],gold=int(b.y[i]),probability=float(v)) for i,v in zip(hold,p))
    fit,val=next(group_splits(b,dev,59,n=5))
    hold_details=[dict(train_ids=[b.ids[i] for i in dev],test_ids=[b.ids[i] for i in hold],
                     inner_fit_ids=[b.ids[i] for i in fit],inner_validation_ids=[b.ids[i] for i in val])]
    em=json.loads((ROOT/'artifacts_received/embeddings_manifest.json').read_text(encoding='utf-8')) if (ROOT/'artifacts_received/embeddings_manifest.json').exists() else dict(
        data_signature=b.signature,fingerprint=json.loads((ROOT/'embedding_cache/manifest.json').read_text())['fingerprint'])
    with TemporaryDirectory() as temp:
        out=Path(temp)
        tables={'cv_predictions.csv':oof,'cv_fold_metrics.csv':fm,'cv_repeat_metrics.csv':rm,'cv_metrics.csv':cv,
                'cv_seed_predictions.csv':sp,'cv_seed_metrics.csv':sm,'holdout_metrics.csv':pd.DataFrame(hm),
                'holdout_predictions.csv':pd.DataFrame(hp),'holdout_seed_predictions.csv':pd.DataFrame(hs),
                'holdout_comparisons.csv':pd.DataFrame(columns=['reference','candidate','ROC_AUC_difference','low','high'])}
        for name,table in tables.items(): table.to_csv(out/name,index=False,float_format='%.17g')
        (out/'experiment_manifest.json').write_text(json.dumps(dict(data_signature=b.signature,methods=METHODS,repeats=1,seeds=[11,29])))
        (out/'embeddings_manifest.json').write_text(json.dumps(em))
        (out/'fold_details.json').write_text(json.dumps(dict(cv=details,holdout=hold_details)))
        archive=package(out,b)
        assert archive.is_file()
        bad=pd.read_csv(out/'holdout_metrics.csv'); bad.loc[0,'ROC_AUC']+=.01
        bad.to_csv(out/'holdout_metrics.csv',index=False)
        try: package(out,b)
        except ValueError: pass
        else: raise AssertionError('Повреждённая метрика не обнаружена')
    print('OK all 31 methods: metrics, seed ensembles, groups, ZIP; corruption rejected')

if __name__=='__main__': check()
