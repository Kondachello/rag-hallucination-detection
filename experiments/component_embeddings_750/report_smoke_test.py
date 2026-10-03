"""Проверка упаковки на искусственных прогнозах; обучение не выполняется."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from data_io import load_bundle, component_key
from compact_experiment import METHODS, metrics, group_splits, bootstrap_difference, PRIMARY
from embeddings import MODEL_ID, MODEL_REVISION
from report import package

def run():
    b=load_bundle(); rng=np.random.default_rng(5)
    dev=np.where(b.partitions=='development')[0]; hold=np.where(b.partitions=='holdout')[0]
    with tempfile.TemporaryDirectory() as temporary:
        out=Path(temporary); records=[]; folds=[]; repeat_rows=[]; repeats=2
        for repeat in range(repeats):
            for fold,(_,test) in enumerate(group_splits(b,dev,42+101*repeat)):
                for method in METHODS:
                    p=rng.uniform(.1,.9,len(test))
                    folds.append(dict(repeat=repeat,fold=fold,method=method,**metrics(b.y[test],p)))
                    records.extend(dict(repeat=repeat,fold=fold,method=method,source_id=b.ids[i],gold=int(b.y[i]),probability=float(v)) for i,v in zip(test,p))
        oof=pd.DataFrame(records); fm=pd.DataFrame(folds)
        for (repeat,method),g in oof.groupby(['repeat','method']):
            repeat_rows.append(dict(repeat=repeat,method=method,**metrics(g.gold.to_numpy(),g.probability.to_numpy())))
        rm=pd.DataFrame(repeat_rows); summary=[]
        for method in METHODS:
            row=dict(method=method,description='Искусственные данные для проверки упаковки')
            f=fm[fm.method==method]; r=rm[rm.method==method]
            for metric in ('ROC_AUC','PR_AUC','log_loss','F1'):
                row[metric+'_fold_mean']=f[metric].mean(); row[metric+'_fold_std']=f[metric].std(ddof=1)
                row[metric+'_repeat_mean']=r[metric].mean(); row[metric+'_repeat_std']=r[metric].std(ddof=1)
            summary.append(row)
        preds={m:rng.uniform(.1,.9,len(hold)) for m in METHODS}
        hm=pd.DataFrame([dict(method=m,**metrics(b.y[hold],p)) for m,p in preds.items()])
        hp=pd.DataFrame([dict(method=m,source_id=b.ids[i],gold=int(b.y[i]),probability=float(v)) for m,p in preds.items() for i,v in zip(hold,p)])
        comp=pd.DataFrame([bootstrap_difference(b,preds,*PRIMARY,hold,repeats=10)])
        for name,table in [('cv_metrics.csv',pd.DataFrame(summary)),('cv_fold_metrics.csv',fm),('cv_repeat_metrics.csv',rm),('cv_predictions.csv',oof),('holdout_metrics.csv',hm),('holdout_predictions.csv',hp),('holdout_comparisons.csv',comp)]:
            table.to_csv(out/name,index=False)
        (out/'experiment_manifest.json').write_text(json.dumps(dict(dataset_signature=b.signature,n_answers=len(b.ids),n_methods=32,methods=METHODS,development=len(dev),holdout=len(hold),repeats=repeats)),encoding='utf-8')
        (out/'embeddings_manifest.json').write_text(json.dumps(dict(data_signature=b.signature,model_id=MODEL_ID,revision=MODEL_REVISION,embedding_dim=4096,weight_quantization='int8',component_order=[component_key(c) for c in b.components],response_order=b.ids)),encoding='utf-8')
        np.savez_compressed(out/'embeddings.npz',context=np.zeros((len(b.components),4096),np.float16),answer_cls=np.zeros((len(b.ids),4096),np.float16))
        (out/'embedding_trace.jsonl').write_text('{}\n',encoding='utf-8')
        (out/'fold_details.json').write_text('{}',encoding='utf-8')
        archive=package(out)
        assert archive.is_file()
        # Независимый пересчёт должен отвергнуть испорченную метрику.
        hm.loc[0,'ROC_AUC']=.123456; hm.to_csv(out/'holdout_metrics.csv',index=False)
        try: package(out)
        except ValueError: pass
        else: raise AssertionError('Испорченная метрика не обнаружена')
        print('OK: полный ZIP и обнаружение неверной метрики; обучение не выполнялось')

if __name__=='__main__': run()
