"""32 заранее выбранных метода, повторная групповая оценка и новая проверка."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from data_io import TYPES, load_bundle

VARIANTS = {
    'Q': '9 доступных признаков подтверждённости и количества',
    'Q_MEAN': 'Признаки + средние E/R/C, по 8 координат каждого типа',
    'Q_STATUS': 'Признаки + средние по исходам проверки компонентов',
    'Q_E': 'Признаки + среднее сущностей',
    'Q_R': 'Признаки + среднее отношений',
    'Q_C': 'Признаки + среднее утверждений',
    'Q_ANSWER': 'Признаки + общий вектор ответа',
    'EMBED_ONLY': 'Средние E/R/C без подтверждённости',
    'Q_SHUFFLED': 'Признаки + переставленные средние E/R/C',
}
MODELS = {'logreg': 'Логистическая регрессия с внутренним подбором',
          'boosting': 'Градиентный бустинг', 'forest': 'Случайный лес',
          'knn': '11 ближайших соседей'}
METHODS = tuple(f'{v}|{m}' for v in list(VARIANTS)[:7] for m in MODELS) + (
    'EMBED_ONLY|logreg', 'Q_SHUFFLED|logreg', 'BASELINE|frequency', 'BASELINE|entity_risk')
PRIMARY = ('Q|logreg', 'Q_MEAN|logreg')

def metrics(y, p):
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Неверные вероятности')
    return dict(ROC_AUC=roc_auc_score(y,p), PR_AUC=average_precision_score(y,p),
                log_loss=log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),
                F1=f1_score(y,p>=.5,zero_division=0), n=len(y))

def estimator(name, seed, c=.1):
    if name=='logreg':
        return make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=seed))
    if name=='boosting':
        return HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=7,min_samples_leaf=8,l2_regularization=5.,random_state=seed)
    if name=='forest':
        return RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_features='sqrt',n_jobs=2,random_state=seed)
    if name=='knn':
        return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=11,weights='distance',metric='cosine',algorithm='brute'))
    raise ValueError(name)

class Representations:
    def __init__(self,bundle,embeddings):
        self.q=bundle.q
        raw=embeddings['context']
        self.dimension=raw.shape[1]
        self.mean={k:np.zeros((len(bundle.ids),self.dimension),np.float32) for k in TYPES}
        self.grouped={k:[np.zeros_like(self.mean[k]) for _ in range(3)] for k in TYPES}
        self.present={k:np.zeros(len(bundle.ids),bool) for k in TYPES}
        self.group_counts=[]
        index={sid:i for i,sid in enumerate(bundle.ids)}
        buckets={k:[[] for _ in bundle.ids] for k in TYPES}
        for j,row in enumerate(bundle.components):
            buckets[row['component_type']][index[int(row['source_id'])]].append(j)
        for k in TYPES:
            counts=np.zeros((len(bundle.ids),3),np.float32)
            for i,indices in enumerate(buckets[k]):
                if not indices:
                    continue
                self.present[k][i]=True
                self.mean[k][i]=raw[indices].mean(0)
                for g in range(3):
                    chosen=[]
                    for j in indices:
                        status=bundle.components[j]['confirmation']
                        code=0 if status in ('grounded','entailed') else 1 if status in ('ungrounded','unsupported','contradicted') else 2
                        if code==g: chosen.append(j)
                    counts[i,g]=len(chosen)
                    if chosen: self.grouped[k][g][i]=raw[chosen].mean(0)
            self.group_counts.append(np.log1p(counts))
        self.answer=embeddings['answer_cls']

    def build(self,train):
        parts, status_parts={},[]
        for k in TYPES:
            fit=train[self.present[k][train]]
            if len(fit)<8:
                raise ValueError(f'Недостаточно обучающих ответов с компонентами {k}')
            pca=PCA(8,svd_solver='randomized',random_state=42).fit(self.mean[k][fit])
            parts[k]=pca.transform(self.mean[k]).astype(np.float32)
            parts[k][~self.present[k]]=0
            for g in range(3):
                value=pca.transform(self.grouped[k][g]).astype(np.float32)
                # Пустая группа не должна превращаться в ненулевой PCA-вектор.
                value[~np.any(self.grouped[k][g],axis=1)]=0
                status_parts.append(value)
        answer=PCA(8,svd_solver='randomized',random_state=42).fit(self.answer[train]).transform(self.answer)
        mean=np.column_stack([parts[k] for k in TYPES])
        matrices={'Q':self.q,'Q_MEAN':np.column_stack([self.q,mean]),
            'Q_STATUS':np.column_stack([self.q,*status_parts,*self.group_counts]),
            'Q_ANSWER':np.column_stack([self.q,answer]),'EMBED_ONLY':mean,
            'Q_SHUFFLED':np.column_stack([self.q,mean])}
        for code,k in zip('ERC',TYPES): matrices['Q_'+code]=np.column_stack([self.q,parts[k]])
        return matrices

def group_splits(bundle,indices,seed,n=5):
    split=StratifiedGroupKFold(n,shuffle=True,random_state=seed)
    for local_train,local_test in split.split(indices,bundle.y[indices],bundle.groups[indices]):
        train,test=indices[local_train],indices[local_test]
        if set(bundle.groups[train]) & set(bundle.groups[test]):
            raise AssertionError('Пересечение групп')
        if len(set(bundle.y[test]))!=2:
            raise ValueError('В проверочной части отсутствует один класс')
        yield train,test

def inner_select(bundle,store,train,seed,variants):
    losses={v:np.zeros(3) for v in variants}
    cs=(.01,.1,1.)
    for fit,val in group_splits(bundle,train,seed,n=3):
        matrices=store.build(fit)
        for variant in variants:
            base='Q_MEAN' if variant=='Q_SHUFFLED' else variant
            for j,c in enumerate(cs):
                p=estimator('logreg',seed,c).fit(matrices[base][fit],bundle.y[fit]).predict_proba(matrices[base][val])[:,1]
                losses[variant][j]+=log_loss(bundle.y[val],p,labels=[0,1])
    return {v:cs[int(np.argmin(values))] for v,values in losses.items()}

def run_split(bundle,store,train,test,seed,methods):
    matrices=store.build(train)
    linear=[m.split('|')[0] for m in methods if m.endswith('|logreg') and not m.startswith('Q_SHUFFLED')]
    chosen=inner_select(bundle,store,train,seed,linear) if linear else {}
    output,details={},[]
    for method in methods:
        variant,name=method.split('|')
        if method=='BASELINE|frequency': p=np.full(len(test),bundle.y[train].mean())
        elif method=='BASELINE|entity_risk': p=bundle.q[test,1]
        elif variant=='Q_SHUFFLED':
            predictions=[]; c=chosen.get('Q_MEAN',.1)
            for repeat in range(5):
                rng=np.random.default_rng(seed+repeat*71); matrix=matrices['Q_MEAN'].copy()
                width=bundle.q.shape[1]
                matrix[train,width:]=matrix[rng.permutation(train),width:]
                matrix[test,width:]=matrix[rng.permutation(test),width:]
                model=estimator('logreg',seed,c).fit(matrix[train],bundle.y[train])
                predictions.append(model.predict_proba(matrix[test])[:,1])
            p=np.mean(predictions,axis=0)
        else:
            c=chosen.get(variant,.1)
            model=estimator(name,seed,c).fit(matrices[variant][train],bundle.y[train])
            p=model.predict_proba(matrices[variant][test])[:,1]
            details.append(dict(method=method,C=c if name=='logreg' else None,feature_count=matrices[variant].shape[1]))
        metrics(bundle.y[test],p)
        output[method]=p
    return output,details

def bootstrap_difference(bundle,predictions,reference,candidate,indices,repeats=2000):
    # Пересэмплируем группы целиком, сохраняя зависимость копий ответов.
    y=bundle.y[indices]; groups=bundle.groups[indices]
    unique=np.unique(groups); positions={g:np.where(groups==g)[0] for g in unique}
    rng=np.random.default_rng(42); deltas=[]
    a=predictions[reference]; b=predictions[candidate]
    for _ in range(repeats):
        rows=np.concatenate([positions[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(set(y[rows]))==2:
            deltas.append(roc_auc_score(y[rows],b[rows])-roc_auc_score(y[rows],a[rows]))
    return dict(reference=reference,candidate=candidate,ROC_AUC_difference=roc_auc_score(y,b)-roc_auc_score(y,a),
                low=float(np.quantile(deltas,.025)),high=float(np.quantile(deltas,.975)))

def run_experiment(bundle,embeddings,output_dir,*,repeats=3,methods=METHODS):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    dev=np.where(bundle.partitions=='development')[0]; hold=np.where(bundle.partitions=='holdout')[0]
    store=Representations(bundle,embeddings)
    rows,fold_rows,details=[],[],[]
    with threadpool_limits(limits=2):
        for repeat in range(repeats):
            for fold,(train,test) in enumerate(group_splits(bundle,dev,42+repeat*101)):
                predictions,config=run_split(bundle,store,train,test,42+repeat*101+fold,methods)
                for method,p in predictions.items():
                    fold_rows.append(dict(repeat=repeat,fold=fold,method=method,**metrics(bundle.y[test],p)))
                    rows.extend(dict(repeat=repeat,fold=fold,method=method,source_id=bundle.ids[i],gold=int(bundle.y[i]),probability=float(value)) for i,value in zip(test,p))
                details.append(dict(repeat=repeat,fold=fold,train_ids=[bundle.ids[i] for i in train],test_ids=[bundle.ids[i] for i in test],settings=config))
                print(f'Повтор {repeat+1}/{repeats}, часть {fold+1}/5: {len(methods)} методов',flush=True)
        predictions,config=run_split(bundle,store,dev,hold,20261003,methods)
    oof=pd.DataFrame(rows); fm=pd.DataFrame(fold_rows)
    repeat_rows=[]
    for (repeat,method),group in oof.groupby(['repeat','method']):
        if len(group)!=len(dev) or group.source_id.duplicated().any(): raise AssertionError('Неполные прогнозы')
        repeat_rows.append(dict(repeat=repeat,method=method,**metrics(group.gold.to_numpy(),group.probability.to_numpy())))
    rm=pd.DataFrame(repeat_rows)
    summary=[]
    for method in methods:
        fold_data=fm[fm.method==method]; repeat_data=rm[rm.method==method]
        row=dict(method=method,description=VARIANTS.get(method.split('|')[0],method),model_description=MODELS.get(method.split('|')[1],method))
        for metric in ('ROC_AUC','PR_AUC','log_loss','F1'):
            row[metric+'_fold_mean']=fold_data[metric].mean()
            row[metric+'_fold_std']=fold_data[metric].std(ddof=1)
            row[metric+'_repeat_mean']=repeat_data[metric].mean()
            row[metric+'_repeat_std']=repeat_data[metric].std(ddof=1) if repeats>1 else 0.
        summary.append(row)
    cv=pd.DataFrame(summary).sort_values('ROC_AUC_repeat_mean',ascending=False)
    hold_metrics=pd.DataFrame([dict(method=m,**metrics(bundle.y[hold],p)) for m,p in predictions.items()])
    hold_predictions=pd.DataFrame([dict(method=m,source_id=bundle.ids[i],gold=int(bundle.y[i]),probability=float(v)) for m,p in predictions.items() for i,v in zip(hold,p)])
    comparisons=[]
    if set(PRIMARY).issubset(methods):
        comparisons.append(bootstrap_difference(bundle,predictions,*PRIMARY,hold))
    if 'Q_SHUFFLED|logreg' in methods and PRIMARY[1] in methods:
        comparisons.append(bootstrap_difference(bundle,predictions,'Q_SHUFFLED|logreg',PRIMARY[1],hold))
    for name,table in [('cv_metrics.csv',cv),('cv_fold_metrics.csv',fm),('cv_repeat_metrics.csv',rm),('cv_predictions.csv',oof),('holdout_metrics.csv',hold_metrics),('holdout_predictions.csv',hold_predictions),('holdout_comparisons.csv',pd.DataFrame(comparisons))]:
        table.to_csv(output_dir/name,index=False)
    (output_dir/'fold_details.json').write_text(json.dumps(dict(cv=details,holdout_settings=config),ensure_ascii=False,indent=2),encoding='utf-8')
    (output_dir/'experiment_manifest.json').write_text(json.dumps(dict(dataset_signature=bundle.signature,n_methods=len(methods),n_answers=len(bundle.ids),development=len(dev),holdout=len(hold),repeats=repeats,folds=5,primary=PRIMARY,methods=list(methods),relation_confirmation_available=False,feature_count=bundle.q.shape[1],holdout_selection='Fixed seed; groups absent from old pilot only'),ensure_ascii=False,indent=2),encoding='utf-8')
    return cv,hold_metrics
