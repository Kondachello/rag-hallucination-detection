"""Совместное обучение FFN и линейной головы с групповой оценкой."""
import hashlib
import json
import time
import joblib
import sklearn
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from compact_experiment import group_splits, metrics, bootstrap_difference
from mlp_models import CONFIGS, BASELINES, METHODS, Detector, seed_everything, parameter_count
from mlp_data import Store

TRAINING=dict(max_epochs=100,patience=10,min_epochs=10,learning_rate=.001,
              weight_decay=.01,batch_size=32,min_delta=.0001)

def predict(model,view,rows,config,device,batch_size=64):
    model.eval()
    values=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            logits=model(view.batch(rows[start:start+batch_size],config,device))
            values.extend(torch.sigmoid(logits).cpu().double().tolist())
    return np.asarray(values,np.float64)

def train_network(config,view,bundle,train,seed,device,settings,*,validation=None,epochs=None):
    seed_everything(seed)
    model=Detector(config,view.dimension,bundle.q.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    criterion=nn.BCEWithLogitsLoss()
    rng=np.random.default_rng(seed)
    labels=bundle.y
    best_loss=float('inf'); best_epoch=1; stale=0; history=[]
    limit=settings['max_epochs'] if epochs is None else epochs
    for epoch in range(1,limit+1):
        model.train(); total=0.
        shuffled=rng.permutation(train)
        for start in range(0,len(train),settings['batch_size']):
            rows=shuffled[start:start+settings['batch_size']]
            target=torch.as_tensor(labels[rows],device=device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(view.batch(rows,config,device)),target)
            if not torch.isfinite(loss):
                raise ValueError('Нечисловая ошибка обучения: '+config.name)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total+=loss.item()*len(rows)
        record=dict(epoch=epoch,train_loss=total/len(train))
        if validation is not None:
            p=predict(model,view,validation,config,device)
            value=metrics(labels[validation],p)['log_loss']
            record['validation_loss']=value
            if value<best_loss-settings['min_delta']:
                best_loss=value; best_epoch=epoch; stale=0
            else:
                stale+=1
        history.append(record)
        if validation is not None and epoch>=settings['min_epochs'] and stale>=settings['patience']:
            break
    return model,best_epoch if validation is not None else limit,history

def read_result(path,run_signature,test_ids):
    if not path.is_file():
        return None
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['run_signature']!=run_signature or data['test_ids']!=test_ids:
        raise ValueError('Несовместимый промежуточный результат: '+str(path))
    p=np.asarray(data['probability'],np.float64)
    if len(p)!=len(test_ids) or not np.isfinite(p).all():
        raise ValueError('Повреждён промежуточный результат')
    return data

def write_result(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)

def run_split(bundle,store,train,test,split_seed,output_dir,tag,signature,*,
              configs=CONFIGS,seeds=(11,29),settings=TRAINING,device='cpu',pca_dimension=128,save_models=False):
    train=np.asarray(train,int); test=np.asarray(test,int)
    if set(bundle.groups[train]) & set(bundle.groups[test]):
        raise AssertionError('Пересечение групп')
    methods=BASELINES+tuple(c.name for c in configs)
    directory=Path(output_dir)/'progress'/tag
    directory.mkdir(parents=True,exist_ok=True)
    test_ids=[bundle.ids[i] for i in test]
    pending=False
    for method in methods:
        for seed in (seeds if method not in BASELINES else (0,)):
            if read_result(directory/f'{method}_{seed}.json',signature,test_ids) is None:
                pending=True
    if pending:
        fit,val=next(group_splits(bundle,train,split_seed+17,n=5))
        inner=store.fit(fit,pca_dimension,include_raw=any(c.space=='raw' for c in configs))
        final=store.fit(train,pca_dimension,include_raw=any(c.space=='raw' for c in configs))
        if save_models:
            folder=Path(output_dir)/'models'; folder.mkdir(exist_ok=True)
            joblib.dump(final['preprocessing'],folder/'preprocessing.joblib',compress=3)
        assert set(inner['pca'].fit_indices)==set(fit)
        assert set(final['pca'].fit_indices)==set(train)
        common=dict(run_signature=signature,test_ids=test_ids,
                    train_ids=[bundle.ids[i] for i in train],
                    inner_fit_ids=[bundle.ids[i] for i in fit],inner_validation_ids=[bundle.ids[i] for i in val])
        for method in BASELINES:
            path=directory/f'{method}_0.json'
            if read_result(path,signature,test_ids) is not None:
                continue
            losses=[]; cs=(.01,.1,1.)
            for c in cs:
                model=LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=split_seed)
                model.fit(inner['pca'].linear(method)[fit],bundle.y[fit])
                p=model.predict_proba(inner['pca'].linear(method)[val])[:,1]
                losses.append(metrics(bundle.y[val],p)['log_loss'])
            chosen=cs[int(np.argmin(losses))]
            model=LogisticRegression(C=chosen,solver='liblinear',max_iter=2000,random_state=split_seed)
            model.fit(final['pca'].linear(method)[train],bundle.y[train])
            p=model.predict_proba(final['pca'].linear(method)[test])[:,1]
            if save_models:
                joblib.dump(model,folder/f'{method}.joblib',compress=3)
            write_result(path,common|dict(method=method,seed=0,probability=p.tolist(),C=chosen,validation_losses=losses,
                                         parameters=model.coef_.size+1))
        for config in configs:
            for seed in seeds:
                path=directory/f'{config.name}_{seed}.json'
                if read_result(path,signature,test_ids) is not None:
                    continue
                clock=time.monotonic()
                model,epochs,selection_history=train_network(config,inner[config.space],bundle,fit,split_seed+seed,device,settings,validation=val)
                del model
                model,_,history=train_network(config,final[config.space],bundle,train,split_seed+seed,device,settings,epochs=epochs)
                p=predict(model,final[config.space],test,config,device)
                metrics(bundle.y[test],p)
                if save_models:
                    folder=Path(output_dir)/'models'; folder.mkdir(exist_ok=True)
                    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                                    config=vars(config),dimension=final[config.space].dimension),folder/f'{config.name}_{seed}.pt')
                write_result(path,common|dict(method=config.name,seed=seed,probability=p.tolist(),epochs=epochs,
                              parameters=parameter_count(model),seconds=time.monotonic()-clock,
                              selection_history=selection_history,training_history=history,config=vars(config)))
                print(f'  {tag}: {config.name}, начало {seed}, эпох {epochs}',flush=True)
                del model
    output={}; seed_outputs={}; details=[]
    for method in methods:
        candidates=[]
        for seed in (seeds if method not in BASELINES else (0,)):
            data=read_result(directory/f'{method}_{seed}.json',signature,test_ids)
            candidates.append(np.asarray(data['probability'],np.float64))
            seed_outputs[(method,seed)]=candidates[-1]
            details.append({k:v for k,v in data.items() if k!='probability'})
        output[method]=np.mean(candidates,axis=0)
    return output,seed_outputs,details

def summarise(rows,fold_rows,seed_rows,methods):
    oof=pd.DataFrame(rows); fm=pd.DataFrame(fold_rows); sp=pd.DataFrame(seed_rows)
    rm=pd.DataFrame([dict(repeat=r,method=m,**metrics(g.gold.to_numpy(),g.probability.to_numpy()))
                     for (r,m),g in oof.groupby(['repeat','method'])])
    sm=pd.DataFrame([dict(repeat=r,method=m,seed=s,**metrics(g.gold.to_numpy(),g.probability.to_numpy()))
                     for (r,m,s),g in sp.groupby(['repeat','method','seed'])])
    descriptions={c.name:c.description() for c in CONFIGS}
    descriptions.update(LR_Q='Логрег только на 9 признаках',LR_Q_MEAN='Логрег: признаки + E/R/C (PCA128)',
                        LR_Q_ANSWER='Логрег: признаки + ответ (PCA128)',LR_Q_ALL='Логрег: те же четыре вектора, что FFN (PCA128)')
    summary=[]
    for method in methods:
        row=dict(method=method,description=descriptions[method])
        f=fm[fm.method==method]; r=rm[rm.method==method]; s=sm[sm.method==method]
        for metric in ('ROC_AUC','PR_AUC','log_loss','F1'):
            row[metric+'_fold_mean']=f[metric].mean(); row[metric+'_fold_std']=f[metric].std(ddof=1)
            row[metric+'_repeat_mean']=r[metric].mean(); row[metric+'_repeat_std']=r[metric].std(ddof=1) if len(r)>1 else 0.
            variability=s.groupby('repeat')[metric].std(ddof=1).fillna(0)
            row[metric+'_initialization_std_mean']=variability.mean()
        summary.append(row)
    return oof,fm,rm,sp,sm,pd.DataFrame(summary).sort_values('ROC_AUC_repeat_mean',ascending=False)

def run_experiment(bundle,embeddings,output_dir,*,repeats=3,seeds=(11,29),configs=CONFIGS,
                   settings=None,pca_dimension=128,device=None):
    settings=TRAINING|dict(settings or {})
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    methods=BASELINES+tuple(c.name for c in configs)
    code_hash=hashlib.sha256(b''.join((Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')
                            for name in ('mlp_models.py','mlp_data.py','mlp_experiment.py'))).hexdigest()
    protocol=dict(data_signature=bundle.signature,embedding_fingerprint=embeddings['manifest']['fingerprint'],
                  repeats=repeats,seeds=list(seeds),settings=settings,pca_dimension=pca_dimension,
                  configs=[vars(c) for c in configs],code_hash=code_hash,device=device,
                  versions=dict(numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__))
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    store=Store(bundle,embeddings)
    dev=np.where(bundle.partitions=='development')[0]; hold=np.where(bundle.partitions=='holdout')[0]
    rows=[]; folds=[]; seed_rows=[]; details=[]
    with threadpool_limits(limits=2):
        for repeat in range(repeats):
            for fold,(train,test) in enumerate(group_splits(bundle,dev,42+repeat*101)):
                print(f'Повтор {repeat+1}/{repeats}, часть {fold+1}/5',flush=True)
                p,sp,info=run_split(bundle,store,train,test,42+repeat*101+fold,out,f'r{repeat}_f{fold}',signature,
                     configs=configs,seeds=seeds,settings=settings,device=device,pca_dimension=pca_dimension)
                for method,prob in p.items():
                    folds.append(dict(repeat=repeat,fold=fold,method=method,**metrics(bundle.y[test],prob)))
                    rows.extend(dict(repeat=repeat,fold=fold,method=method,source_id=bundle.ids[i],gold=int(bundle.y[i]),probability=float(v)) for i,v in zip(test,prob))
                for (method,seed),prob in sp.items():
                    seed_rows.extend(dict(repeat=repeat,fold=fold,method=method,seed=seed,source_id=bundle.ids[i],gold=int(bundle.y[i]),probability=float(v)) for i,v in zip(test,prob))
                details.append(dict(repeat=repeat,fold=fold,settings=info))
        p,sp,info=run_split(bundle,store,dev,hold,20261003,out,'holdout',signature,configs=configs,seeds=seeds,
                 settings=settings,device=device,pca_dimension=pca_dimension,save_models=True)
    oof,fm,rm,spm,sm,cv=summarise(rows,folds,seed_rows,methods)
    hm=pd.DataFrame([dict(method=m,**metrics(bundle.y[hold],prob)) for m,prob in p.items()])
    hp=pd.DataFrame([dict(method=m,source_id=bundle.ids[i],gold=int(bundle.y[i]),probability=float(v)) for m,prob in p.items() for i,v in zip(hold,prob)])
    hs=pd.DataFrame([dict(method=m,seed=s,source_id=bundle.ids[i],gold=int(bundle.y[i]),probability=float(v)) for (m,s),prob in sp.items() for i,v in zip(hold,prob)])
    comparisons=[]
    for reference,candidate in [('LR_Q','MLP_Q'),('LR_Q_ALL','FFN_shared_after_mean'),
                                ('LR_Q_ALL','FFN_typed_after_mean'),('LR_Q_ALL','FFN_separate_after_mean'),
                                ('LR_Q_ANSWER','FFN_typed_after_mean')]:
        if reference in p and candidate in p:
            comparisons.append(bootstrap_difference(bundle,p,reference,candidate,hold))
    tables={'cv_predictions.csv':oof,'cv_fold_metrics.csv':fm,'cv_repeat_metrics.csv':rm,'cv_metrics.csv':cv,
            'cv_seed_predictions.csv':spm,'cv_seed_metrics.csv':sm,'holdout_metrics.csv':hm,
            'holdout_predictions.csv':hp,'holdout_seed_predictions.csv':hs,'holdout_comparisons.csv':pd.DataFrame(comparisons)}
    for name,table in tables.items():
        table.to_csv(out/name,index=False,float_format='%.17g')
    manifest=protocol|dict(run_signature=signature,methods=methods,device=device,development=len(dev),holdout=len(hold),
             folds=5,feature_count=9,holdout_previously_inspected=True,initialization_ensemble='mean probability',
             validation='one fixed grouped inner fifth; choose epochs/C; refit full outer training',metrics_precision='float64')
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'fold_details.json').write_text(json.dumps(dict(cv=details,holdout=info),ensure_ascii=False),encoding='utf-8')
    (out/'embeddings_manifest.json').write_text(json.dumps(embeddings['manifest'],ensure_ascii=False,indent=2),encoding='utf-8')
    return cv,hm
