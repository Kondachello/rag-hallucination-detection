"""Один шаг всех архитектур, формы, пустые группы и короткая вложенная проверка."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from mlp_models import CONFIGS, Detector, seed_everything, parameter_count
from mlp_data import Store
from mlp_experiment import run_split, TRAINING
from data_io import load_bundle
from compact_experiment import group_splits
from threadpoolctl import threadpool_limits

def check_architectures():
    torch.set_num_threads(2)
    owners=torch.tensor([0,0,0,1,1,1,1])
    types=torch.tensor([0,0,3,0,1,2,3])
    for config in CONFIGS:
        seed_everything(1)
        width=4096 if config.space=='raw' else 128
        batch=dict(q=torch.randn(2,9),mean=torch.randn(2,4,width),max=torch.randn(2,4,width),
                   present=torch.tensor([[1,0,0,1],[1,1,1,1]],dtype=torch.float32),
                   vectors=torch.randn(7,width),owners=owners,types=types)
        model=Detector(config,width)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        logits=model(batch)
        assert logits.shape==(2,) and torch.isfinite(logits).all()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        model.eval()
        if config.order=='before':
            permutation=torch.tensor([6,1,4,2,0,5,3])
            changed=batch|{k:batch[k][permutation] for k in ('vectors','owners','types')}
            assert torch.allclose(model(batch),model(changed),atol=1e-6,rtol=1e-6)
        if not config.q_only and 1 in config.channels:
            with torch.no_grad():
                model.head.weight.zero_(); model.head.bias.zero_()
                offset=(9 if config.use_q else 0)+config.channels.index(1)*config.bottleneck
                model.head.weight[:,offset:offset+config.bottleneck]=1
                assert abs(model(batch)[0].item())<1e-7
        print('OK',config.name,'parameters',parameter_count(model))

def check_pipeline():
    bundle=load_bundle(); rng=np.random.default_rng(42)
    embeddings=dict(context=rng.normal(size=(len(bundle.components),32)).astype(np.float32),
                    answer_cls=rng.normal(size=(len(bundle.ids),32)).astype(np.float32))
    store=Store(bundle,embeddings)
    dev=np.where(bundle.partitions=='development')[0]
    train,test=next(group_splits(bundle,dev,42))
    # Небольшое число целых групп; каждая нейросеть делает ровно один шаг на этап.
    fit_groups=np.unique(bundle.groups[train])[:120]
    test_groups=np.unique(bundle.groups[test])[:35]
    train=train[np.isin(bundle.groups[train],fit_groups)]
    test=test[np.isin(bundle.groups[test],test_groups)]
    configs=tuple(c for c in CONFIGS if c.name in ('FFN_shared_after_mean','FFN_typed_before_max','FFN_shared_raw'))
    settings=TRAINING|dict(max_epochs=1,min_epochs=1,patience=1,batch_size=1024)
    with TemporaryDirectory() as temp,threadpool_limits(limits=2):
        results,_,details=run_split(bundle,store,train,test,42,Path(temp),'smoke','smoke',
                      configs=configs,seeds=(11,),settings=settings,pca_dimension=16,device='cpu')
        assert len(results)==7 and all(len(p)==len(test) for p in results.values())
        import mlp_experiment
        original=mlp_experiment.train_network
        def forbidden(*args,**kwargs): raise AssertionError('Повторное обучение вместо кэша')
        mlp_experiment.train_network=forbidden
        try:
            restored,_,_=run_split(bundle,store,train,test,42,Path(temp),'smoke','smoke',
                      configs=configs,seeds=(11,),settings=settings,pca_dimension=16,device='cpu')
            assert all(np.array_equal(results[m],restored[m]) for m in results)
        finally:
            mlp_experiment.train_network=original
    print('OK nested preprocessing, group boundaries, one step, resume')

if __name__=='__main__':
    check_architectures()
    check_pipeline()
