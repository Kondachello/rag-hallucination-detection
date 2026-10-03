"""Один ограниченный проход моделей; веса кодировщика не скачиваются."""
import numpy as np
from threadpoolctl import threadpool_limits
from data_io import load_bundle
from embeddings import make_jobs
from compact_experiment import METHODS, Representations, group_splits, run_split

def run():
    b=load_bundle(); rng=np.random.default_rng(42)
    jobs,_=make_jobs(b)
    assert len(jobs)==len(b.components)+len(b.ids)
    assert set(b.splits[b.splits.partition=='holdout'].seen_in_pilot)=={0}
    dev=np.where(b.partitions=='development')[0]
    for seed in (42,143,244):
        coverage=[]
        for train,test in group_splits(b,dev,seed):
            assert not (set(b.groups[train])&set(b.groups[test]))
            coverage.extend(test)
        assert sorted(coverage)==sorted(dev)
    embeddings={'context':rng.normal(size=(len(b.components),32)).astype(np.float32),
                'answer_cls':rng.normal(size=(len(b.ids),32)).astype(np.float32)}
    train,test=next(group_splits(b,dev,42))
    train=rng.choice(train,100,replace=False); test=test[:30]
    with threadpool_limits(limits=2):
        store=Representations(b,embeddings)
        matrices=store.build(train)
        assert matrices['Q'].shape==(750,9)
        assert matrices['Q_MEAN'].shape==(750,33)
        assert matrices['Q_E'].shape==(750,17)
        predictions,_=run_split(b,store,train,test,42,METHODS)
    assert len(predictions)==32
    assert all(p.shape==(len(test),) and np.isfinite(p).all() for p in predictions.values())
    print('OK: данные, 3 групповых разбиения, 32 метода; только один короткий проход')

if __name__=='__main__': run()
