"""Преобразования обучаются только на заданных обучающих ответах."""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data_io import TYPES

class Store:
    def __init__(self,bundle,embeddings):
        self.bundle=bundle
        self.raw=np.asarray(embeddings['context'],dtype=np.float32)
        self.answers=np.asarray(embeddings['answer_cls'],dtype=np.float32)
        self.dimension=self.raw.shape[1]
        n=len(bundle.ids)
        self.indices=[[[] for _ in range(4)] for _ in range(n)]
        lookup={sid:i for i,sid in enumerate(bundle.ids)}
        for j,row in enumerate(bundle.components):
            self.indices[lookup[int(row['source_id'])]][TYPES.index(row['component_type'])].append(j)
        for i in range(n):
            self.indices[i][3]=[len(self.raw)+i]
        self.mean=np.zeros((n,4,self.dimension),np.float32)
        self.maximum=np.zeros_like(self.mean)
        self.present=np.zeros((n,4),bool)
        for i in range(n):
            for kind in range(3):
                indices=self.indices[i][kind]
                if indices:
                    self.present[i,kind]=True
                    self.mean[i,kind]=self.raw[indices].mean(0)
                    self.maximum[i,kind]=self.raw[indices].max(0)
        self.mean[:,3]=self.maximum[:,3]=self.answers
        self.present[:,3]=True

    def fit(self,train,pca_dimension=128,include_raw=True):
        train=np.asarray(train,int)
        # Общая PCA: каждый непустой тип каждого ответа даёт ровно один вектор.
        fit=self.mean[train][self.present[train]]
        dim=min(pca_dimension,len(fit)-1,self.dimension)
        pca=PCA(dim,svd_solver='randomized',random_state=42).fit(fit)
        scaler=StandardScaler().fit(pca.transform(fit))
        q_scaler=StandardScaler().fit(self.bundle.q[train])
        def transform(x):
            shape=x.shape
            return scaler.transform(pca.transform(x.reshape(-1,self.dimension))).astype(np.float32).reshape(*shape[:-1],dim)
        means=transform(self.mean); maximum=transform(self.maximum)
        means[~self.present]=0; maximum[~self.present]=0
        # Линейные преобразования выполняются до сети; FFN обучается по меткам ответов.
        vectors=np.concatenate([transform(self.raw),transform(self.answers)],axis=0)
        view=View(q_scaler.transform(self.bundle.q).astype(np.float32),means,maximum,
                  vectors,self.indices,self.present,train,dim)
        raw=None; raw_scaler=None
        if include_raw:
            raw_scaler=StandardScaler().fit(fit)
            raw_mean=raw_scaler.transform(self.mean.reshape(-1,self.dimension)).reshape(self.mean.shape).astype(np.float32)
            raw_max=raw_scaler.transform(self.maximum.reshape(-1,self.dimension)).reshape(self.maximum.shape).astype(np.float32)
            raw_mean[~self.present]=0; raw_max[~self.present]=0
            raw=View(view.q,raw_mean,raw_max,None,self.indices,self.present,train,self.dimension)
        return {'pca':view,'raw':raw,'preprocessing':dict(pca=pca,pca_scaler=scaler,
                q_scaler=q_scaler,raw_scaler=raw_scaler,fit_ids=[self.bundle.ids[i] for i in train])}

class View:
    def __init__(self,q,mean,maximum,vectors,indices,present,fit_indices,dimension):
        self.q=q; self.mean=mean; self.maximum=maximum; self.vectors=vectors
        self.indices=indices; self.present=present
        self.fit_indices=np.asarray(fit_indices).copy(); self.dimension=dimension

    def batch(self,rows,config,device):
        rows=np.asarray(rows,int)
        values={'q':self.q[rows],'mean':self.mean[rows],'max':self.maximum[rows],
                'present':self.present[rows].astype(np.float32)}
        if not config.q_only and config.order=='before':
            if self.vectors is None:
                raise ValueError('Сырые векторы поддерживаются только после пула.')
            positions=[]; owners=[]; types=[]
            for owner,row in enumerate(rows):
                for kind in range(4):
                    indices=self.indices[row][kind]
                    positions.extend(indices); owners.extend([owner]*len(indices)); types.extend([kind]*len(indices))
            values.update(vectors=self.vectors[positions],owners=np.asarray(owners,np.int64),types=np.asarray(types,np.int64))
        return {k:torch.as_tensor(v,device=device) for k,v in values.items()}

    def linear(self,name):
        if name=='LR_Q':
            return self.q
        if name=='LR_Q_MEAN':
            return np.column_stack([self.q,self.mean[:,:3].reshape(len(self.q),-1)])
        if name=='LR_Q_ANSWER':
            return np.column_stack([self.q,self.mean[:,3]])
        if name=='LR_Q_ALL':
            return np.column_stack([self.q,self.mean.reshape(len(self.q),-1)])
        raise ValueError(name)
