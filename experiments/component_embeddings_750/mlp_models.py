"""Малые FFN: общий блок, указатель типа, раздельные блоки и два порядка пула."""
from dataclasses import dataclass, asdict, replace
import numpy as np
import torch
from torch import nn

@dataclass(frozen=True)
class Config:
    name: str
    sharing: str = 'shared'
    order: str = 'after'
    pooling: str = 'mean'
    space: str = 'pca'
    bottleneck: int = 32
    activation: str = 'relu'
    use_q: bool = True
    channels: tuple = (0,1,2,3)
    q_only: bool = False

    def description(self):
        if self.q_only:
            return 'MLP 9→32→16→1 только на признаках'
        sharing={'shared':'Общий FFN','typed':'Общий FFN + тип','separate':'Четыре FFN'}[self.sharing]
        order='FFN до пула' if self.order=='before' else 'FFN после пула'
        return f'{sharing}; {order}; {self.pooling}; {self.space}; выход {self.bottleneck}; {self.activation}; признаки={self.use_q}; типы={self.channels}'

def configurations():
    result=[]
    for sharing in ('shared','typed','separate'):
        for order in ('after','before'):
            for pooling in ('mean','max'):
                result.append(Config(f'FFN_{sharing}_{order}_{pooling}',sharing,order,pooling))
    for sharing in ('shared','typed','separate'):
        base=Config('',sharing)
        result.extend([replace(base,name=f'FFN_{sharing}_noQ',use_q=False),
                       replace(base,name=f'FFN_{sharing}_out16',bottleneck=16),
                       replace(base,name=f'FFN_{sharing}_gelu',activation='gelu')])
    for sharing in ('shared','typed'):
        result.append(Config(f'FFN_{sharing}_raw',sharing,space='raw'))
    result.extend([Config('MLP_Q',q_only=True),
                   Config('FFN_shared_components',channels=(0,1,2)),
                   Config('FFN_shared_answer',channels=(3,)),
                   Config('FFN_shared_linear',activation='identity')])
    return tuple(result)

CONFIGS=configurations()
BASELINES=('LR_Q','LR_Q_MEAN','LR_Q_ANSWER','LR_Q_ALL')
METHODS=BASELINES+tuple(c.name for c in CONFIGS)

class Detector(nn.Module):
    def __init__(self,config,dimension,q_dimension=9):
        super().__init__()
        self.config=config
        if config.q_only:
            self.head=nn.Sequential(nn.Linear(q_dimension,32),nn.ReLU(),nn.Dropout(.2),
                                    nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
            return
        self.type_embedding=nn.Embedding(4,8) if config.sharing=='typed' else None
        width=dimension+(8 if self.type_embedding is not None else 0)
        hidden=256 if config.space=='raw' else 2*width
        def block():
            activation={'relu':nn.ReLU,'gelu':nn.GELU,'identity':nn.Identity}[config.activation]()
            dropout=nn.Dropout(.2) if config.activation!='identity' else nn.Identity()
            return nn.Sequential(nn.Linear(width,hidden),activation,dropout,nn.Linear(hidden,config.bottleneck))
        self.blocks=nn.ModuleList([block() for _ in range(4 if config.sharing=='separate' else 1)])
        self.head=nn.Linear(len(config.channels)*config.bottleneck+(q_dimension if config.use_q else 0),1)

    def project(self,x,types):
        if self.type_embedding is not None:
            x=torch.cat([x,self.type_embedding(types)],dim=1)
        if self.config.sharing!='separate':
            return self.blocks[0](x)
        output=x.new_zeros((len(x),self.config.bottleneck))
        for kind,block in enumerate(self.blocks):
            mask=types==kind
            if mask.any():
                output[mask]=block(x[mask])
        return output

    def forward(self,batch):
        q=batch['q']
        if self.config.q_only:
            return self.head(q).squeeze(-1)
        count=len(q)
        if self.config.order=='after':
            x=batch[self.config.pooling]
            types=torch.arange(4,device=x.device).repeat(count)
            pooled=self.project(x.reshape(count*4,-1),types).reshape(count,4,-1)
        else:
            values=self.project(batch['vectors'],batch['types'])
            segments=batch['owners']*4+batch['types']
            width=self.config.bottleneck
            if self.config.pooling=='mean':
                pooled=values.new_zeros((count*4,width)).index_add(0,segments,values)
                sizes=torch.bincount(segments,minlength=count*4).clamp_min(1)
                pooled=pooled/sizes[:,None]
            else:
                pooled=values.new_full((count*4,width),float('-inf'))
                pooled.scatter_reduce_(0,segments[:,None].expand(-1,width),values,reduce='amax',include_self=True)
                pooled=torch.where(torch.isfinite(pooled),pooled,torch.zeros_like(pooled))
            pooled=pooled.reshape(count,4,width)
        # Пустые списки не превращаются в признак за счёт смещения FFN.
        pooled=pooled*batch['present'][:,:,None]
        parts=[pooled[:,self.config.channels,:].reshape(count,-1)]
        if self.config.use_q:
            parts.insert(0,q)
        return self.head(torch.cat(parts,dim=1)).squeeze(-1)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
