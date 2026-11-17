"""Frozen segment proposer and its external-training objective."""
import numpy as np
from evaluation import possession_metrics as pm
import torch
from torch import nn
from torch.nn import functional as F
from tiki_data.p2.model import TCN
from tiki_data.p2.data import make_window, timeline_blocks


class SegmentTCN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.backbone=TCN(19,**config)
        self.radius=self.backbone.radius
        self.boundary=nn.Conv1d(config['width'],1,1)
        self.segment=nn.Sequential(nn.Linear(3*config['width']+4,64),nn.ReLU(),
                                   nn.Dropout(.1),nn.Linear(64,4))

    def components(self,x):
        h=x
        for layer in list(self.backbone.network.children())[:-1]:h=layer(h)
        return h,self.backbone.network[-1](h),self.boundary(h).squeeze(1)

    def intervals(self,h,logits,batch,starts,stops):
        """Pool interval interiors/endpoints in O(number of proposals)."""
        prefix=F.pad(h.cumsum(-1),(1,0));prob=F.pad(logits.softmax(1).cumsum(-1),(1,0))
        length=(stops-starts).to(h.dtype)
        pooled=(prefix[batch,:,stops]-prefix[batch,:,starts])/length[:,None]
        evidence=(prob[batch,:,stops]-prob[batch,:,starts])/length[:,None]
        features=torch.cat((pooled,h[batch,:,starts],h[batch,:,stops-1],evidence,
                            torch.log1p(length[:,None]/25.)),1)
        return self.segment(features)


def known_runs(labels):
    return pm.valid_runs(np.where(labels>=0,labels,3))


def complete_intervals(labels):
    return [(a,b,c) for left,right in known_runs(labels)
            for a,b,c in pm.segments(labels,[(left,right)]) if a>left and b<right]


def training_proposals(labels,rng,maximum=36):
    """Complete GT, shifted, merged, split and model-like random intervals.

    Targets are assigned by best temporal IoU with complete known training
    intervals. Border-censored and unknown-touching segments are excluded.
    """
    complete=complete_intervals(labels)
    if not complete:return []
    short=[s for s in complete if s[1]-s[0]<=75]
    chosen=[]
    for _ in range(maximum//3):
        pool=short if short and rng.random()<.5 else complete
        a,b,c=pool[int(rng.integers(len(pool)))];chosen.append((a,b))
        da,db=rng.integers(-50,51,size=2)
        u,v=max(0,a+int(da)),min(len(labels),b+int(db))
        if u<v and (labels[u:v]>=0).all():chosen.append((u,v))
    for _ in range(maximum):
        a,b,c=complete[int(rng.integers(len(complete)))];mode=int(rng.integers(3))
        if mode==0:
            u,v=a,min(len(labels),b+int(rng.integers(5,101)))
        elif mode==1:
            u,v=a,a+max(1,int((b-a)*rng.uniform(.1,.6)))
        else:
            u=int(rng.integers(max(1,len(labels)-2)));v=min(len(labels),u+int(rng.integers(3,251)))
        if u<v and (labels[u:v]>=0).all():chosen.append((u,v))
        if len(chosen)>=maximum:break
    result=[]
    for a,b in chosen[:maximum]:
        matches=[(max(0,min(b,v)-max(a,u))/(max(b,v)-min(a,u)),c) for u,v,c in complete]
        overlap,c=max(matches,key=lambda x:x[0])
        if overlap>=.5:target=int(c)
        elif overlap<.3:target=3
        else:continue
        result.append((a,b,target))
    return result


def objective(model,x,y,radius,core,rng):
    h,logits,boundary=model.components(x)
    h=h[:,:,radius:radius+core];logits=logits[:,:,radius:radius+core]
    boundary=boundary[:,radius:radius+core]
    frame=F.cross_entropy(logits,y,ignore_index=-100)
    valid=(y[:,1:]>=0)&(y[:,:-1]>=0)
    change=(y[:,1:]!=y[:,:-1])&valid
    target=torch.zeros_like(boundary);target[:,1:]=change.to(boundary.dtype)
    # Three-frame triangular targets preserve exact transition supervision while
    # giving a nonzero gradient near the correct cut.
    center=target.clone()
    target[:,1:]=torch.maximum(target[:,1:],.5*center[:,:-1])
    target[:,:-1]=torch.maximum(target[:,:-1],.5*center[:,1:])
    mask=torch.zeros_like(y,dtype=torch.bool);mask[:,1:]=valid
    bce=F.binary_cross_entropy_with_logits(boundary,target,reduction='none')
    weights=1+49*target
    boundary_loss=(bce*weights*mask).sum()/mask.sum().clamp_min(1)
    proposals=[]
    for i,labels in enumerate(y.detach().cpu().numpy()):
        proposals += [(i,a,b,c) for a,b,c in training_proposals(labels,rng)]
        # Include current model run intervals, so the head also sees imperfect
        # predictions rather than just GT-centred examples.
        pred=logits[i].detach().argmax(0).cpu().numpy()
        true=complete_intervals(labels)
        for a,b,c in pm.segments(pred,known_runs(labels))[:12]:
            matches=[(max(0,min(b,v)-max(a,u))/(max(b,v)-min(a,u)),d) for u,v,d in true]
            if matches:
                overlap,d=max(matches,key=lambda x:x[0])
                if overlap>=.5:proposals.append((i,a,b,int(d)))
                elif overlap<.3:proposals.append((i,a,b,3))
    if proposals:
        t=torch.tensor(proposals,device=x.device,dtype=torch.long)
        segment_loss=F.cross_entropy(model.intervals(h,logits,t[:,0],t[:,1],t[:,2]),t[:,3])
    else:segment_loss=frame*0
    return frame+.2*boundary_loss+.5*segment_loss,dict(frame=float(frame.detach()),
        boundary=float(boundary_loss.detach()),segment=float(segment_loss.detach()),proposals=len(proposals))


@torch.inference_mode()
def infer(model,features,norm,core=512):
    n=features.match.n_frames;w=model.backbone.network[0].out_channels
    h=np.empty((n,w),np.float32);logits=np.empty((n,3),np.float64);boundary=np.empty(n,np.float64)
    model.eval()
    for left,right in timeline_blocks(features,include_outside=True):
        for a in range(left,right,core):
            b=min(a+core,right)
            x=make_window(features,norm,left,right,a,core,model.radius)
            z,p,q=model.components(torch.from_numpy(x[None]))
            h[a:b]=z[0,:,model.radius:model.radius+b-a].T.numpy()
            logits[a:b]=p[0,:,model.radius:model.radius+b-a].T.double().numpy()
            boundary[a:b]=q[0,model.radius:model.radius+b-a].double().sigmoid().numpy()
    return h,logits,boundary


def peaks(probability,threshold):
    values=np.asarray(probability,dtype=np.float32)
    z=torch.from_numpy(values)[None,None]
    maximum=F.max_pool1d(z,7,stride=1,padding=3)[0,0].numpy()
    ids=np.flatnonzero((values>=threshold)&(values>=maximum))
    return ids[np.r_[True,np.diff(ids)>1]] if len(ids) else ids


def timing_refine(baseline,boundary,spec,maximum_seconds):
    out=baseline.copy();limit=int(round(maximum_seconds*spec.fps))
    for left,right in spec.periods:
        labels=baseline[left:right];p=boundary[left:right]
        cuts=np.flatnonzero(labels[1:]!=labels[:-1])+1;positions=cuts.copy()
        for i,cut in enumerate(cuts):
            a=max(1,int(cut)-limit,int(positions[i-1])+1 if i else 1)
            b=min(len(labels)-1,int(cut)+limit,int(cuts[i+1])-1 if i+1<len(cuts) else len(labels)-1)
            if a<=b:
                best=a+int(p[a:b+1].argmax())
                if p[best]>=.25 and p[best]>p[cut]+.05:positions[i]=best
        old=np.r_[0,cuts,len(labels)];new=np.r_[0,positions,len(labels)]
        assert (np.diff(new)>0).all() and np.all(np.abs(positions-cuts)<=limit)
        for a,b,c in zip(new[:-1],new[1:],labels[old[:-1]]):out[left+a:left+b]=c
    return out


@torch.inference_mode()
def segment_decode(model,cache,baseline,spec,threshold=.3,beta=3.,transition_cost=1.):
    """Partition each period over learned proposals; no target/count/timestamp rules."""
    h,logits,boundary=cache
    result=np.full(spec.n_frames,'X',dtype='<U1')
    segment_counts=0
    for left,right in spec.periods:
        n=right-left;bp=boundary[left:right];base=baseline[left:right]
        basecuts=np.flatnonzero(base[1:]!=base[:-1])+1
        cuts=peaks(bp,threshold);cuts=cuts[(cuts>0)&(cuts<n)]
        nodes=np.unique(np.r_[0,basecuts,cuts,n]).astype(np.int64)
        index={int(v):i for i,v in enumerate(nodes)}
        edges=set()
        for i,a in enumerate(nodes[:-1]):
            for step in (1,2,3,4,6,8,12,16,24,32):
                j=i+step
                if j<len(nodes) and (nodes[j]-a<=1500 or step==1):edges.add((i,j))
            edges.add((i,len(nodes)-1))
        old=np.r_[0,basecuts,n]
        for a,b in zip(old[:-1],old[1:]):edges.add((index[int(a)],index[int(b)]))
        edges=sorted(edges,key=lambda x:(x[1],x[0]))
        ei=np.asarray([a for a,b in edges]);ej=np.asarray([b for a,b in edges])
        starts=nodes[ei];stops=nodes[ej]
        ht=torch.from_numpy(h[left:right].T[None].copy())
        lt=torch.from_numpy(logits[left:right].T[None].astype(np.float32))
        q=[]
        for a in range(0,len(edges),4096):
            s=torch.from_numpy(starts[a:a+4096]);t=torch.from_numpy(stops[a:a+4096])
            q.append(model.intervals(ht,lt,torch.zeros_like(s),s,t).double().log_softmax(1).numpy())
        q=np.concatenate(q)
        logp=logits[left:right]/4.;logp-=np.logaddexp.reduce(logp,axis=1)[:,None]
        cumulative=np.vstack((np.zeros(3),logp.cumsum(0)))
        evidence=cumulative[stops]-cumulative[starts]
        quality=np.clip(q[:,:3]-q[:,3,None],-6,6)
        scores=evidence+beta*quality
        # A transition gets evidence from its learned boundary probability. The
        # penalty is fixed/selected externally, never derived from target counts.
        boundary_score=np.log(np.clip(bp[np.minimum(starts,n-1)],1e-4,1-1e-4)/
                              np.clip(1-bp[np.minimum(starts,n-1)],1e-4,1))
        scores += np.where(starts[:,None]>0,boundary_score[:,None]-transition_cost,0.)
        best=np.full((len(nodes),3),-np.inf);back=np.full((len(nodes),3,2),-1,np.int32)
        for k,(i,j) in enumerate(edges):
            for c in range(3):
                if i==0:value=scores[k,c];previous=-1
                else:
                    options=best[i].copy();options[c]=-np.inf
                    previous=int(options.argmax());value=options[previous]+scores[k,c]
                if value>best[j,c]:best[j,c]=value;back[j,c]=[i,previous]
        j=len(nodes)-1;c=int(best[j].argmax())
        assert np.isfinite(best[j,c]),'No segment partition'
        while j>0:
            i,previous=back[j,c];assert 0<=i<j
            result[left+nodes[i]:left+nodes[j]]=('H','A','D')[c]
            segment_counts+=1;j,c=int(i),int(previous)
    return result,dict(proposed_output_segments=segment_counts)
