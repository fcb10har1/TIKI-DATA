"""Matched TCN initialization and an explicitly conditional live/team head."""
from common import *
import torch
from torch import nn
from torch.nn import functional as F
from tiki_data.p2.model import TCN

class TwoHeadTCN(nn.Module):
    def __init__(self,n_features,model_config):
        super().__init__();original=TCN(n_features,**model_config)
        self.radius=original.radius;self.trunk=nn.Sequential(*list(original.network.children())[:-1])
        width=model_config['width'];self.live=nn.Conv1d(width,2,1);self.team=nn.Conv1d(width,2,1)
    def components(self,x):
        h=self.trunk(x);return self.live(h),self.team(h)
    @staticmethod
    def joint(live,team):
        lp=F.log_softmax(live,dim=1);tp=F.log_softmax(team,dim=1)
        return torch.stack((lp[:,1]+tp[:,0],lp[:,1]+tp[:,1],lp[:,0]),dim=1)
    def forward(self,x):return self.joint(*self.components(x))

def fresh_model(n_features,config,architecture='tcn',policy='recorded'):
    # Reproduce the baseline's original 19-feature initialization, then add age
    # at zero weight. Preserve the RNG state for comparable dropout sequences.
    old=TCN(19,**config);rng=torch.get_rng_state();reference=TCN(20,**config)
    state=old.state_dict();new=reference.state_dict()
    for key in state:
        if key=='network.0.weight':
            new[key].zero_();new[key][:,:19]=state[key][:,:19];new[key][:,20:39]=state[key][:,19:38]
        else:new[key].copy_(state[key])
    reference.load_state_dict(new)
    if architecture=='boundary':
        model=BoundaryTCN(n_features,config)
        matched=TCN(n_features,**config);target=matched.state_dict();origin=reference.state_dict()
        for key in origin:
            if key=='network.0.weight':
                target[key].zero_();target[key][:,:20]=origin[key][:,:20];target[key][:,n_features:n_features+20]=origin[key][:,20:40]
            else:target[key].copy_(origin[key])
        model.load_state_dict(target,strict=False)
    elif architecture=='two_head':
        model=TwoHeadTCN(n_features,config)
        model.trunk.load_state_dict(nn.Sequential(*list(reference.network.children())[:-1]).state_dict())
        last=reference.network[-1]
        with torch.no_grad():
            model.live.weight[0]=last.weight[2];model.live.bias[0]=last.bias[2]
            model.live.weight[1]=last.weight[:2].mean(0);model.live.bias[1]=last.bias[:2].mean()
            model.team.weight.copy_(last.weight[:2]);model.team.bias.copy_(last.bias[:2])
    elif n_features==20:model=reference
    else:
        model=TCN(n_features,**config);target=model.state_dict();origin=reference.state_dict()
        for key in origin:
            if key=='network.0.weight':
                target[key].zero_()
                if policy=='removed':target[key].copy_(origin[key][:,list(range(1,20))+list(range(21,40))])
                else:target[key][:,:20]=origin[key][:,:20];target[key][:,n_features:n_features+20]=origin[key][:,20:40]
            else:target[key].copy_(origin[key])
        model.load_state_dict(target)
    torch.set_rng_state(rng)
    return model

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    nf=len(saved['normalization']['feature_names'])
    model=(BoundaryTCN(nf,saved['model_config']) if saved.get('architecture')=='boundary' else TwoHeadTCN(nf,saved['model_config']) if saved.get('architecture')=='two_head' else TCN(nf,**saved['model_config']))
    model.load_state_dict(saved['state_dict'],strict=True)
    return model.eval(),saved

def classification_loss(model,x,y,radius,core):
    if isinstance(model,BoundaryTCN):
        logits,boundary=model.components(x);logits=logits[:,:,radius:radius+core];boundary=boundary[:,:,radius:radius+core]
        loss=F.cross_entropy(logits,y,ignore_index=-100,reduction='none')
        target,valid=boundary_targets(y)
        aux=F.binary_cross_entropy_with_logits(boundary,target,reduction='none',pos_weight=torch.full((1,3,1),20.,device=x.device)).mean(1)
        return loss+.1*aux*valid,logits
    if isinstance(model,TwoHeadTCN):
        live,team=model.components(x);live=live[:,:,radius:radius+core];team=team[:,:,radius:radius+core]
        alive=torch.where(y>=0,(y<2).long(),-100)
        team_target=torch.where((y>=0)&(y<2),y,-100)
        losses=F.cross_entropy(live,alive,ignore_index=-100,reduction='none')+F.cross_entropy(team,team_target,ignore_index=-100,reduction='none')
        return losses,model.joint(live,team)
    logits=model(x)[:,:,radius:radius+core]
    return F.cross_entropy(logits,y,ignore_index=-100,reduction='none'),logits

def consistency_penalty(logits,y):
    p=torch.softmax(logits,dim=1)
    eligible=(y[:,1:]==y[:,:-1])&(y[:,1:]>=0)&(y[:,:-1]>=0)
    difference=(p[:,:,1:]-p[:,:,:-1]).abs().sum(dim=1)
    return difference[eligible].mean() if bool(eligible.any()) else logits.sum()*0.


class BoundaryTCN(TCN):
    def __init__(self,n_features,config):
        super().__init__(n_features,**config)
        self.boundary=nn.Conv1d(config['width'],3,1)
    def components(self,x):
        h=x
        for layer in list(self.network.children())[:-1]:h=layer(h)
        return self.network[-1](h),self.boundary(h)
    def forward(self,x):return self.components(x)[0]

def boundary_targets(y):
    target=torch.zeros((y.shape[0],3,y.shape[1]),device=y.device)
    valid=torch.zeros_like(y,dtype=torch.bool)
    a,b=y[:,:-1],y[:,1:];valid[:,1:]=(a>=0)&(b>=0)
    target[:,0,1:]=((a<2)&(b<2)&(a!=b)&valid[:,1:]).float()
    target[:,1,1:]=((a<2)&(b==2)&valid[:,1:]).float()
    target[:,2,1:]=((a==2)&(b<2)&valid[:,1:]).float()
    return target,valid
