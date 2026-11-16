from common import *
import torch
from models import boundary_targets,fresh_model,classification_loss
from boundary import decode_boundary
from decoder import decode,BASE
from feature_engine import Factory
from raw_features import batch_from_raw

def test_boundary_targets_exclude_unknown_and_core_start():
    y=torch.tensor([[0,1,2,0,-100,1]])
    t,v=boundary_targets(y)
    assert v.tolist()==[[False,True,True,True,False,False]]
    assert t[0,:,1:4].tolist()==[[1,0,0],[0,1,0],[0,0,1]]

def test_disabled_boundary_equals_frozen_decoder():
    rng=np.random.default_rng(3);spec=MatchSpec('external-test',500,25,((0,250),(250,500)))
    p=rng.dirichlet([1,1,1],500);visible=rng.random(500)>.2;b=rng.random((500,3))
    np.testing.assert_array_equal(decode(p,visible,spec,BASE),decode_boundary(p,visible,spec,BASE,b,0.))

def fixture():
    n=120;t=np.arange(n)*.04
    hp=np.stack([np.column_stack((t+k,np.zeros(n))) for k in (1,3)],1).astype('f')
    return dict(home_positions=hp,away_positions=hp+5,home_speeds=np.ones((n,2),'f'),away_speeds=np.ones((n,2),'f'),ball_positions=np.column_stack((2*t,np.zeros(n))).astype('f'),ball_speeds=np.full(n,2,'f'))

def test_motion_permutation_missing_and_window_state():
    raw=fixture();spec=MatchSpec('external-motion',120,25);base,_=batch_from_raw(raw,spec)
    f=Factory(base,raw,'relative');hidden=np.zeros(120,bool);hidden[40:55]=True
    a=f.corrupted(hidden,False)
    swapped=dict(raw,home_positions=raw['home_positions'][:,::-1],away_positions=raw['away_positions'][:,::-1])
    b=Factory(base,swapped,'relative').corrupted(hidden,False)
    np.testing.assert_array_equal(a.values,b.values)
    assert not a.observed[40:55,21:].any()
    clean=f.clean();v,o=f.window(30,100,0,np.zeros(70,bool),False)
    np.testing.assert_array_equal(clean.values[30:100],v)
    poison={k:v.copy() for k,v in raw.items()};poison['ball_positions'][hidden]=1e9
    other=Factory(base,poison,'relative').corrupted(hidden,False)
    np.testing.assert_array_equal(a.values,other.values)

def test_boundary_head_backprop():
    torch.manual_seed(42);cfg=dict(width=8,kernel=3,dilations=[1,2],dropout=.1)
    model=fresh_model(21,cfg,'boundary','kalman');core=20
    x=torch.randn(2,42,core+2*model.radius);y=torch.zeros((2,core),dtype=torch.long);y[:,10:]=2
    loss,_=classification_loss(model,x,y,model.radius,core);loss.mean().backward()
    assert model.boundary.weight.grad.abs().sum()>0
