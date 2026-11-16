"""Provider-independent speed alternatives, after raw-observation corruption."""
from common import *
from numba import njit
from raw_features import features,raw_slice,age_seconds,sample_occlusion,schema_with_age
from tiki_data.contracts import FeatureBatch
from tiki_data.p1.kalman import DEFAULT_SETTINGS
from tiki_data.p2.data import encode,timeline_blocks
import torch
from torch.utils.data import Dataset
import hashlib

KF_KEYS=tuple(DEFAULT_SETTINGS)
KF_PARAMETERS=np.asarray([DEFAULT_SETTINGS[k] for k in KF_KEYS],dtype=np.float64)

@njit(cache=True)
def kalman_positions(positions,fps,offset,initial,parameters):
    """Same causal filter as P1, with explicit state for clean-prefix warm starts."""
    measurement,acceleration,initial_std,max_std,max_gap,gate,min_observations=parameters
    dt=1./fps;q=acceleration**2;r=measurement**2
    q00=q*dt**4/4;q01=q*dt**3/2;q11=q*dt**2
    state=initial.copy();n=len(positions)
    values=np.zeros((n,2),np.float32);valid=np.zeros(n,np.bool_);snapshots=np.zeros((n,10))
    for j in range(n):
        t=offset+j;x,y=positions[j,0],positions[j,1]
        observed=np.isfinite(x) and np.isfinite(y)
        if state[0]>0:
            if (t-state[8])*dt>max_gap:state[0]=0;state[9]=0
            else:
                state[1]+=state[3]*dt;state[2]+=state[4]*dt
                p00,p01,p11=state[5],state[6],state[7]
                state[5]=p00+2*dt*p01+dt*dt*p11+q00
                state[6]=p01+dt*p11+q01;state[7]=p11+q11
        if observed:
            if state[0]>0:
                dx=x-state[1];dy=y-state[2]
                if (dx*dx+dy*dy)/(state[5]+r)>gate:state[0]=0;state[9]=0
            if state[0]==0:
                state[0]=1;state[1]=x;state[2]=y;state[3]=0;state[4]=0
                state[5]=r;state[6]=0;state[7]=initial_std**2
            else:
                p00,p01,p11=state[5],state[6],state[7]
                k0=p00/(p00+r);k1=p01/(p00+r)
                state[1]+=k0*dx;state[2]+=k0*dy;state[3]+=k1*dx;state[4]+=k1*dy
                state[5]=max(0.,(1-k0)*p00);state[6]=(1-k0)*p01;state[7]=max(0.,p11-k1*p01)
            state[9]+=1;state[8]=t
        if state[0]>0 and state[9]>=min_observations:
            std=np.sqrt(state[7])
            if std<=max_std:
                values[j,0]=np.sqrt(state[3]**2+state[4]**2);values[j,1]=std;valid[j]=True
        snapshots[j]=state
    return values,valid,snapshots

def initial_kalman():
    state=np.zeros(10);state[8]=-1
    return state

def base_schema(policy):
    value=schema_with_age()
    if policy=='removed':
        value['features']=value['features'][1:]
        value['ball_dependent']=[x for x in value['ball_dependent'] if x!='ball_speed_mps']
    elif policy in ('difference','kalman'):
        name='ball_difference_speed_mps' if policy=='difference' else 'ball_kalman_speed_mps'
        definition=('Causal planar speed from consecutive finite ball coordinates at 25 Hz; no gap bridging'
            if policy=='difference' else 'Causal Kalman speed estimate, available only on directly observed position frames after warm-up')
        value['features'][0]=dict(name=name,definition=definition,sources=['ball.positions','frame_time'])
        value['ball_dependent']=[name if x=='ball_speed_mps' else x for x in value['ball_dependent']]
        if policy=='kalman':
            value['features'].append(dict(name='ball_kalman_velocity_std_mps',
                definition='Per-axis uncertainty of a valid causal velocity estimate on an observed ball-position frame',sources=['ball.positions','frame_time']))
            value['ball_dependent'].append('ball_kalman_velocity_std_mps')
            value['notes']+=' Kalman estimates are explicitly derived; extrapolated speeds are not exposed as observations. Fixed settings: '+json.dumps(DEFAULT_SETTINGS,sort_keys=True)
    elif policy!='recorded':raise ValueError('Unknown speed policy')
    return value

def speed_transform(values,observed,positions,policy,fps,offset,previous=None,initial=None):
    """Identical transform for training windows and complete provider timelines."""
    if policy=='recorded':return values,observed
    if policy=='removed':return values[:,1:],observed[:,1:]
    values=values.copy();observed=observed.copy()
    visible=np.isfinite(positions).all(axis=1)
    if policy=='difference':
        prev=np.vstack((np.full((1,2),np.nan) if previous is None else np.asarray(previous).reshape(1,2),positions[:-1]))
        valid=visible&np.isfinite(prev).all(axis=1)
        speed=np.linalg.norm(positions-prev,axis=1)*fps
        values[:,0]=np.where(valid,speed,0);observed[:,0]=valid
    else:
        motion,valid,_=kalman_positions(np.asarray(positions,dtype=np.float64),fps,offset,
            initial_kalman() if initial is None else initial,KF_PARAMETERS)
        valid=valid&visible # never mark extrapolated estimates as directly observed.
        values[:,0]=np.where(valid,motion[:,0],0);observed[:,0]=valid
        values=np.column_stack((values,np.where(valid,motion[:,1],0))).astype(np.float32)
        observed=np.column_stack((observed,valid))
    return values,observed

def transform_batch(base,positions,policy):
    """Public inference interface: base contains the shared 19 P1 inputs plus age.

    Works with any provider after metric-coordinate conversion. This function
    never receives targets; no competition input is opened by this study.
    """
    names=tuple(x['name'] for x in schema(policy)['features'])
    values=np.zeros((base.match.n_frames,len(names)),np.float32);observed=np.zeros_like(values,dtype=bool)
    for a,b in base.match.periods:
        values[a:b],observed[a:b]=speed_transform(base.values[a:b],base.observed[a:b],positions[a:b],policy,base.match.fps,a)
    return FeatureBatch(base.match,base.frames,values,observed,names,base.periods).validate(base.match)

class BaseFactory:
    def __init__(self,base,raw,policy):
        self.base,self.raw,self.policy=base,raw,policy
        self.age=np.zeros(base.match.n_frames,np.float32);self.prefix=None
        for a,b in base.match.periods:self.age[a:b]=age_seconds(np.isfinite(raw['ball_positions'][a:b]).all(axis=1),base.match.fps)
        if policy=='kalman':
            directory=OUT/'feature_cache/kalman_prefix';directory.mkdir(parents=True,exist_ok=True)
            fingerprint=hashlib.sha256(np.asarray(raw['ball_positions']).tobytes()).hexdigest()[:16]
            path=directory/(base.match.match_id+'_'+fingerprint+'.npy')
            if not path.exists():
                state=np.zeros((base.match.n_frames,10))
                for a,b in base.match.periods:
                    _,_,state[a:b]=kalman_positions(np.asarray(raw['ball_positions'][a:b],dtype=np.float64),base.match.fps,a,initial_kalman(),KF_PARAMETERS)
                np.save(path,state)
            self.prefix=np.load(path,mmap_mode='r')

    def clean(self):
        base=FeatureBatch(self.base.match,self.base.frames,
            np.column_stack((self.base.values,self.age)).astype(np.float32),
            np.column_stack((self.base.observed,self.base.periods>0)),
            tuple(x['name'] for x in schema('recorded')['features']),self.base.periods)
        return transform_batch(base,self.raw['ball_positions'],self.policy)

    def window(self,a,b,left,missing,no_speed):
        initial_age=float(self.age[a-1]) if a>left else 0.
        values,observed,_=features(raw_slice(self.raw,a,b),self.base.match.fps,missing,no_speed,True,initial_age)
        positions=np.array(self.raw['ball_positions'][a:b],dtype=np.float64,copy=True);positions[missing]=np.nan
        previous=self.raw['ball_positions'][a-1] if a>left else None
        initial=self.prefix[a-1].copy() if self.prefix is not None and a>left else None
        return speed_transform(values,observed,positions,self.policy,self.base.match.fps,a,previous,initial)

    def corrupted(self,missing,no_speed):
        values=[];observed=[]
        names=tuple(x['name'] for x in schema(self.policy)['features'])
        v=np.zeros((self.base.match.n_frames,len(names)),np.float32);o=np.zeros_like(v,dtype=bool)
        for a,b in self.base.match.periods:v[a:b],o[a:b]=self.window(a,b,a,missing[a:b],no_speed)
        return FeatureBatch(self.base.match,self.base.frames,v,o,names,self.base.periods).validate(self.base.match)

class TrainingWindows(Dataset):
    def __init__(self,records,factories,normalization,core,radius,augmentation,seed,boundary_radius=0,boundary_weight=1.):
        self.records,self.factories,self.norm=records,factories,normalization
        self.core,self.radius,self.augmentation,self.seed,self.epoch=core,radius,augmentation,seed,0
        self.index=[];self.loss_weights=[]
        for i,(base,y) in enumerate(records):
            w=np.ones(len(y),np.float32)
            for left,right in timeline_blocks(base):
                if boundary_radius:
                    changes=np.flatnonzero((y[left+1:right]!=y[left:right-1])&(y[left+1:right]>=0)&(y[left:right-1]>=0))+left+1
                    for t in changes:w[max(left,t-boundary_radius):min(right,t+boundary_radius)]=boundary_weight
                for start in range(left,right,core):
                    stop=min(start+core,right)
                    if np.any(y[start:stop]>=0):self.index.append((i,left,right,start,stop))
            self.loss_weights.append(w)

    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        i,left,right,start,stop=self.index[index];base,labels=self.records[i]
        a,b=max(left,start-self.radius),min(right,start+self.core+self.radius)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        missing,no_speed=sample_occlusion(b-a,rng,self.augmentation)
        values,observed=self.factories[i].window(a,b,left,missing,no_speed)
        x=np.zeros((2*values.shape[1],self.core+2*self.radius),np.float32)
        offset=a-(start-self.radius);x[:,offset:offset+b-a]=encode(values,observed,self.norm)
        y=np.full(self.core,-100,np.int64);y[:stop-start]=labels[start:stop]
        weights=np.zeros(self.core,np.float32);weights[:stop-start]=self.loss_weights[i][start:stop]
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(weights)


MOTION_NAMES=tuple(side+'_'+name for side in ('home','away') for name in ('reliable_distance_m','closing_mps','relative_speed_mps','direction_alignment','observation_age_s'))+('home_away_distance_margin_m','home_away_closing_margin_mps','home_away_relative_speed_margin_mps')

def schema(policy):
    result=base_schema('kalman' if policy=='relative' else policy)
    if policy=='relative':
        result['features'] += [dict(name=n,definition='Identity-continuous nearest reliable player/ball motion aggregate; unavailable observations masked',sources=['home.positions','away.positions','ball.positions','frame_time']) for n in MOTION_NAMES]
        result['ball_dependent']+=list(MOTION_NAMES)
    return result

class Factory(BaseFactory):
    def __init__(self,base,raw,policy):
        self.relative=policy=='relative'
        super().__init__(base,raw,'kalman' if self.relative else policy)
    def motion(self,a,b,left,missing,values,observed):
        positions=np.array(self.raw['ball_positions'][a:b],dtype=float,copy=True);positions[missing]=np.nan
        initial=self.prefix[a-1].copy() if a>left else initial_kalman()
        _,valid,states=kalman_positions(positions,self.base.match.fps,a,initial,KF_PARAMETERS)
        ball_velocity=states[:,3:5];ball_valid=valid&np.isfinite(positions).all(1)&(np.linalg.norm(ball_velocity,axis=1)<=60)
        n=b-a;extras=np.zeros((n,13),np.float32);masks=np.zeros((n,13),bool)
        for side,offset in [('home',0),('away',5)]:
            p=np.asarray(self.raw[side+'_positions'][a:b],dtype=float)
            prev=np.concatenate((np.asarray(self.raw[side+'_positions'][a-1:a],dtype=float) if a>left else np.full_like(p[:1],np.nan),p[:-1]))
            velocity=(p-prev)*self.base.match.fps
            good=np.isfinite(p).all(2)&np.isfinite(prev).all(2)&(np.linalg.norm(velocity,axis=2)<=15)&ball_valid[:,None]
            displacement=positions[:,None,:]-p;distance=np.linalg.norm(displacement,axis=2)
            idx=np.argmin(np.where(good,distance,np.inf),axis=1);rows=np.arange(n);usable=good[rows,idx]
            d=distance[rows,idx];v=velocity[rows,idx];delta=displacement[rows,idx]
            rel=ball_velocity-v;closing=-np.sum(rel*delta,axis=1)/np.maximum(d,1e-8)
            denom=np.linalg.norm(v,axis=1)*np.linalg.norm(ball_velocity,axis=1)
            alignment=np.sum(v*ball_velocity,axis=1)/np.maximum(denom,1e-8)
            block=np.column_stack((d,closing,np.linalg.norm(rel,axis=1),alignment,np.zeros(n)))
            extras[:,offset:offset+5]=np.where(usable[:,None],block,0)
            masks[:,offset:offset+5]=usable[:,None];masks[:,offset+3]&=denom>1e-8
            extras[~masks]=0
        for col,leftcol,rightcol in [(10,0,5),(11,1,6),(12,2,7)]:
            ok=masks[:,leftcol]&masks[:,rightcol];masks[:,col]=ok;extras[:,col]=np.where(ok,extras[:,leftcol]-extras[:,rightcol],0)
        return np.column_stack((values,extras)),np.column_stack((observed,masks))
    def window(self,a,b,left,missing,no_speed):
        v,o=super().window(a,b,left,missing,no_speed)
        return self.motion(a,b,left,missing,v,o) if self.relative else (v,o)
    def clean(self):
        if not self.relative:return super().clean()
        return self.corrupted(np.zeros(self.base.match.n_frames,bool),False)
    def corrupted(self,missing,no_speed):
        if not self.relative:return super().corrupted(missing,no_speed)
        names=tuple(x['name'] for x in schema('relative')['features'])
        v=np.zeros((self.base.match.n_frames,len(names)),np.float32);o=np.zeros_like(v,dtype=bool)
        for a,b in self.base.match.periods:v[a:b],o[a:b]=self.window(a,b,a,missing[a:b],no_speed)
        return FeatureBatch(self.base.match,self.base.frames,v,o,names,self.base.periods).validate(self.base.match)
