"""Frozen Kalman/observation-age implementation for the selected ensemble."""
import numpy as np
from numba import njit
from tiki_data.contracts import FeatureBatch, require
from tiki_data.p1.features import FEATURE_NAMES
from tiki_data.p1.kalman import DEFAULT_SETTINGS

AGE_CAP = 10.0
KF_PARAMETERS = np.asarray(list(DEFAULT_SETTINGS.values()), dtype=np.float64)
SELECTED_NAMES = ('ball_kalman_speed_mps',) + FEATURE_NAMES[1:] + ('ball_seconds_since_observed', 'ball_kalman_velocity_std_mps')

def age_seconds(visible, fps, initial=0.0):
    """Causal age; initial means elapsed time just before the first input frame.

    Before any observation, age counts from the start of the period. No future
    observation is used. Capping at 10 seconds is a fixed, unfitted definition.
    """
    t = np.arange(len(visible), dtype=np.float64)
    last = np.maximum.accumulate(np.where(visible, t, -1.0 - initial * fps))
    return np.minimum((t - last) / fps, AGE_CAP).astype(np.float32)


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


def build_features(base, ball_positions):
    """Accept shared 19-feature P1 inputs and aligned metric x/y observations.

    Missing raw positions must be NaN. Recompute P1 geometry after any raw
    corruption before calling this function. Filter state resets per half.
    """
    base.validate(base.match)
    require(base.feature_names == FEATURE_NAMES, "expected original 19 P1 features")
    require(base.match.fps == 25, "selected checkpoints require 25 Hz inputs")
    positions = np.asarray(ball_positions, dtype=np.float64)
    require(positions.shape == (base.match.n_frames, 2), "ball positions must align with P1 frames")
    values = np.zeros((base.match.n_frames, 21), np.float32)
    observed = np.zeros_like(values, dtype=bool)
    for a, b in base.match.periods:
        visible = np.isfinite(positions[a:b]).all(axis=1)
        motion, valid, _ = kalman_positions(positions[a:b], base.match.fps, a,
                                           initial_kalman(), KF_PARAMETERS)
        valid &= visible
        values[a:b, :19] = base.values[a:b]
        observed[a:b, :19] = base.observed[a:b]
        values[a:b, 0] = np.where(valid, motion[:, 0], 0)
        observed[a:b, 0] = valid
        values[a:b, 19] = age_seconds(visible, base.match.fps)
        observed[a:b, 19] = True
        values[a:b, 20] = np.where(valid, motion[:, 1], 0)
        observed[a:b, 20] = valid
    return FeatureBatch(base.match, base.frames.copy(), values, observed,
                        SELECTED_NAMES, base.periods.copy()).validate(base.match)
