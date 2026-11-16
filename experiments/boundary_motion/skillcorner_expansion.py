"""Two external SkillCorner matches with conservative partial H/A supervision.

The publicly released provider-derived possession/phase annotations are weak
labels, not independent manually annotated H/A/D ground truth. Only agreement
inside certified in-play phases is supervised. No absence is interpreted as D.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tiki_data.contracts import MatchSpec,FeatureBatch
from tiki_data.artifacts import read_batch,write_batch
from tiki_data.p1.features import compute_frames,FEATURE_NAMES

DEST=ROOT/'runs/historical-factorial-20261116/external_expansion'
RAW='https://raw.githubusercontent.com/SkillCorner/opendata/master/'
MEDIA='https://media.githubusercontent.com/media/SkillCorner/opendata/master/'
IDS=(1874553,1886347)
RAW_KEYS=('home_positions','away_positions','home_speeds','away_speeds','ball_positions','ball_speeds')


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def get_small(url,path):
    path=Path(path)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response:body=response.read(10000000)
        assert len(body)<10000000,'Unexpectedly large metadata'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
    return path


def download(mid):
    out=DEST/'downloads'/str(mid);out.mkdir(parents=True,exist_ok=True)
    relative=f'data/matches/{mid}/'
    for suffix in ('match.json','phases_of_play.csv'):
        get_small(RAW+relative+f'{mid}_{suffix}',out/f'{mid}_{suffix}')
    pointer=get_small(RAW+relative+f'{mid}_tracking_extrapolated.jsonl',out/'tracking.lfs').read_text()
    fields=dict(line.split(' ',1) for line in pointer.splitlines())
    assert fields['version']=='https://git-lfs.github.com/spec/v1'
    expected_hash=fields['oid'].split(':')[1];expected_size=int(fields['size'])
    assert expected_size<200000000,'Bounded per-match download limit exceeded'
    target=out/'tracking.jsonl'
    if not target.exists():
        temporary=target.with_suffix('.part');digest=hashlib.sha256();size=0
        url=MEDIA+relative+f'{mid}_tracking_extrapolated.jsonl'
        with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as handle:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                size+=len(chunk);assert size<=expected_size;handle.write(chunk);digest.update(chunk)
        assert size==expected_size and digest.hexdigest()==expected_hash
        temporary.replace(target)
    assert target.stat().st_size==expected_size
    assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==expected_hash
    save_json(out/'source.json',dict(provider='SkillCorner',source=MEDIA+relative+f'{mid}_tracking_extrapolated.jsonl',
        bytes=expected_size,sha256=expected_hash,raw_metadata=RAW+relative+f'{mid}_match.json',
        phases=RAW+relative+f'{mid}_phases_of_play.csv',release=RAW+'README.md',license=RAW+'LICENSE'))
    print(f'External match {mid}: verified download ({expected_size/1e6:.1f} MB).',flush=True)
    return out


def resample_positions(positions):
    """10->25Hz; interpolate only between consecutive finite observed endpoints."""
    n=len(positions);q=np.arange(int(np.floor((n-1)*2.5))+1,dtype=float)*.4
    left=np.floor(q+1e-9).astype(int);right=np.ceil(q-1e-9).astype(int)
    alpha=(q-left).reshape((-1,)+(1,)*(positions.ndim-1))
    valid=np.isfinite(positions[left]).all(axis=-1)&np.isfinite(positions[right]).all(axis=-1)
    values=(1-alpha)*positions[left]+alpha*positions[right]
    values[~valid]=np.nan
    return values.astype(np.float32),left,right


def player_speeds(positions):
    """Planar speed from consecutive finite resampled positions; no gap bridging."""
    speed=np.full(positions.shape[:-1],np.nan,dtype=np.float32)
    good=np.isfinite(positions[1:]).all(-1)&np.isfinite(positions[:-1]).all(-1)
    value=np.linalg.norm(positions[1:]-positions[:-1],axis=-1)*25
    # Fixed physical reliability bound, independent of targets and competition.
    good &= value<=15.
    speed[1:]=np.where(good,value,np.nan)
    return speed


def phase_codes(phases,period,start,stop,home,away):
    labels=np.full(stop-start,-100,dtype=np.int64);conflict=np.zeros(len(labels),bool)
    for phase in phases:
        if int(phase['period'])!=period:continue
        team=int(float(phase['team_in_possession_id']))
        if team not in (home,away):continue
        # End-exclusive convention is conservative at the ambiguous final sample.
        a=max(start,int(float(phase['frame_start'])));b=min(stop,int(float(phase['frame_end'])))
        if b<=a:continue
        code=0 if team==home else 1;v=labels[a-start:b-start]
        conflict[a-start:b-start] |= (v>=0)&(v!=code)
        v[:]=code
    labels[conflict]=-100
    return labels


def convert(mid,source):
    out=DEST/'matches'/('SC_'+str(mid));out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists():return json.loads((out/'audit.json').read_text())
    meta=json.loads((source/f'{mid}_match.json').read_text())
    home,away=int(meta['home_team']['id']),int(meta['away_team']['id'])
    assert home!=away
    roster={side:[p['id'] for p in meta['players'] if p['team_id']==team] for side,team in [('home',home),('away',away)]}
    slots={pid:(side,j) for side,ids in roster.items() for j,pid in enumerate(ids)}
    periods=[p for p in meta['match_periods'] if p['period'] in (1,2)]
    assert len(periods)==2
    streams={}
    for p in periods:
        start,stop=int(p['start_frame']),int(p['end_frame'])+1;n=stop-start
        streams[p['period']]=dict(start=start,stop=stop,seen=np.zeros(n,bool),ball_positions=np.full((n,2),np.nan,np.float32),
            home_positions=np.full((n,len(roster['home']),2),np.nan,np.float32),away_positions=np.full((n,len(roster['away']),2),np.nan,np.float32),
            target=np.full(n,-100,np.int64))
    unknown_player=0;duplicate=0
    with (source/'tracking.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);period=row.get('period')
            if period not in streams:continue
            stream=streams[period];i=int(row['frame'])-stream['start']
            if not 0<=i<len(stream['seen']):continue
            assert not stream['seen'][i],'Duplicate provider frame'
            stream['seen'][i]=True
            ball=row.get('ball_data') or {}
            if ball.get('is_detected') is True and ball.get('x') is not None and ball.get('y') is not None:
                stream['ball_positions'][i]=(float(ball['x']),float(ball['y']))
            for player in row.get('player_data') or []:
                if player.get('is_detected') is not True:continue
                pid=player['player_id']
                if pid not in slots:unknown_player+=1;continue
                side,j=slots[pid]
                if player.get('x') is not None and player.get('y') is not None:
                    assert not np.isfinite(stream[side+'_positions'][i,j]).any(),'Duplicate player observation'
                    stream[side+'_positions'][i,j]=(float(player['x']),float(player['y']))
            group=(row.get('possession') or {}).get('group')
            if group in ('home team','away team'):stream['target'][i]=0 if group=='home team' else 1
    assert unknown_player==0,'Unmapped detected player identities'
    with (source/f'{mid}_phases_of_play.csv').open(newline='',encoding='utf-8') as handle:phases=list(csv.DictReader(handle))
    combined={k:[] for k in RAW_KEYS};targets=[];bounds=[];offset=0;source_frames=0;source_seen=0
    for period in (1,2):
        s=streams[period];source_frames+=len(s['seen']);source_seen+=int(s['seen'].sum())
        certified=phase_codes(phases,period,s['start'],s['stop'],home,away)
        target=np.where((certified>=0)&(certified==s['target'])&s['seen'],certified,-100)
        values={}
        for key in ('home_positions','away_positions','ball_positions'):
            values[key],left,right=resample_positions(s[key])
        n=len(left)
        for side in ('home','away'):values[side+'_speeds']=player_speeds(values[side+'_positions'])
        values['ball_speeds']=np.full(n,np.nan,np.float32)
        # Targets must agree across interpolation endpoints; uncertain transitions stay X.
        target25=np.where((target[left]>=0)&(target[left]==target[right]),target[left],-100)
        targets.append(target25)
        for key in RAW_KEYS:combined[key].append(values[key])
        bounds.append((offset,offset+n));offset+=n
    raw={k:np.concatenate(v) for k,v in combined.items()};codes=np.concatenate(targets)
    spec=MatchSpec('SC_'+str(mid),offset,25,tuple(bounds))
    # Features receive geometry/velocity arrays only; never possession or phase fields.
    values=np.zeros((offset,len(FEATURE_NAMES)),np.float32);observed=np.zeros_like(values,dtype=bool)
    for a,b in spec.periods:
        values[a:b],observed[a:b]=compute_frames(raw['home_positions'][a:b],raw['away_positions'][a:b],
            raw['home_speeds'][a:b],raw['away_speeds'][a:b],raw['ball_positions'][a:b],raw['ball_speeds'][a:b])
    batch=FeatureBatch(spec,np.arange(offset),values,observed,FEATURE_NAMES,spec.period_ids()).validate(spec)
    assert np.sum(codes>=0)>offset*.2,'Insufficient certified live-possession coverage'
    assert set(np.unique(codes))<= {-100,0,1}
    if (out/'features.npz').exists():
        previous=read_batch(out/'features.npz',spec,'features')
        assert np.array_equal(previous.values,batch.values) and np.array_equal(previous.observed,batch.observed),'Interrupted conversion differs'
    else:write_batch(out/'features.npz',batch)
    np.save(out/'targets.npy',codes)
    raw_out=out/'raw_arrays';raw_out.mkdir(exist_ok=True)
    for key,array in raw.items():np.save(raw_out/(key+'.npy'),array)
    save_json(out/'match.json',spec.as_dict())
    audit=dict(match=spec.as_dict(),split='train',source_match=mid,provider='SkillCorner',
        source_frames=source_frames,source_seen=source_seen,frames_25hz=offset,
        supervised_frames=int(np.sum(codes>=0)),home_frames=int(np.sum(codes==0)),away_frames=int(np.sum(codes==1)),
        dead_frames=0,unknown_frames=int(np.sum(codes<0)),
        ball_available_fraction=float(np.isfinite(raw['ball_positions']).all(1).mean()),
        supervised_fraction=float(np.mean(codes>=0)),
        label_kind='Provider-derived weak partial H/A labels; no D labels',
        label_policy='H/A only when in-play phase team and tracking possession agree at both interpolation endpoints. All uncertain frames X.',
        feature_policy='Detected coordinates only; interpolate only between consecutive finite10Hz observations; observed player planar speed from consecutive resampled points, <=15m/s; ball speed masked. No targets used as features.',
        resampling='Half-separated10Hz to25Hz; no observation interpolation across gaps or half boundaries.',
        source=json.loads((source/'source.json').read_text()))
    save_json(out/'audit.json',audit)
    print(json.dumps(dict(match=spec.match_id,supervised_fraction=audit['supervised_fraction'],ball_available_fraction=audit['ball_available_fraction'],frames=offset)),flush=True)
    return audit


def load_expansion():
    """Return ([(FeatureBatch, integer target codes)], {match_id: raw arrays})."""
    records=[];raw_by_id={}
    for mid in IDS:
        folder=DEST/'matches'/('SC_'+str(mid))
        spec=MatchSpec.from_dict(json.loads((folder/'match.json').read_text()))
        records.append((read_batch(folder/'features.npz',spec,'features'),np.load(folder/'targets.npy')))
        raw_by_id[spec.match_id]={k:np.load(folder/'raw_arrays'/(k+'.npy'),mmap_mode='r') for k in RAW_KEYS}
    return records,raw_by_id


def self_test():
    p=np.asarray([[0.,0.],[1.,0.],[np.nan,np.nan],[3.,0.]])
    out,left,right=resample_positions(p)
    assert np.allclose(out[:3,0],[0,.4,.8])
    assert not np.isfinite(out[3:]).any(),'No interpolation or extrapolation through missing observations'
    phases=[dict(period='1',team_in_possession_id='1',frame_start='0',frame_end='3'),
            dict(period='1',team_in_possession_id='2',frame_start='2',frame_end='4')]
    assert np.array_equal(phase_codes(phases,1,0,5,1,2),[0,0,-100,1,-100])
    speed=player_speeds(np.asarray([[[0.,0.]],[[.04,0.]],[[np.nan,np.nan]],[[.12,0.]]]))
    assert np.isclose(speed[1,0],1) and np.isnan(speed[2:,0]).all()


def main():
    self_test();DEST.mkdir(parents=True,exist_ok=True)
    get_small(RAW+'LICENSE',DEST/'LICENSE.txt');get_small(RAW+'README.md',DEST/'SOURCE_README.txt')
    audits=[convert(mid,download(mid)) for mid in IDS]
    save_json(DEST/'manifest.json',dict(kind='external',provider='SkillCorner',match_ids=['SC_'+str(mid) for mid in IDS],
        selection='First two numerical IDs with tracking, match metadata and phases available; fixed before model training.',
        actual_new_matches=2,weak_supervision=True,full_had_ground_truth=False,test_matches_consumed=False,
        attribution='SkillCorner Open Data; credit SkillCorner. Public release README and MIT license retained locally.',matches=audits))
    print('Two external matches prepared; H/A partial supervision only, uncertain/dead frames unlabelled.',flush=True)


if __name__=='__main__':main()
