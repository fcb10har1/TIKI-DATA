"""Retrieve only complete BALL FrameSets by HTTP byte ranges; ignore target attributes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np

def main():
    root=Path('runs/cd-ball-inputs')
    root.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://api.figshare.com/v2/articles/28196177',timeout=30) as response:
        metadata=json.load(response)
    manifest=json.loads(Path('data/external/idsse_processed/manifest.json').read_text())
    entries=[e for e in manifest['matches'] if e['split'] in ('train','validation')]
    def fetch(entry):
        match=entry['match']; mid=match['match_id']
        target=root/(mid+'.npz')
        if target.exists():
            print('Already cached '+mid,flush=True);return
        file=next(f for f in metadata['files'] if 'positions_raw' in f['name'] and ('DFL-MAT-'+mid+'.xml') in f['name'])
        size=file['size']; width=min(32*1024*1024,size)
        while True:
            first=size-width
            req=urllib.request.Request(file['download_url'],headers={'Range':f'bytes={first}-{size-1}'})
            with urllib.request.urlopen(req,timeout=120) as response:
                assert response.status==206 and response.headers['Content-Range']==f'bytes {first}-{size-1}/{size}'
                payload=response.read()
            assert len(payload)==width
            blocks={}
            for header in re.finditer(rb'<FrameSet\b[^>]*>',payload):
                attrs=ET.fromstring(header.group()+b'</FrameSet>').attrib
                if attrs.get('TeamId','').upper()!='BALL':continue
                period=attrs.get('GameSection')
                if period not in ('firstHalf','secondHalf'):continue
                end=payload.find(b'</FrameSet>',header.end())
                if end<0:continue
                element=ET.fromstring(payload[header.start():end+len(b'</FrameSet>')])
                assert attrs['MatchId']=='DFL-MAT-'+mid
                # Deliberately select ONLY frame number and observed planar position.
                blocks[period]=np.asarray([(int(f.attrib['N']),float(f.attrib['X']),float(f.attrib['Y'])) for f in element],dtype=np.float64)
            if set(blocks)=={'firstHalf','secondHalf'}:break
            assert width<size,'Complete ball periods not found'
            width=min(width*2,size)
        conversion=json.loads(Path(f'data/external/idsse_processed/matches/{mid}/conversion.json').read_text())
        for p,expected in zip(('firstHalf','secondHalf'),conversion['period_source_frames']):
            b=blocks[p]
            assert len(b)==expected['n_frames'] and b[0,0]==expected['start'] and b[-1,0]==expected['stop_inclusive']
            assert np.all(np.diff(b[:,0])==1)
        joined=np.concatenate([blocks['firstHalf'],blocks['secondHalf']])
        assert len(joined)==match['n_frames']
        np.savez_compressed(target,positions=joined[:,1:].astype(np.float64),raw_frames=joined[:,0].astype(np.int64),frames=np.arange(len(joined)),periods=np.asarray(match['periods']),match=np.asarray(json.dumps(match)))
        evidence={'source':file['download_url'],'source_file':file['name'],'byte_range':[first,size-1],
                  'range_sha256':hashlib.sha256(payload).hexdigest(),'frames':len(joined),'split':entry['split'],
                  'selected_source_attributes':['N','X','Y'],'target_attributes_used':False}
        (root/(mid+'.json')).write_text(json.dumps(evidence,indent=2)+'\n')
        print(f'Cached {mid}: {len(joined)} ball positions from {width/1024/1024:.1f} MB',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch,entries))


if __name__ == "__main__":
    main()
