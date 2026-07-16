#!/usr/bin/env python3
"""Independently recompute V22's per-opponent release gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(ROOT/'scripts')]
import run_thesis_five_state_v22_observer_release_probe as runner

def verify(config:Path,output:Path|None=None):
 d=runner._load_config(config);root=output or runner.v17._resolve(d['outputs']['root']);m=runner.v17._load(runner.v17._resolve(d['inputs']['manifest']));rows=[]
 for o in runner.OPPONENTS:
  for s in m['scenarios']:
   row=json.loads((root/'episodes'/o/f"{s['name']}.json").read_text(encoding='utf-8'))['record']
   if row['source_id']!=runner.SOURCE_ID or row['opponent']!=o or row['pair_key']!=s['metadata']['pair_key']:raise RuntimeError('V22 record identity mismatch')
   if any(x.get('mode')=='predicted_target' and x.get('reason')!='v22_first_valid_observer_receipt' for x in row['anchor_switches']):raise RuntimeError('V22 invalid release receipt')
   rows.append(row)
 matrix={o:{'records':sum(x['opponent']==o for x in rows),'released':sum(x['opponent']==o and x['released'] for x in rows),'post_release_steps':sum(x['window_steps'] for x in rows if x['opponent']==o and x['released'])}for o in runner.OPPONENTS};gate={'source_id':runner.SOURCE_ID,'records':len(rows),'matrix':matrix,'passes':len(rows)==72 and all(x['released']>=2 and x['post_release_steps']>=20 for x in matrix.values())}
 if json.loads((root/'gate.json').read_text(encoding='utf-8'))!=gate:raise RuntimeError('V22 reported gate mismatch')
 return {'source_id':runner.SOURCE_ID,'records':len(rows),'reported_gate_matches_recomputed':True,'passes':gate['passes']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--write',action='store_true');a=p.parse_args();r=verify(a.config,a.output);print(json.dumps(r,indent=2))
