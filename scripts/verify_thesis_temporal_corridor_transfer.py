#!/usr/bin/env python3
"""Independently recompute the temporal-corridor transfer gate from episode files."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Optional
ROOT=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_thesis_temporal_corridor_transfer as runner
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer import METHODS,OPPONENTS,SOURCE_ID,summarize_records

def verify(config_path:Path,output:Optional[Path]=None)->dict:
    d=runner._config(config_path);m=runner._load(runner._resolve(d['inputs']['manifest']));root=output or runner._resolve(d['outputs']['root']);records=[]
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in m['scenarios']:
                row=json.loads((root/'episodes'/method/opponent/f"{scenario['name']}.json").read_text(encoding='utf-8'))['record']
                if row.get('source_id')!=SOURCE_ID or row.get('method')!=method or row.get('opponent')!=opponent or row.get('pair_key')!=scenario['metadata']['pair_key']:raise RuntimeError('temporal-corridor record identity mismatch')
                for index,step in enumerate(row.get('window',[])):
                    if int(step.get('local_step',-1))!=index:raise RuntimeError('temporal-corridor local-step mismatch')
                    if method== 'frozen_fixed_head_on' and step.get('temporal_corridor_receipt') is not None:raise RuntimeError('baseline unexpectedly used corridor compiler')
                    if method!= 'frozen_fixed_head_on':
                        receipt=step.get('temporal_corridor_receipt')
                        if not isinstance(receipt,dict) or any(abs(float(x))>1.0+1e-8 for x in receipt.get('applied_action',[])):raise RuntimeError('temporal-corridor receipt invalid')
                records.append(row)
    gate=summarize_records(records,scenarios_per_opponent=len(m['scenarios']),window_steps=int(d['observability']['local_window_steps']),gates=d['gates']);reported=json.loads((root/'gate.json').read_text(encoding='utf-8'))
    if reported!={'source_id':SOURCE_ID,**gate}:raise RuntimeError('temporal-corridor reported gate mismatch')
    return {'source_id':SOURCE_ID,'records':len(records),'reported_gate_matches_recomputed':True,'passes':gate['passes']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();print(json.dumps(verify(a.config,a.output),indent=2))
