#!/usr/bin/env python3
"""V21 non-learning collector for the one-way warm-up anchor protocol."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_thesis_five_state_v17_path_context_preflight as v17
from v21_path_context_episode import V21PathContextEpisode
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder

SOURCE_ID='THESIS-FIVE-STATE-V21-WARMUP-ANCHOR-PROTOCOL'
OPPONENTS=('expert','end_to_end','independent_ppo_vpp')

def _load_config(path):
 d=v17._load(path)
 if 'base_config' not in d:return d
 base=v17._resolve(d['base_config'])
 if v17._hash(base)!=d.get('base_config_sha256'):raise RuntimeError('V21 authorization base hash mismatch')
 result=copy.deepcopy(v17._load(base));result['authorization']=copy.deepcopy(d.get('authorization',{}));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();d=_load_config(a.config)
 m=v17._load(v17._resolve(d['inputs']['manifest']));out=v17._resolve(d['outputs']['root'])
 pre={'source_id':SOURCE_ID,'planned_records':72,'output_root_absent':not out.exists(),'execution_permitted':bool(d['authorization']['execution_permitted'])}
 if not a.execute: print(json.dumps(pre,indent=2));return
 if not pre['output_root_absent'] or not pre['execution_permitted']: raise RuntimeError('V21 execution is not authorized or output exists')
 runtime=copy.deepcopy(v17._load(v17._resolve(d['inputs']['runtime_template'])));runtime['outputs']['root']=str(out);registry=v17._load(v17._resolve(d['inputs']['runtime_registry']));out.mkdir(parents=True);rows=[]
 for opponent in OPPONENTS:
  for scenario in m['scenarios']:
   env=v17.v16.v15.v14.build_record_only_env(runtime,registry,opponent);policy=engine._specialist(registry,'frozen_fixed_head_on');encoder=FrozenP3Encoder(Path(runtime['inputs']['frozen_p3_encoder']),runtime['inputs']['frozen_p3_encoder_sha256']);ep=V21PathContextEpisode(env,policy,encoder,runtime,v17.v16._compiler(runtime))
   try:r=ep.run(scenario,int(scenario['metadata']['scenario_seed']))
   finally:ep.close()
   row={'source_id':SOURCE_ID,'opponent':opponent,'pair_key':scenario['metadata']['pair_key'],**r};rows.append(row);v17._write_json(out/'episodes'/opponent/f"{scenario['name']}.json",{'record':row})
 matrix={o:{'records':sum(x['opponent']==o for x in rows),'released':sum(x['opponent']==o and x['v21_anchor_released'] for x in rows),'post_release_steps':sum(x['opponent']==o and x['v21_anchor_released']*x['window_steps'] for x in rows)} for o in OPPONENTS}
 gate={'source_id':SOURCE_ID,'records':len(rows),'matrix':matrix,'passes':len(rows)==72 and all(x['released']>=2 and x['post_release_steps']>=20 for x in matrix.values())};v17._write_json(out/'gate.json',gate);print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
