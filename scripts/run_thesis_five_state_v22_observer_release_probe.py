#!/usr/bin/env python3
"""V22 observer-only predictor release collector."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_thesis_five_state_v17_path_context_preflight as v17
from uav_vpp_guidance.common.provenance import record_config_override
from uav_vpp_guidance.evaluation.v22_observer_release_protocol import V22ObserverReleaseMixin
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
SOURCE_ID='THESIS-FIVE-STATE-V22-OBSERVER-RELEASE-PROBE';OPPONENTS=('expert','end_to_end','independent_ppo_vpp')
def _load_config(path):
 d=v17._load(path)
 if 'base_config' not in d:return d
 base=v17._resolve(d['base_config'])
 if v17._hash(base)!=d.get('base_config_sha256'):raise RuntimeError('V22 authorization base hash mismatch')
 result=copy.deepcopy(v17._load(base));result['authorization']=copy.deepcopy(d.get('authorization',{}));return result
class Episode(V22ObserverReleaseMixin,v17.PathContextEpisode):
 def run(self,s,seed): self._v21_reset_anchor_protocol();r=super().run(s,seed);r['anchor_switches']=self._v21_switches;r['released']=self._v21_released;return r
 def _observe(self,o): super()._observe(o);self._v22_after_observation()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();d=_load_config(a.config);m=v17._load(v17._resolve(d['inputs']['manifest']));out=v17._resolve(d['outputs']['root']);pre={'source_id':SOURCE_ID,'planned_records':72,'output_root_absent':not out.exists(),'execution_permitted':bool(d['authorization']['execution_permitted'])}
 if not a.execute:print(json.dumps(pre,indent=2));return
 if not pre['output_root_absent'] or not pre['execution_permitted']:raise RuntimeError('V22 not authorized')
 runtime=copy.deepcopy(v17._load(v17._resolve(d['inputs']['runtime_template'])));old=runtime['trajectory_prediction']['integration'].get('observer_only_warmup');runtime['trajectory_prediction']['integration']['observer_only_warmup']=True;record_config_override(runtime,'trajectory_prediction.integration.observer_only_warmup',True,old,'V22 observer-only release')
 registry=v17._load(v17._resolve(d['inputs']['runtime_registry']));out.mkdir(parents=True);rows=[]
 for opp in OPPONENTS:
  for s in m['scenarios']:
   env=v17.v16.v15.v14.build_record_only_env(runtime,registry,opp);pol=engine._specialist(registry,'frozen_fixed_head_on');enc=FrozenP3Encoder(Path(runtime['inputs']['frozen_p3_encoder']),runtime['inputs']['frozen_p3_encoder_sha256']);ep=Episode(env,pol,enc,runtime,v17.v16._compiler(runtime))
   try:r=ep.run(s,int(s['metadata']['scenario_seed']))
   finally:ep.close()
   row={'source_id':SOURCE_ID,'opponent':opp,'pair_key':s['metadata']['pair_key'],**r};rows.append(row);v17._write_json(out/'episodes'/opp/f"{s['name']}.json",{'record':row})
 matrix={o:{'records':sum(x['opponent']==o for x in rows),'released':sum(x['opponent']==o and x['released'] for x in rows),'post_release_steps':sum(x['window_steps'] for x in rows if x['opponent']==o and x['released'])}for o in OPPONENTS};gate={'source_id':SOURCE_ID,'records':len(rows),'matrix':matrix,'passes':len(rows)==72 and all(x['released']>=2 and x['post_release_steps']>=20 for x in matrix.values())};v17._write_json(out/'gate.json',gate);print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
