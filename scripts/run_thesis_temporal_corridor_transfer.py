#!/usr/bin/env python3
"""Run the preregistered non-learning temporal-corridor transfer probe."""
from __future__ import annotations
import argparse, copy, json, subprocess, sys
from pathlib import Path
from typing import Any, Mapping, Optional
import numpy as np

ROOT=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_thesis_five_state_reentry_v16_dynamic_reference_path as v16
from uav_vpp_guidance.envs.observation import compute_relative_geometry
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer import METHODS, OPPONENTS, REFERENCE_METHOD, SOURCE_ID, is_ego_failure, summarize_records
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.virtual_point.temporal_corridor import TemporalCorridorConfig, compile_temporal_corridor_action

def _load(path:Path)->dict: return v16._load(path)
def _resolve(value:str)->Path: return v16._resolve(value)
def _hash(path:Path)->str: return v16._hash(path)
def _write(path:Path,value:Mapping[str,Any])->None: v16._write_json(path,value)

def _config(path:Path)->dict:
    overlay=_load(path)
    if 'base_config' not in overlay:return overlay
    base=_resolve(overlay['base_config'])
    if _hash(base)!=overlay.get('base_config_sha256'):raise RuntimeError('temporal-corridor authorization base hash mismatch')
    result=copy.deepcopy(_load(base));result['authorization']=copy.deepcopy(overlay.get('authorization',{}));result['_execution']=copy.deepcopy(overlay.get('execution',{}));return result

def validate_design(path:Path,allow_output:bool=False)->dict:
    d=_config(path)
    if d.get('source_id')!=SOURCE_ID or tuple(d.get('opponents',{}).get('order',()))!=OPPONENTS or d['opponents'].get('pooled_gate')!='prohibited':raise RuntimeError('temporal-corridor source or opponent contract drifted')
    if tuple(d.get('methods',{}))!=METHODS:raise RuntimeError('temporal-corridor method contract drifted')
    for key in ('training_permitted','tuning_permitted','reward_change_permitted','candidate_policy_loading_permitted','high_level_ppo_training_permitted','four_skill_training_permitted','combat_finetune_permitted'):
        if d['authorization'].get(key) is not False:raise RuntimeError(f'temporal-corridor must prohibit {key}')
    for key in ('manifest','runtime_template','runtime_registry','frozen_p3_encoder'):
        p=_resolve(d['inputs'][key]); expected=d['inputs'].get(key+'_sha256')
        if not p.is_file() or (expected and _hash(p)!=expected):raise RuntimeError(f'temporal-corridor frozen input mismatch: {key}')
    manifest=_load(_resolve(d['inputs']['manifest']))
    if manifest.get('source_id')!=SOURCE_ID+'-MANIFEST12' or len(manifest.get('scenarios',[]))!=12:raise RuntimeError('temporal-corridor manifest contract drifted')
    out=_resolve(d['outputs']['root'])
    if not out.is_absolute() or (out.exists() and not allow_output):raise RuntimeError('temporal-corridor output root must be absolute and absent')
    return {'source_id':SOURCE_ID,'planned_records':len(METHODS)*len(OPPONENTS)*len(manifest['scenarios']),'output_root_absent':not out.exists(),'execution_permitted':bool(d['authorization'].get('execution_permitted'))}

def _authorize(d:Mapping[str,Any],expected:int)->None:
    e=d.get('_execution',{})
    if d['authorization'].get('execution_permitted') is not True or int(e.get('expected_records',-1))!=expected:raise RuntimeError('temporal-corridor execution is not authorized')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if sha!=str(e.get('required_implementation_git_sha') or ''):raise RuntimeError('temporal-corridor implementation SHA is not exactly authorized')
    files=e.get('authorized_code_files')
    if not isinstance(files,list) or not files:raise RuntimeError('temporal-corridor authorized file list missing')
    for item in files:
        p=_resolve(item['path'])
        if not p.is_file() or _hash(p)!=item['sha256']:raise RuntimeError(f'temporal-corridor authorized code mismatch: {p}')

class Episode(v16.DynamicReferencePathEpisode):
    def _context(self):
        own,target=self.env._get_current_states(noisy=True);target=self.env._task_get_target_state(target,own);own,target=self.env._task_pre_step(own,target)
        if not self.env._sync_prediction_buffer_for_current_state(own,target,compute_relative_geometry(own,target)):raise RuntimeError('temporal-corridor predictor synchronization failed')
        predicted,_,info=self.env.trajectory_predictor_adapter.predict(target)
        predicted=np.asarray(predicted,dtype=np.float64).reshape(-1)
        if predicted.shape!=(3,) or not np.isfinite(predicted).all() or bool(info.get('fallback',False)):raise RuntimeError('temporal-corridor prediction fallback')
        return own,target,predicted
    def run(self,scenario:Mapping[str,Any],seed:int,method:str,compiler:Optional[TemporalCorridorConfig]):
        self.history.clear();self.history_features.reset();self.observation=None;self.base=None;self.last_info={};self.total_steps=0;self.phase_trace=[]
        event=v16.v15.v14.v11.RawSIFirstPassConfig(**dict(self.config['observability']['raw_si_first_pass_event']));self.phase_tracker=v16.v15.v14.v11.RawSIReentryPhaseTracker(v16.v15.v14.v11.RawSIReentryPhaseConfig(first_pass=event))
        observation=self.env.reset(scenario=copy.deepcopy(dict(scenario)),seed=int(seed))
        if self.env._backend!='jsbsim':raise RuntimeError('temporal-corridor requires strict JSBSim')
        self._assert_path_interface();self.env.set_runtime_specialist_context(specialist_key='head_on',specialist_profile=None,specialist_mode_name='tc_frozen_run_in',specialist_reason='tc_run_in');self.observation_schema=copy.deepcopy(observation['observation_schema']);self._observe(observation)
        done=False;terminal='horizon_before_handoff'
        while not done and self.total_steps<int(self.config['observability']['max_high_level_steps']):
            if self._trigger_ready():break
            info,done=self._reference_step();terminal=str(info.get('reason') or 'timeout')
        if not self._trigger_ready():return {'handoff_reached':False,'window_steps':0,'contract_passed':not bool(self.last_info.get('backend_fallback_occurred',False)),'terminal_reason':terminal,'ego_failure':is_ego_failure(terminal),'phase_trace':self.phase_trace}
        handoff_hash=v16.v15.v14._stable_hash({'base':self.base,'history':tuple(self.history),'phase':self._phase()});window=[];previous=np.zeros(3)
        self.env.set_runtime_specialist_context(specialist_key='head_on',specialist_profile='temporal_corridor' if compiler else None,specialist_mode_name='tc_window',specialist_reason=method)
        for local in range(int(self.config['observability']['local_window_steps'])):
            profile='range_extension' if local<10 else 'reentry_preparation';before,errors,weighted=self._snapshot(profile);reference=v16.v15.v14.v11.v9.finite_action(self.policy.get_deterministic_action(self.observation['observation_vector']));receipt=None
            if compiler is None:action=reference
            else:
                own,target,predicted=self._context();receipt=compile_temporal_corridor_action(own_position_neu=own['position_neu'],predicted_target_position_neu=predicted,phase_step=local,previous_action=previous,raw_action_ranges=self.config['raw_action_ranges'],config=compiler);action=np.asarray(receipt['applied_action']);previous=action.copy()
            observation,_,terminated,truncated,info=self.env.step(np.asarray(action,dtype=np.float32));self.total_steps+=1;self.last_info=info;self._observe(observation)
            if bool(info.get('backend_fallback_occurred',False)) or info.get('prediction_fallback') is True:raise RuntimeError('temporal-corridor runtime fallback')
            after,after_errors,after_weighted=self._snapshot(profile);own_after=info.get('own_state',{})
            window.append({'local_step':local,'objective_profile':profile,'before_component_errors':errors,'after_component_errors':after_errors,'before_weighted_error':weighted,'after_weighted_error':after_weighted,'temporal_corridor_receipt':receipt,'applied_action':np.asarray(action).tolist(),'env_vpp_world_offset_m':self._finite_vector(info.get('vp_world_offset'),'vp_world_offset'),'guidance_command_normalized':self._guidance_vector(info),'actual_nz_g':float(own_after.get('nz_g',info.get('nz_g'))),'prediction_valid':info.get('prediction_valid') is True,'prediction_fallback':bool(info.get('prediction_fallback',False))})
            done=bool(terminated or truncated or self.total_steps>=int(self.config['observability']['max_high_level_steps']))
            terminal=str(info.get('reason') or 'timeout')
            if done:break
        while not done and self.total_steps<int(self.config['observability']['max_high_level_steps']):info,done=self._reference_step();terminal=str(info.get('reason') or 'timeout')
        finite=all(np.isfinite(np.asarray(s['after_component_errors'],dtype=float)).all() and np.isfinite(float(s['after_weighted_error'])) for s in window)
        return {'handoff_reached':True,'handoff_state_hash':handoff_hash,'window_steps':len(window),'terminal_reason':terminal,'ego_failure':is_ego_failure(terminal),'contract_passed':finite,'window':window,'phase_trace':self.phase_trace}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();d=_config(a.config);pre=validate_design(a.config)
    if not a.execute:print(json.dumps(pre,indent=2));return
    _authorize(d,int(pre['planned_records']))
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('temporal-corridor requires clean worktree')
    manifest=_load(_resolve(d['inputs']['manifest']));runtime=copy.deepcopy(_load(_resolve(d['inputs']['runtime_template'])));registry=_load(_resolve(d['inputs']['runtime_registry']));out=_resolve(d['outputs']['root']);out.mkdir(parents=True);_write(out/'authorized_config.json',{'config':d,'config_sha256':_hash(a.config)});records=[]
    try:
        for method in METHODS:
            compiler=None if method==REFERENCE_METHOD else TemporalCorridorConfig(**d['methods'][method]['compiler'])
            for opponent in OPPONENTS:
                for scenario in manifest['scenarios']:
                    env=v16.v15.v14.build_record_only_env(runtime,registry,opponent);policy=engine._specialist(registry,'frozen_fixed_head_on');encoder=FrozenP3Encoder(Path(d['inputs']['frozen_p3_encoder']),d['inputs']['frozen_p3_encoder_sha256']);ep=Episode(env,policy,encoder,d,v16._compiler(d))
                    try:result=ep.run(scenario,int(scenario['metadata']['scenario_seed']),method,compiler)
                    finally:ep.close()
                    row={'source_id':SOURCE_ID,'method':method,'opponent':opponent,'pair_key':scenario['metadata']['pair_key'],**result};records.append(row);_write(out/'episodes'/method/opponent/f"{scenario['name']}.json",{'record':row})
        gate=summarize_records(records,scenarios_per_opponent=len(manifest['scenarios']),window_steps=int(d['observability']['local_window_steps']),gates=d['gates']);_write(out/'gate.json',{'source_id':SOURCE_ID,**gate});_write(out/'run_manifest.json',{'source_id':SOURCE_ID,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config_sha256':_hash(a.config),'manifest_sha256':_hash(_resolve(d['inputs']['manifest'])),'records':len(records),'training':'prohibited','reward_recording':'prohibited','authorized_code_files':d['_execution']['authorized_code_files']});print(json.dumps(gate,indent=2))
    except Exception as error:_write(out/'failure.json',{'source_id':SOURCE_ID,'error':repr(error),'completed_records':len(records)});raise
if __name__=='__main__':main()
