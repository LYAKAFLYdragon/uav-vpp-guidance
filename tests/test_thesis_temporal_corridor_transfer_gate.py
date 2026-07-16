from __future__ import annotations
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer import summarize_records

def _record(method,opponent,key):
    window=[]
    for _ in range(20):
        receipt=None if method=='frozen_fixed_head_on' else {'projection_residual_norm_m':0.0,'hard_bound_active':False}
        window.append({'after_component_errors':[0.0]*6,'after_weighted_error':0.0,'temporal_corridor_receipt':receipt})
    return {'method':method,'opponent':opponent,'pair_key':key,'handoff_reached':True,'contract_passed':True,'window_steps':20,'handoff_state_hash':f'{opponent}:{key}','ego_failure':False,'window':window}

def test_transfer_gate_requires_two_unpooled_candidates():
    rows=[_record(method,opponent,str(i)) for method in ('frozen_fixed_head_on','moderate_left','moderate_right') for opponent in ('expert','end_to_end','independent_ppo_vpp') for i in range(4)]
    gate={'min_qualifying_pairs_per_opponent':4,'max_mean_projection_residual_m':25,'hard_bound_steps_max':0,'ego_crash_oob_delta_vs_reference_max':.05,'weighted_error_delta_vs_reference_max':-.01,'min_improved_components':2,'improved_component_delta_max':-.01,'component_worsening_delta_max':.02,'candidate_opponents_required':2}
    for row in rows:
        if row['method']=='moderate_left':
            for step in row['window']:step['after_component_errors']=[-.02,-.02,0,0,0,0];step['after_weighted_error']=-.02
    assert summarize_records(rows,scenarios_per_opponent=4,window_steps=20,gates=gate)['passes'] is True
