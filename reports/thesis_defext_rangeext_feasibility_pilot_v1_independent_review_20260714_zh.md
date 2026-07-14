# Defensive-Extension Feasibility Pilot V1 独立预注册复核

**复核结论：** `PASS`

本复核只检查设计、资产、分割、门槛和 stop rule；未启动训练、JSBSim evaluation 或 output root。通过也不构成执行授权。

| Check | Result | Detail |
|---|---|---|
| `source_id` | PASS | THESIS-DEFEXT-RANGEEXT-FEASIBILITY-PILOT-V1 |
| `execution_locked` | PASS | {'training_permitted': False, 'pilot_execution_permitted': False, 'baseline_evaluation_permitted': False, 'heldout_evaluation_permitted': False, 'high_level_ppo_training_permitted': False, 'four_skill_training_permitted': False, 'combat_finetune_permitted': False} |
| `implementation_commit_exists` | PASS | 8cec8f0c0b3cac1c50adbfc1581b06d45df09d8e |
| `implementation_source_hashes` | PASS | 12/12 matched |
| `dirty_run_not_promoted` | PASS | {'status': 'research_only_not_paper_safe', 'gate_path': 'E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/single_motif_continuous_66d_r1/analysis_r1/single_motif_continuous_66d_gate.json', 'gate_sha256_at_freeze': '2135aecb8bcbdc812637e4cd1fd4d30de9d02809f02601d0dc516522d37b6dcf', 'promotion_to_paper_safe': False} |
| `p3_frozen_hash` | PASS | 385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59 |
| `fixed_skill_profile` | PASS | defensive_extension/range_extension |
| `routing_disabled` | PASS | {'skill': 'defensive_extension', 'profile': 'range_extension', 'routing_enabled': False, 'high_level_policy_present': False, 'observation_dim': 66, 'action_dim': 3, 'action_semantics': 'normalized_vpp_forward_lateral_vertical_bias', 'encoder': {'checkpoint': 'E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_phaseconditional_v2/checkpoints/best.pt', 'sha256': '385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59', 'trainable': False, 'fallback': 'prohibited'}, 'prediction_reward': 'prohibited', 'policy_history_padding': 'prohibited', 'predictor_vpp_guidance_pid': 'frozen_existing_chain'} |
| `minimal_method_matrix` | PASS | ['candidate_fixed_defensive_extension', 'frozen_fixed_crossing', 'frozen_fixed_head_on'] |
| `candidate_untrained` | PASS | {'role': 'experimental', 'checkpoint': None, 'checkpoint_status': 'untrained_not_authorized', 'skill': 'defensive_extension', 'profile': 'range_extension', 'routing': 'disabled'} |
| `baseline_checkpoint_hashes` | PASS | 2/2 matched |
| `manifest_payload_hashes` | PASS | dev=4240e9d245d38b8f30fceb092e73ec1e6601d855b4ef4cea3ef991b66ecbc0c1 heldout=ec004d389330b8236cf3ed9548c1e9c118d6c57b499625a5a5feb4b81c19bfbc |
| `manifest_counts` | PASS | dev=12 heldout=24 |
| `v2_package_disjointness` | PASS | [(1075.0, 175.0, 345.0), (1125.0, 180.0, 350.0), (1525.0, 195.0, 395.0), (1600.0, 210.0, 410.0), (1775.0, 215.0, 420.0), (2000.0, 225.0, 430.0)] |
| `dev_heldout_disjointness` | PASS | set() |
| `train_fixed_package_disjointness` | PASS | train_range=[1250.0, 1450.0] fixed_ranges=[1075.0, 1125.0, 1200.0, 1500.0, 1525.0, 1600.0, 1775.0, 2000.0] |
| `scenario_contract` | PASS | 36 scenarios |
| `opponents_separate` | PASS | {'order': ['expert', 'end_to_end', 'independent_ppo_vpp'], 'balanced_training_sampling': True, 'report_separately': True, 'pooled_gate': 'prohibited'} |
| `fixed_training_budget` | PASS | {'total_timesteps': 50000, 'training_seed': 2026071402, 'budget_extension': 'prohibited', 'reward_change_after_start': 'prohibited', 'scenario_support_change_after_start': 'prohibited', 'profile_change_after_start': 'prohibited', 'opponent_sampling': 'balanced_round_robin', 'checkpoint_evaluation_steps': [10000, 20000, 30000, 40000, 50000], 'checkpoint_selection_split': 'dev12_only', 'heldout_use_for_selection': 'prohibited', 'ppo': {'rollout_steps': 2048, 'learning_rate': 0.0003, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_coef': 0.2, 'entropy_coef': 0.005, 'update_epochs': 10, 'minibatch_size': 256}, 'geometry_reward': {'source': 'frozen_profile_weighted_geometry_progress_v1', 'progress_weight': 2.0, 'alignment_weight': 0.1, 'terminal_ego_crash_oob_penalty': 2.0, 'physical_command_saturation_penalty': 0.05, 'intent_error_clip': 2.0}} |
| `preregistered_thresholds` | PASS | {'per_opponent_contract_minimum': {'qualifying_paired_episodes': 2, 'valid_target_steps': 20, 'distinct_scenario_signatures': 2, 'distinct_mirror_signs': 2, 'all_66d_and_3d_finite': True, 'no_padding': True, 'no_prediction_fallback': True, 'no_reset': True, 'no_backend_fallback': True, 'no_checkpoint_fallback': True}, 'per_opponent_claim_ready_coverage': {'qualifying_paired_episodes': 8, 'valid_target_steps': 160}, 'safety_noninferiority_vs_head_on': {'ego_crash_oob_rate_delta_max': 0.05}, 'geometry_noninferiority_vs_head_on': {'paired_intent_loss_auc20_delta_max': 0.02}, 'practical_improvement_vs_head_on': {'paired_intent_loss_auc20_delta_max': -0.05, 'minimum_opponents_passing': 2}, 'practical_improvement_vs_best_existing_specialist': {'paired_intent_loss_auc20_delta_max': -0.02, 'minimum_opponents_passing': 2}, 'remaining_opponent_noninferiority_vs_best_existing_specialist': {'paired_intent_loss_auc20_delta_max': 0.02}} |
| `fail_closed_stop_rule` | PASS | {'immediate_abort_on': ['nonfinite_66d_observation', 'nonfinite_or_out_of_bounds_3d_action', 'jsbsim_backend_fallback', 'checkpoint_fallback', 'abnormal_environment_reset', 'history_padding_or_future_state_injection'], 'dev_abort_on': {'metric': 'ego_crash_oob_rate_delta_vs_frozen_fixed_head_on', 'threshold': 0.05}, 'after_abort': {'freeze_as_negative_evidence': True, 'reward_tuning': 'prohibited', 'add_training_steps': 'prohibited', 'replace_scenarios': 'prohibited', 'replace_checkpoint_or_encoder': 'prohibited', 'rerun_same_source_id': 'prohibited'}} |
| `decision_tree_complete` | PASS | dict_keys(['new_skill_gap_supported', 'existing_library_routing_or_composition_gap', 'defensive_extension_hypothesis_not_supported', 'safety_no_go']) |
| `fresh_output_id` | PASS | E:\uav-vpp-guidance-five-state-heldout-envelope-v1-results\defensive_extension_range_extension_feasibility_v1 |

## 授权边界

- `training_authorized=false`。
- `pilot_execution_authorized=false`。
- 下一步必须由用户单独授权，且执行前重新验证 clean worktree、fresh output root、资产 SHA 与 config source ID。
