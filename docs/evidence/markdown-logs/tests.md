# Tests

````text
test_all_twenty_partitions_disjoint_and_cover_target (test_core.DataTests) ... ok
test_constant_segment_and_outlier_alignment (test_core.DataTests) ... ok
test_csv_roundtrip_and_duplicate_cycle_rejection (test_core.DataTests) ... ok
test_gap_is_not_a_consecutive_tmr_pair (test_core.DataTests) ... ok
test_invalid_input_rejected (test_core.DataTests) ... ok
test_scaler_only_fits_source_training (test_core.DataTests) ... ok
test_windows_and_successors_stay_within_battery (test_core.DataTests) ... ok
test_source_only_ignores_target_features (test_core.IntegrationTests) ... {"epoch": 1, "task": 0.0028924900107085705, "domain": 0.0, "tmr": 0.002020475920289755, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06932828316664273, "supported_health_points": 0}
{"epoch": 1, "task": 0.0028924900107085705, "domain": 0.0, "tmr": 0.002020475920289755, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06932828316664273, "supported_health_points": 0}
ok
test_target_label_changes_cannot_change_training (test_core.IntegrationTests) ... {"epoch": 1, "task": 0.004500733804889023, "domain": 0.7010716795921326, "tmr": 0.011020907666534185, "auxiliary": 0.054671330377459526, "source_val_rmse": 0.0669386283408293, "supported_health_points": 2}
{"epoch": 2, "task": 0.004326762049458921, "domain": 0.6992915868759155, "tmr": 0.01438581245020032, "auxiliary": 0.06064852885901928, "source_val_rmse": 0.0643616474880556, "supported_health_points": 2}
{"epoch": 1, "task": 0.004500733804889023, "domain": 0.7010716795921326, "tmr": 0.011020907666534185, "auxiliary": 0.054671330377459526, "source_val_rmse": 0.0669386283408293, "supported_health_points": 2}
{"epoch": 2, "task": 0.004326762049458921, "domain": 0.6992915868759155, "tmr": 0.01438581245020032, "auxiliary": 0.06064852885901928, "source_val_rmse": 0.0643616474880556, "supported_health_points": 2}
ok
test_unlabeled_target_training_and_prediction (test_core.IntegrationTests) ... {"epoch": 1, "task": 0.0028924900107085705, "domain": 0.6990596652030945, "tmr": 0.0, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06913279184297853, "supported_health_points": 0}
ok
test_forward_backprop_and_batch_independence (test_core.MathTests) ... ok
test_gate_bounds_and_unsupported_fallback (test_core.MathTests) ... ok
test_gradient_reversal_sign (test_core.MathTests) ... ok
test_identical_domain_moments_disable_modulation (test_core.MathTests) ... ok
test_metrics_fraction_and_percent_units (test_core.MathTests) ... ok
test_no_valid_tmr_has_zero_differentiable_loss (test_core.MathTests) ... ok
test_tmr_ignores_invalid_pairs_and_tolerance (test_core.MathTests) ... ok
test_zoh_matches_augmented_matrix_exponential (test_core.MathTests) ... ok
test_zoh_small_step_is_finite_and_first_order (test_core.MathTests) ... ok
test_all_backbones_forward_backward_and_inference (test_extended.DeploymentTests) ... ok
test_fifo_128_32_cold_start_and_cell_isolation (test_extended.DeploymentTests) ... ok
test_inference_never_calls_discriminator (test_extended.DeploymentTests) ... ok
test_invalid_vectors_do_not_advance_stride_and_time_reversal_rejected (test_extended.DeploymentTests) ... ok
test_replay_reference_changes_leave_predictions_identical (test_extended.DeploymentTests) ... ok
test_torchscript_reload_prediction_parity (test_extended.DeploymentTests) ... D:\ProgramData\Anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\instancenorm.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if input.size(feature_dim) != self.num_features:
D:\ProgramData\Anaconda3\envs\pytorch\lib\site-packages\torch\nn\functional.py:2835: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if size_prods == 1:
D:\ProgramData\Anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\rnn.py:314: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if self.input_size != input.size(-1):
D:\ProgramData\Anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\rnn.py:347: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if hx.size() != expected_hidden_size:
ok
test_audit_is_label_independent_in_counts (test_extended.WorkflowTests) ... ok
test_group_summary_units_and_absent_group_are_explicit (test_extended.WorkflowTests) ... ok
test_native_mit_hdf5_layout_and_nominal_soh (test_extended.WorkflowTests) ... ok
test_prepare_emits_numeric_cycle_order_for_online_replay (test_extended.WorkflowTests) ... ok
test_resume_after_interruption_matches_uninterrupted_training (test_extended.WorkflowTests) ... {"epoch": 1, "task": 0.0028924900107085705, "domain": 0.6990596652030945, "tmr": 0.0010102379601448774, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06895294150620598, "supported_health_points": 2}
{"epoch": 2, "task": 0.007417934946715832, "domain": 0.7028284668922424, "tmr": 0.0018965578638017178, "auxiliary": 0.0463850162923336, "source_val_rmse": 0.06790247800250693, "supported_health_points": 2}
{"epoch": 3, "task": 0.00036880074185319245, "domain": 0.6992059350013733, "tmr": 0.006749408785253763, "auxiliary": 0.02038584090769291, "source_val_rmse": 0.06732312187571043, "supported_health_points": 2}
{"epoch": 1, "task": 0.0028924900107085705, "domain": 0.6990596652030945, "tmr": 0.0010102379601448774, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06895294150620598, "supported_health_points": 2}
{"epoch": 2, "task": 0.007417934946715832, "domain": 0.7028284668922424, "tmr": 0.0018965578638017178, "auxiliary": 0.0463850162923336, "source_val_rmse": 0.06790247800250693, "supported_health_points": 2}
{"epoch": 3, "task": 0.00036880074185319245, "domain": 0.6992059350013733, "tmr": 0.006749408785253763, "auxiliary": 0.02038584090769291, "source_val_rmse": 0.06732312187571043, "supported_health_points": 2}
ok
test_sensitivity_constants_match_revised_paper (test_extended.WorkflowTests) ... ok
test_tuning_trials_never_generate_test_predictions (test_extended.WorkflowTests) ... [I 2026-10-25 21:17:36,995] A new study created in memory with name: no-name-0dd73db5-5cb5-4b8d-ad9a-55882aef6c9c
{"epoch": 1, "task": 0.0028924900107085705, "domain": 0.6990596652030945, "tmr": 0.0, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06954942779299707, "supported_health_points": 1}
[I 2026-10-25 21:17:37,031] Trial 0 finished with value: 0.06954942779299707 and parameters: {'learning_rate': 0.0002108628632940777, 'lambda_domain': 1.0912281685088012, 'lambda_phy': 1.4649532172404518, 'epsilon': 0.025}. Best is trial 0 with value: 0.06954942779299707.
{"epoch": 1, "task": 0.0028924900107085705, "domain": 0.6990596652030945, "tmr": 0.002010237891227007, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06959746512692869, "supported_health_points": 1}
[I 2026-10-25 21:17:37,060] Trial 1 finished with value: 0.06959746512692869 and parameters: {'learning_rate': 0.00019619069877630524, 'lambda_domain': 2.288116436428659, 'lambda_phy': 0.8373751722987874, 'epsilon': 0.001}. Best is trial 0 with value: 0.06954942779299707.
ok
test_bms_matching_does_not_broadcast_pack_or_future_references (test_response_commitments.ResponseCommitmentTests) ... ok
test_can_mapping_and_past_only_current_matching (test_response_commitments.ResponseCommitmentTests) ... ok
test_nonuniform_power_integration_matches_energy_per_window (test_response_commitments.ResponseCommitmentTests) ... ok
test_operation_scope_and_stride_amortization (test_response_commitments.ResponseCommitmentTests) ... ok
test_partial_charge_and_discharge_capacity_formulas (test_response_commitments.ResponseCommitmentTests) ... ok
test_resource_report_leaves_unmeasured_power_unset (test_response_commitments.ResponseCommitmentTests) ... ok
test_shared_cell_ids_remain_isolated_by_vehicle_and_protocol (test_response_commitments.ResponseCommitmentTests) ... ok
test_vehicle_adaptation_is_fixed_budget_label_blind_and_vehicle_disjoint (test_response_commitments.ResponseCommitmentTests) ... {"epoch": 1, "task": 0.0028924900107085705, "domain": 0.6990596652030945, "tmr": 0.0010102379601448774, "auxiliary": 0.049795813858509064, "source_val_rmse": 0.06898056008216763, "supported_health_points": 1}
ok

----------------------------------------------------------------------
Ran 40 tests in 2.912s

OK

````
