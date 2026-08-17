# Architecture evidence receipts

The JSON files below this directory are historical, scoped receipts linked by
the architecture guides and shared plans.  Their recorded commands, paths,
hashes, and tested boundaries remain snapshots; moving the files out of the
former `todos/evidence/` directory does not widen the evidence they establish.

Receipt filenames follow the batch names used by the
[RTC landing plan](../plans/dexteleop_rtc_landing_plan.md). They are indexed
here for discovery only: status, environment, exclusions, and conclusions must
be read from the receipt itself and its owning plan.

## RTC producer and local integration

- [RTC-3b.5 local](./rtc/rtc_3b5_local_evidence_v1.json)
- [RTC-3c local](./rtc/rtc_3c_local_evidence_v1.json)
- [RTC-4a model-agnostic local](./rtc/rtc_4a_model_agnostic_local_evidence_v1.json)
- [RTC-4b-A OpenPI final-chunk local](./rtc/rtc_4b_a_openpi_final_chunk_local_evidence_v1.json)
- [RTC-4b-B1 OpenPI remote source](./rtc/rtc_4b_b1_openpi_remote_source_evidence_v1.json)

## RTC transport and composition

- [RTC-4b-T1 transport lifecycle](./rtc/rtc_4b_t1_transport_lifecycle_local_evidence_v1.json)
- [RTC-4b-T2a transport deadline](./rtc/rtc_4b_t2a_transport_deadline_local_evidence_v1.json)
- [RTC-4b-T2b OpenPI live no-action](./rtc/rtc_4b_t2b_openpi_live_no_action_evidence_v1.json)
- [RTC-4b-T2c live worker-plan no-action](./rtc/rtc_4b_t2c_openpi_live_worker_plan_no_action_evidence_v1.json)
- [RTC-4b-T2d framework composition](./rtc/rtc_4b_t2d_framework_composition_local_evidence_v1.json)
- [RTC-4b-T2e VLA scratch additive](./rtc/rtc_4b_t2e_vla_scratch_additive_local_evidence_v1.json)
- [RTC-4b-T2f VLA scratch dataset view](./rtc/rtc_4b_t2f_vla_scratch_dataset_view_local_evidence_v1.json)
- [RTC-4b-T2g process-boundary virtual policy](./rtc/rtc_4b_t2g_vla_scratch_process_boundary_virtual_policy_evidence_v1.json)
- [RTC-4b-T2h raw-RTC final-chunk composition](./rtc/rtc_4b_t2h_raw_rtc_final_chunk_composition_local_evidence_v1.json)
- [RTC-4b-T3 OpenPI raw-RTC local](./rtc/rtc_4b_t3_openpi_raw_rtc_local_evidence_v1.json)

## Model-RTC sampler

- [MRTC-S1b1 conditioned sampler synthetic](./rtc/mrtc_s1b1_openpi_gpu_conditioned_sampler_synthetic_evidence_v1.json)
- [MRTC-S1b2 GPU wire frontend shadow](./rtc/mrtc_s1b2_openpi_gpu_wire_frontend_shadow_evidence_v1.json)

## Relative and real-observation gates

- [RTC-RR1 relative adapter local](./rtc/rtc_rr1_openpi_relative_adapter_local_evidence_v1.json)
- [RTC-RR2 relative live no-action](./rtc/rtc_rr2_openpi_relative_live_no_action_evidence_v1.json)
- [RTC-RR3 real observation no-action](./rtc/rtc_rr3_openpi_relative_real_observation_no_action_evidence_v1.json)
- [RTC-RR3b real observation admission no-action](./rtc/rtc_rr3b_openpi_relative_real_observation_admission_no_action_evidence_v1.json)
- [RTC-RR3c local request-path regression](./rtc/rtc_rr3c_local_request_path_regression_v1.json)
- [RTC-RR3c real rerun blocked by camera](./rtc/rtc_rr3c_real_rerun_blocked_camera_v1.json)
- [RTC-RR3d real rerun blocked by DMTAC](./rtc/rtc_rr3d_real_rerun_blocked_dmtac_v1.json)
- [RTC-RR4 real observation no-action default profile](./rtc/rtc_rr4_openpi_raw_rtc_real_observation_no_action_default_profile_evidence_v1.json)
- [RTC-RR4 real observation no-action drain gate](./rtc/rtc_rr4_openpi_raw_rtc_real_observation_no_action_drain_gate_evidence_v1.json)

## Target compatibility and preparation

- [RTC-TX1 emission/target compatibility](./rtc/rtc_tx1_policy_emission_target_compatibility_local_evidence_v1.json)
- [RTC-TX3a proposal preparation](./rtc/rtc_tx3a_policy_emission_target_proposal_preparation_local_evidence_v1.json)
