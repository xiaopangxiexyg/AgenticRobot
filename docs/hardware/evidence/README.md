# Hardware evidence receipts

This directory contains versioned historical receipts owned by hardware-facing
documentation.  A receipt records only its stated scope and never authorizes a
device connection, configuration change, or motion command.

| Receipt | Recorded scope |
|---|---|
| [2026-08-14 camera role registry](./camera_semantic_corrections/20260814_batch_v1_camera_role_registry.json) | Frozen camera-role input used by the camera-semantic correction batch |
| [2026-08-14 formal tactile export targets](./formal_tactile_export/20260814_formal_collection_targets.json) | Source-preserving NAS export and safe-removal readiness target list |
| [2026-09-06 B601 manual single-axis operator attestation](./b601_rs/20260906_manual_single_axis_operator_attestation.json) | Append-only synchronization of the operator-observed single-axis result and the isolated 0.12 rad/s normal-qvel decision; does not replace the failed source run or claim formal hardware acceptance |
| [2026-10-25 four-centre arrival and antiwindup](./20261025-four-centres-arrival-antiwindup.md) | Preserved source records and bounded feedback analysis; integration does not repeat the physical run |
| [2026-10-26 near-zero terminal reference](./20261026-near-zero-terminal-reference.md) | Preserved source diagnosis separating arrival control, acquisition outcome and final return receipts |
