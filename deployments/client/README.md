# deployments.client

This package is the model-agnostic deployment client layer.

It owns:

- remote server metadata validation
- versioned policy-artifact JSON loading and strict input/output tensor binding
- remote policy response normalization into the unified dense action chunk
- generic source -> policy -> action-preparer -> dispatcher runtime glue

It does not own:

- OpenPI websocket/msgpack details
- ACT/DiffusionPolicy/etc. model-family field names
- real hardware backends
- MuJoCo or other sim backends

Model-specific packages, for example `deployments.openpi.client`, provide the
transport and observation adapter. Real and sim launchers provide their own
observation sources, canonical action preparers, and dispatchers.

Current production note: `deployments.openpi.client.real` is a thin supported
launcher over the supported `teleop_runtime.deploy.run_policy` runtime; the generic
`DeploymentClientRuntime` in this package is a composition contract, not the
current RM65 hardware loop. Position-control work must therefore keep common
planning in `robot_motion` and connect it at that existing real-runtime
boundary, rather than creating a second loop here.

## v2.2 policy artifact

`dexteleop_policy_artifact_v1` embeds the canonical `PolicyIOLayout` and its
hash. Input/output float32 dimensions and independent int64 token dimensions
are derived from that layout; a sidecar cannot redefine them independently.
`deploy_contract_from_policy_artifact()` turns it into a strict
`dexteleop_deploy_v2` server handshake. Dataset, checkpoint/server metadata and
runtime must agree on `policy_io_layout_hash`.

For a state-dropout-trained model, the artifact additionally declares
`state_input_mode="masked_normalized_concat_v1"`. Its float input dimension is
twice the layout base-state dimension: normalized state followed by a binary
availability mask. The client/deployment adapter appends an all-one mask at
runtime and requires the server metadata to declare the same mode and doubled
input dimension. Existing artifacts default to `raw_state_v1` and retain the
original state dimension unchanged.

Unmarked models retain `dexteleop_deploy_v1` behavior (26D action, 38D state,
legacy q6 hands). A v2.2 artifact is opt-in and does not reinterpret an old
checkpoint.
