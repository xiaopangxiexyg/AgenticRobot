# deployments.client

This package is the model-agnostic deployment client layer.

It owns:

- remote server metadata validation
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

