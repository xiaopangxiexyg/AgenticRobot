# Reviewed workload selections

This directory contains protected, human-reviewed `lab.workload-selection.v1`
manifests. A workload is a project-defined runtime variant; the generic control
plane does not assume that it is a machine-learning model. A selection binds one
non-authorizing asset observation to one exact artifact-content receipt for the
`gpu-only-validation` purpose.

A Task payload, remote request, host observation, or artifact inventory must never
create or modify a selection. Adding a selection is a control-plane change reviewed
before rebuilding `stack/remote-assets.json`. The `projectctl assets select` command
only emits a candidate lock; it does not install or activate that lock.

Selection manifests must keep:

- `device_access: false`
- `network_egress: loopback-only`
- `downloads_permitted: false`

No workload is currently selected. Do not add placeholder hashes.
