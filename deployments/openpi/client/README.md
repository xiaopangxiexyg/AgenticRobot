# OpenPI Client

This directory contains only OpenPI-specific client pieces:

- OpenPI websocket/msgpack transport
- OpenPI observation field adapter
- OpenPI policy factory for the generic `deployments.client` remote policy
- OpenPI startup command wrappers

The generic remote-policy contract, action chunk, and runtime glue live in
`deployments.client`.

Example real dry-run:

```bash
python -m deployments.openpi.client.real \
  --host 127.0.0.1 \
  --port 8000 \
  --model dual_rm65_dmgripper_latest \
  --active-sides both \
  --hand-backend dmtac-gripper \
  --dmtac-config latest
```

The command above expands to the generic real deploy loop with:

```bash
--policy deployments.openpi.client.policy:create_policy
```

`teleop_runtime` still owns real hardware IO and dispatch, but it does not
import or know about OpenPI.

