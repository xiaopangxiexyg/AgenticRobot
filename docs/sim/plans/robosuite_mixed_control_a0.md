# Robosuite mixed-control A0

## Scope and boundary

A0 establishes a simulation-only, backend-injected seam for deterministic
mixed-control experiments. It does not import or launch Robosuite, start a
service, access hardware, dispatch robot motion, or claim real-simulator
readiness. The acceptance backend is a pure Python deterministic mock.

Production entrypoints and the existing control authority model are unchanged.
The adapter consumes a mode that has already been selected by the existing
composition allocation contract.

## Three modes

The adapter reuses the exact existing mode constants:

- `base_only`: apply the base action.
- `relative_overlay`: add the intervention to the base action.
- `absolute_overlay_owner`: apply the intervention action for the selected
  component.

All three selections remain non-authoritative. In particular,
`absolute_overlay_owner` is a soft allocation name, not a motion-authority
transfer. After the selected composition is evaluated, the adapter enforces
the final per-dimension action envelope before the backend receives the action.

## Reset and replay identity

`RobosuiteResetManifest` binds the environment identifier and contract hash,
task and episode identity, reset sequence, seed, action dimension, ordered
observation keys, step limit, and closed primitive reset options. Its canonical
semantic hash is the replay identity.

Backend reset returns an observation and produces a receipt binding:

- the exact manifest ID and hash;
- the injected backend ID; and
- the canonical hash of the initial observation.

The mock runner records content-addressed input frames, composed selections,
backend outcomes, stop reason, and ordered step hashes. Repeating an exact
manifest and exact frame stream produces an exact run hash; changing the seed
changes reset and run identity.

## A0 validation

Focused tests cover manifest fail-closed validation, mapping-order-independent
hashing, the three composition modes, post-composition clipping, retained
non-authority, exact reset receipts, deterministic replay, seed divergence,
stream ordering, step-limit termination, and absence of a real `robosuite`
runtime import.

Real Robosuite binding, simulator assets, controller configuration validation,
performance evidence, and any hardware/production path are explicitly deferred.
