# Robot Motion Arm Init

This package holds arm-initialization logic that is not specific to one robot
model or one runtime.

Current shared pieces:

- `qpos_transition.py`: slow qpos-to-qpos transition planning with joint
  velocity profiles.
- `constrained_transition.py`: risk levels, runtime safety signals, constrained
  qpos transition planning hooks, and model-agnostic FK/collision/singularity
  callbacks.

Robot-specific packages should provide:

- qpos naming and DOF validation,
- target qpos loading/schema compatibility,
- FK/collision/singularity model adapters,
- real or sim command dispatch.

For RM65, those pieces live under `robot_motion/rm65`.
