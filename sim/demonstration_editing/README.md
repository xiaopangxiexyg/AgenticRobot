# Demonstration trajectory editing

This package is the simulator-level, backend-neutral core of
demonstration-conditioned manipulation. It adopts the useful structural idea
from DemoGrasp—sample one high-level edit and apply it to a successful nominal
trajectory—without importing DemoGrasp, IsaacGym, a policy implementation, or
an embodiment-specific task.

It owns:

- an immutable, hash-bound TCP-pose plus explicitly named/unit-bound hand-target demonstration contract;
- an immutable high-level edit identity and its numeric SE(3)/hand delta;
- a pure edit operation;
- time-based interpolation of the resulting nominal references.
- a zero-edit offline identity gate over every source keyframe.
- an exact, authority-free projection into a caller-supplied canonical
  `ComponentActionSchema`.

It does not own:

- teleoperation or ControlTrace extraction;
- PPO storage, log probabilities, critic inputs, or model inference;
- task phase, reward, success, curriculum, or object randomization;
- collision classification, safety projection, IK, authority, or dispatch;
- MuJoCo/IsaacGym stepping or actuator dynamics;
- real-robot deployment.

The edited trajectory is explicitly a **nominal requested reference**. A task
adapter must retain the edit sample identity while recording composed,
admitted, and measured actions as separate downstream facts. Simulator-only
object/contact state may be used by an optional teacher or critic, but this
package does not promote it into an actor input.

`preflight_zero_edit_replay()` checks only that a zero edit survives the pure
edit/interpolation path. Passing it is not evidence for IK feasibility,
collision safety, MuJoCo execution, task success, or real-robot replay.

The MuJoCo-specific read adapter lives at
`sim.mujoco.adapters.demonstration_trace`. It exports only an explicitly
selected span from a committed ControlTrace. Its pose/hand reference is the
recorded `safe_action` (after shared safety and before resolution/IK), while
accepted dispatch, post-dispatch measurement, and task-owned success evidence
remain separate required lineage. It never treats selected action, qcmd, or
measured qpos as an unqualified demonstration action.

`DemonstrationComponentActionAdapter` is the downstream representation seam.
It validates the pose frame, controlled frame, hand component, representation,
coordinate names, units, and exact target schema before returning a native
`ComponentAction`. Its receipt preserves source-trajectory, edit, PPO sample,
edited-trajectory, reference-time, binding, and output-action identities. It
does not submit the action or grant authority; callers must use an existing
composition/direct-source admission path and retain the receipt as lineage.
For composition input, bind `receipt_id` and `semantic_content_hash` under
`DEMONSTRATION_COMPONENT_ACTION_RECEIPT_BINDING_KIND_V1`. The readable sample
ID may additionally be copied into non-authoritative audit metadata, but the
receipt hash—not that audit field—is the semantic proof that the sample,
reference, binding, and output action belong together.

For simple parallel grippers, a trace may retain the editable one-dimensional
`normalized_gripper_close` contract (`0=open`, `1=closed`) instead of a padded
execution carrier. `NormalizedGripperCloseToJointPositionProjection` binds
that scalar to explicit open/closed native joint targets at the final
representation seam. The endpoints and conversion are hash-bound, values
outside `[0, 1]` are rejected rather than silently clipped, and the result is
still neither admitted nor declared safe by this package.

`sim/mujoco_temp_pipeline` may consume this API in an experiment, but this
package never imports or depends on that temporary learner.
