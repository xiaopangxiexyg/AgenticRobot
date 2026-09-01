# Threaded policy prefetch retirement (2026-09-01)

## Retired surface

The legacy `threaded-prefetch` policy-prediction mode in
`teleop_runtime.deploy` has been removed from the versioned source tree.  The
mode was an optional, single-worker inference path for the legacy action queue;
it was not used by the synchronous deployment default or by any RTC profile.

The bounded worker, its deployment lifecycle wiring, the two CLI options, and
their mode-specific replay validation/tests were deleted together.  The
canonical non-RTC path now performs prediction through
`PolicyDeployRuntime.predict_and_queue()` on the existing caller-owned
transaction path.

## Evidence and replacement

Before removal, the versioned deployment/configuration/profile trees had no
tracked invocation of `--policy-prediction-mode threaded-prefetch` or its queue
threshold option.  The only production references were the parser, worker,
lifecycle hook, and `run_policy` branch removed by this change.  Dataset replay
continues to use the existing replay-window/source validators.

The replacement is the established synchronous policy path.  It retains
`PolicyDeployRuntime` queue and chunk lineage, `PreparedActionExecutor` routing,
`PositionDeployExecutor` source-index/time-grid/collision/dispatch ownership,
and policy-error pause/hold behavior.

## Boundaries intentionally preserved

This retirement does not change RTC model/direct scheduling, the
`ThreadedPolicyEmissionSource` used by backend-neutral OpenPI emission, or the
real-teleop HITL policy-source prefetch threshold.  Those are separate owners
and remain available where their contracts require them.  No hardware or
simulation runtime was started as part of the retirement.

Callers that still pass either retired CLI option must migrate to the default
synchronous deployment path; the options are no longer accepted by
`teleop_runtime.deploy`.
