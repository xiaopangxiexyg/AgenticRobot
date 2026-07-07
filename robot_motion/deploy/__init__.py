"""Common deployment runtime glue shared by sim and real launchers.

Migration note:
Large deploy runtime pieces still live under ``teleop_runtime.deploy`` during
the current transition. Model-server client code lives under
``deployments.client``; this package is reserved for sim/real-neutral motion
runtime glue.
"""

from .dispatcher import CommandDispatcher, DryRunCommandDispatcher
from .runtime import DeployRuntime, DeployRuntimeStepResult
from .source import DeploymentObservationSource

__all__ = [
    "CommandDispatcher",
    "DeploymentObservationSource",
    "DeployRuntime",
    "DeployRuntimeStepResult",
    "DryRunCommandDispatcher",
]
