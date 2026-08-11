"""Pure profiles and episode bindings for coarse collection progress."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .canonical_encoding import canonical_content_hash


PROGRESS_STEP_DEFINITION_SCHEMA_V1 = "dexteleop.progress_step_definition.v1"
PROGRESS_STEP_ARM_CONTROL_SCHEMA_V1 = "dexteleop.progress_step_arm_control.v1"
TASK_PROGRESS_PROFILE_SCHEMA_V1 = "dexteleop.task_progress_profile.v1"
PROGRESS_EPISODE_BINDING_SCHEMA_V1 = "dexteleop.progress_episode_binding.v1"

PROGRESS_TOPOLOGY_KINDS_V1 = frozenset({"LINEAR", "ROOTED_TREE"})
PROGRESS_RECORDING_SCOPES_V1 = frozenset(
    {
        "FULL_TASK",
        "PROGRESS_STEP_WINDOW",
        "TRANSITION_WINDOW",
        "EVENT_WINDOW",
    }
)
PROGRESS_ENTRY_MODES_V1 = frozenset(
    {
        "PROFILE_INITIAL",
        "ACTUAL_TRUNCATED_HISTORY",
        "SCENARIO_SETUP",
        "NO_PREHISTORY",
    }
)
PROGRESS_ARM_CONTROL_MODES_V1 = frozenset({"teleop", "fixed_hold"})
PROGRESS_FIXED_SCOPE_ARM_AND_END_EFFECTOR_V1 = "arm_and_end_effector"

_PROFILE_HASH_DOMAIN = "dexteleop.task_progress_profile.semantic.v1"
_BINDING_HASH_DOMAIN = "dexteleop.progress_episode_binding.semantic.v1"


class ProgressProfileContractError(ValueError):
    """Raised when progress topology or episode entry is ambiguous."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProgressProfileContractError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProgressProfileContractError(
            f"{name} must be a positive integer."
        )
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgressProfileContractError(f"{name} must be boolean.")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ProgressProfileContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _unique_strings(
    values: Sequence[object],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    result = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not result:
        raise ProgressProfileContractError(f"{name} must not be empty.")
    if len(result) != len(set(result)):
        raise ProgressProfileContractError(f"{name} entries must be unique.")
    return result


@dataclass(frozen=True)
class ProgressStepArmControlV1:
    """Required per-side arm state for samples in one coarse progress step.

    ``fixed_hold`` always covers the arm and its configured end effector.  It
    is deliberately not a generic per-effector mask: RealTeleop must not record
    a step whose arm and gripper ownership disagree with the profile boundary.
    """

    left: str
    right: str
    fixed_scope: str = PROGRESS_FIXED_SCOPE_ARM_AND_END_EFFECTOR_V1
    schema: str = field(default=PROGRESS_STEP_ARM_CONTROL_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for side in ("left", "right"):
            mode = _required(getattr(self, side), name=side).lower()
            if mode not in PROGRESS_ARM_CONTROL_MODES_V1:
                raise ProgressProfileContractError(
                    f"Unsupported {side} progress arm-control mode {mode!r}."
                )
            object.__setattr__(self, side, mode)
        scope = _required(self.fixed_scope, name="fixed_scope").lower()
        if scope != PROGRESS_FIXED_SCOPE_ARM_AND_END_EFFECTOR_V1:
            raise ProgressProfileContractError(
                "Progress fixed_hold must cover arm_and_end_effector."
            )
        object.__setattr__(self, "fixed_scope", scope)

    @property
    def teleop_sides(self) -> tuple[str, ...]:
        return tuple(
            side for side in ("left", "right") if getattr(self, side) == "teleop"
        )

    @property
    def fixed_sides(self) -> tuple[str, ...]:
        return tuple(
            side
            for side in ("left", "right")
            if getattr(self, side) == "fixed_hold"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "left": self.left,
            "right": self.right,
            "fixed_scope": self.fixed_scope,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressStepArmControlV1":
        if payload.get("schema") != PROGRESS_STEP_ARM_CONTROL_SCHEMA_V1:
            raise ProgressProfileContractError(
                "Unsupported progress-step arm-control schema."
            )
        return cls(
            left=payload.get("left", ""),
            right=payload.get("right", ""),
            fixed_scope=payload.get("fixed_scope", ""),
        )


@dataclass(frozen=True)
class ProgressStepDefinitionV1:
    """One unique progress node; semantic types may repeat across nodes."""

    progress_step_id: str
    semantic_type: str
    display_name: str
    ordinal: int
    allowed_next_step_ids: tuple[str, ...] = ()
    arm_control: ProgressStepArmControlV1 | None = None
    schema: str = field(default=PROGRESS_STEP_DEFINITION_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in ("progress_step_id", "semantic_type", "display_name"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "ordinal",
            _positive_int(self.ordinal, name="ordinal"),
        )
        object.__setattr__(
            self,
            "allowed_next_step_ids",
            _unique_strings(
                self.allowed_next_step_ids,
                name="allowed_next_step_ids",
            ),
        )
        if self.arm_control is not None and not isinstance(
            self.arm_control,
            ProgressStepArmControlV1,
        ):
            raise ProgressProfileContractError(
                "arm_control must be ProgressStepArmControlV1 or None."
            )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "progress_step_id": self.progress_step_id,
            "semantic_type": self.semantic_type,
            "display_name": self.display_name,
            "ordinal": self.ordinal,
            "allowed_next_step_ids": self.allowed_next_step_ids,
        }
        if self.arm_control is not None:
            payload["arm_control"] = self.arm_control.to_payload()
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressStepDefinitionV1":
        if payload.get("schema") != PROGRESS_STEP_DEFINITION_SCHEMA_V1:
            raise ProgressProfileContractError(
                "Unsupported progress-step definition schema."
            )
        arm_control_payload = payload.get("arm_control")
        if arm_control_payload is not None and not isinstance(
            arm_control_payload,
            Mapping,
        ):
            raise ProgressProfileContractError("arm_control must be a mapping.")
        return cls(
            progress_step_id=payload.get("progress_step_id", ""),
            semantic_type=payload.get("semantic_type", ""),
            display_name=payload.get("display_name", ""),
            ordinal=payload.get("ordinal", 0),
            allowed_next_step_ids=tuple(payload.get("allowed_next_step_ids", ())),
            arm_control=(
                None
                if arm_control_payload is None
                else ProgressStepArmControlV1.from_payload(arm_control_payload)
            ),
        )


@dataclass(frozen=True)
class TaskProgressProfileV1:
    """Reusable coarse progress graph with linear or rooted-tree topology."""

    profile_id: str
    profile_version: str
    task_id: str
    steps: tuple[ProgressStepDefinitionV1, ...]
    initial_step_id: str
    terminal_step_ids: tuple[str, ...]
    topology_kind: str = "LINEAR"
    allow_retry_after_failure: bool = True
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=TASK_PROGRESS_PROFILE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in ("profile_id", "profile_version", "task_id", "initial_step_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        topology = _required(self.topology_kind, name="topology_kind").upper()
        if topology not in PROGRESS_TOPOLOGY_KINDS_V1:
            raise ProgressProfileContractError(
                f"Unsupported progress topology: {topology!r}."
            )
        steps = tuple(self.steps)
        if not steps or any(
            not isinstance(step, ProgressStepDefinitionV1) for step in steps
        ):
            raise ProgressProfileContractError(
                "steps must contain ProgressStepDefinitionV1 values."
            )
        ids = tuple(step.progress_step_id for step in steps)
        ordinals = tuple(step.ordinal for step in steps)
        if len(ids) != len(set(ids)):
            raise ProgressProfileContractError(
                "progress_step_id values must be unique."
            )
        if len(ordinals) != len(set(ordinals)):
            raise ProgressProfileContractError("step ordinals must be globally unique.")
        if tuple(sorted(ordinals)) != ordinals:
            raise ProgressProfileContractError(
                "steps must be ordered by strictly increasing ordinal."
            )
        if self.initial_step_id not in set(ids):
            raise ProgressProfileContractError("initial_step_id is unknown.")
        terminals = _unique_strings(
            self.terminal_step_ids,
            name="terminal_step_ids",
            allow_empty=False,
        )
        if unknown := set(terminals) - set(ids):
            raise ProgressProfileContractError(
                f"terminal_step_ids reference unknown steps: {sorted(unknown)}."
            )

        by_id = {step.progress_step_id: step for step in steps}
        if topology == "LINEAR":
            if self.initial_step_id != ids[0]:
                raise ProgressProfileContractError(
                    "A linear profile must start at its first ordinal."
                )
            normalized_steps: list[ProgressStepDefinitionV1] = []
            for index, step in enumerate(steps):
                expected = () if index + 1 == len(steps) else (ids[index + 1],)
                if step.allowed_next_step_ids not in {(), expected}:
                    raise ProgressProfileContractError(
                        "Linear allowed_next_step_ids must be empty or the next "
                        f"ordinal for {step.progress_step_id!r}."
                    )
                normalized_steps.append(
                    replace(step, allowed_next_step_ids=expected)
                )
            steps = tuple(normalized_steps)
            if terminals != (ids[-1],):
                raise ProgressProfileContractError(
                    "A linear profile terminal must be its final ordinal."
                )
        else:
            normalized_steps = []
            for step in steps:
                unknown_children = set(step.allowed_next_step_ids) - set(by_id)
                if unknown_children:
                    raise ProgressProfileContractError(
                        f"Unknown child steps: {sorted(unknown_children)}."
                    )
                normalized_steps.append(
                    replace(
                        step,
                        allowed_next_step_ids=tuple(
                            sorted(
                                step.allowed_next_step_ids,
                                key=lambda child_id: by_id[child_id].ordinal,
                            )
                        ),
                    )
                )
            steps = tuple(normalized_steps)
            by_id = {step.progress_step_id: step for step in steps}
            parent_by_child: dict[str, str] = {}
            for step in steps:
                for child_id in step.allowed_next_step_ids:
                    child = by_id.get(child_id)
                    if child is None:
                        raise ProgressProfileContractError(
                            f"Unknown child step {child_id!r}."
                        )
                    if child.ordinal <= step.ordinal:
                        raise ProgressProfileContractError(
                            "Tree child ordinal must be greater than its parent."
                        )
                    if child_id in parent_by_child:
                        raise ProgressProfileContractError(
                            "ROOTED_TREE forbids branch merges."
                        )
                    parent_by_child[child_id] = step.progress_step_id
            if self.initial_step_id in parent_by_child:
                raise ProgressProfileContractError(
                    "The rooted-tree initial step must have no parent."
                )
            non_root = set(ids) - {self.initial_step_id}
            if set(parent_by_child) != non_root:
                raise ProgressProfileContractError(
                    "Every non-root tree step must have exactly one parent."
                )
            reachable = {self.initial_step_id}
            pending = [self.initial_step_id]
            while pending:
                parent_id = pending.pop()
                for child_id in by_id[parent_id].allowed_next_step_ids:
                    if child_id not in reachable:
                        reachable.add(child_id)
                        pending.append(child_id)
            if reachable != set(ids):
                raise ProgressProfileContractError(
                    "ROOTED_TREE contains unreachable steps."
                )
            leaves = tuple(
                step.progress_step_id
                for step in steps
                if not step.allowed_next_step_ids
            )
            if set(terminals) != set(leaves):
                raise ProgressProfileContractError(
                    "Tree terminal_step_ids must exactly equal its leaves."
                )
            terminals = leaves

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "terminal_step_ids", terminals)
        object.__setattr__(self, "topology_kind", topology)
        object.__setattr__(
            self,
            "allow_retry_after_failure",
            _boolean(
                self.allow_retry_after_failure,
                name="allow_retry_after_failure",
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_PROFILE_HASH_DOMAIN,
            ),
        )

    @property
    def step_by_id(self) -> Mapping[str, ProgressStepDefinitionV1]:
        return {step.progress_step_id: step for step in self.steps}

    def allowed_next_step_ids(self, progress_step_id: str) -> tuple[str, ...]:
        step_id = _required(progress_step_id, name="progress_step_id")
        step = self.step_by_id.get(step_id)
        if step is None:
            raise ProgressProfileContractError(f"Unknown progress step {step_id!r}.")
        if self.topology_kind == "ROOTED_TREE":
            return step.allowed_next_step_ids
        index = tuple(item.progress_step_id for item in self.steps).index(step_id)
        return (
            ()
            if index + 1 == len(self.steps)
            else (self.steps[index + 1].progress_step_id,)
        )

    def ancestor_path_to(self, progress_step_id: str) -> tuple[str, ...]:
        step_id = _required(progress_step_id, name="progress_step_id")
        if step_id not in self.step_by_id:
            raise ProgressProfileContractError(f"Unknown progress step {step_id!r}.")
        if self.topology_kind == "LINEAR":
            ids = tuple(step.progress_step_id for step in self.steps)
            return ids[: ids.index(step_id)]
        parent_by_child = {
            child_id: step.progress_step_id
            for step in self.steps
            for child_id in step.allowed_next_step_ids
        }
        reversed_path: list[str] = []
        current = step_id
        while current in parent_by_child:
            current = parent_by_child[current]
            reversed_path.append(current)
        return tuple(reversed(reversed_path))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "task_id": self.task_id,
            "steps": tuple(step.to_payload() for step in self.steps),
            "initial_step_id": self.initial_step_id,
            "terminal_step_ids": self.terminal_step_ids,
            "topology_kind": self.topology_kind,
            "allow_retry_after_failure": self.allow_retry_after_failure,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TaskProgressProfileV1":
        if payload.get("schema") != TASK_PROGRESS_PROFILE_SCHEMA_V1:
            raise ProgressProfileContractError(
                "Unsupported task-progress profile schema."
            )
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            task_id=payload.get("task_id", ""),
            steps=tuple(
                ProgressStepDefinitionV1.from_payload(step)
                for step in payload.get("steps", ())
            ),
            initial_step_id=payload.get("initial_step_id", ""),
            terminal_step_ids=tuple(payload.get("terminal_step_ids", ())),
            topology_kind=payload.get("topology_kind", ""),
            allow_retry_after_failure=payload.get(
                "allow_retry_after_failure",
                True,
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressProfileContractError("Task-progress profile hash mismatch.")
        return result


@dataclass(frozen=True)
class ProgressEpisodeBindingV1:
    """Bind one externally-owned Episode to its actual progress entry."""

    episode_capture_identity_hash: str
    task_progress_profile_hash: str
    bound_clock_domain_id: str
    recording_scope: str
    entry_step_id: str
    entry_mode: str
    history_complete: bool
    assumed_completed_path_step_ids: tuple[str, ...] = ()
    truncated_at_start: bool = False
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=PROGRESS_EPISODE_BINDING_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "episode_capture_identity_hash",
            "task_progress_profile_hash",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "bound_clock_domain_id",
            _required(self.bound_clock_domain_id, name="bound_clock_domain_id"),
        )
        scope = _required(self.recording_scope, name="recording_scope").upper()
        mode = _required(self.entry_mode, name="entry_mode").upper()
        if scope not in PROGRESS_RECORDING_SCOPES_V1:
            raise ProgressProfileContractError(
                f"Unsupported recording_scope {scope!r}."
            )
        if mode not in PROGRESS_ENTRY_MODES_V1:
            raise ProgressProfileContractError(f"Unsupported entry_mode {mode!r}.")
        object.__setattr__(
            self,
            "entry_step_id",
            _required(self.entry_step_id, name="entry_step_id"),
        )
        object.__setattr__(
            self,
            "assumed_completed_path_step_ids",
            _unique_strings(
                self.assumed_completed_path_step_ids,
                name="assumed_completed_path_step_ids",
            ),
        )
        object.__setattr__(self, "recording_scope", scope)
        object.__setattr__(self, "entry_mode", mode)
        object.__setattr__(
            self,
            "history_complete",
            _boolean(self.history_complete, name="history_complete"),
        )
        object.__setattr__(
            self,
            "truncated_at_start",
            _boolean(self.truncated_at_start, name="truncated_at_start"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_BINDING_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "episode_capture_identity_hash": self.episode_capture_identity_hash,
            "task_progress_profile_hash": self.task_progress_profile_hash,
            "bound_clock_domain_id": self.bound_clock_domain_id,
            "recording_scope": self.recording_scope,
            "entry_step_id": self.entry_step_id,
            "entry_mode": self.entry_mode,
            "history_complete": self.history_complete,
            "assumed_completed_path_step_ids": (
                self.assumed_completed_path_step_ids
            ),
            "truncated_at_start": self.truncated_at_start,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressEpisodeBindingV1":
        if payload.get("schema") != PROGRESS_EPISODE_BINDING_SCHEMA_V1:
            raise ProgressProfileContractError(
                "Unsupported progress-episode binding schema."
            )
        result = cls(
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash",
                "",
            ),
            task_progress_profile_hash=payload.get(
                "task_progress_profile_hash",
                "",
            ),
            bound_clock_domain_id=payload.get("bound_clock_domain_id", ""),
            recording_scope=payload.get("recording_scope", ""),
            entry_step_id=payload.get("entry_step_id", ""),
            entry_mode=payload.get("entry_mode", ""),
            history_complete=payload.get("history_complete", False),
            assumed_completed_path_step_ids=tuple(
                payload.get("assumed_completed_path_step_ids", ())
            ),
            truncated_at_start=payload.get("truncated_at_start", False),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ProgressProfileContractError(
                "Progress-episode binding hash mismatch."
            )
        return result


def validate_progress_episode_binding(
    *,
    profile: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
) -> None:
    if not isinstance(profile, TaskProgressProfileV1):
        raise TypeError("profile must be TaskProgressProfileV1.")
    if not isinstance(binding, ProgressEpisodeBindingV1):
        raise TypeError("binding must be ProgressEpisodeBindingV1.")
    if binding.task_progress_profile_hash != profile.semantic_content_hash:
        raise ProgressProfileContractError("Binding/profile hash mismatch.")
    if binding.entry_step_id not in profile.step_by_id:
        raise ProgressProfileContractError("Binding entry_step_id is unknown.")
    expected_path = profile.ancestor_path_to(binding.entry_step_id)
    declared_path = binding.assumed_completed_path_step_ids
    if declared_path and declared_path != expected_path:
        raise ProgressProfileContractError(
            "assumed_completed_path_step_ids must be the complete ancestor path."
        )
    if binding.history_complete and declared_path != expected_path:
        raise ProgressProfileContractError(
            "history_complete requires the complete ancestor path."
        )
    if binding.recording_scope == "FULL_TASK":
        if (
            binding.entry_step_id != profile.initial_step_id
            or binding.entry_mode != "PROFILE_INITIAL"
            or not binding.history_complete
            or binding.truncated_at_start
        ):
            raise ProgressProfileContractError(
                "FULL_TASK must start from the complete profile initial state."
            )
    if binding.entry_mode == "PROFILE_INITIAL" and (
        binding.entry_step_id != profile.initial_step_id
    ):
        raise ProgressProfileContractError(
            "PROFILE_INITIAL must use the profile initial step."
        )
    if (
        binding.entry_mode in {"SCENARIO_SETUP", "NO_PREHISTORY"}
        and declared_path
    ):
        raise ProgressProfileContractError(
            f"{binding.entry_mode} cannot assert completed history."
        )


__all__ = [
    "PROGRESS_ENTRY_MODES_V1",
    "PROGRESS_EPISODE_BINDING_SCHEMA_V1",
    "PROGRESS_RECORDING_SCOPES_V1",
    "PROGRESS_STEP_DEFINITION_SCHEMA_V1",
    "PROGRESS_TOPOLOGY_KINDS_V1",
    "TASK_PROGRESS_PROFILE_SCHEMA_V1",
    "ProgressEpisodeBindingV1",
    "ProgressProfileContractError",
    "ProgressStepDefinitionV1",
    "TaskProgressProfileV1",
    "validate_progress_episode_binding",
]
