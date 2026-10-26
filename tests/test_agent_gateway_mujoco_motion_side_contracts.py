"""Offline fixed-side contracts; no Scene, native planner, transport or device.

Historical right-side digests come from Git blobs at BASE_COMMIT. Explicit
pose-goal and debug-publication migrations retain those historical pins below.
Only resource imports are omitted when executing
the actual factory/deployment code; contract bodies and validators are intact.
This deliberately does not establish native import or physical-motion safety.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import math
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.tcp_pose_comparison import (
    build_tcp_pose_comparison_executor,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.runtime.passive_read_executor import PassiveReadResult
from robot_motion.action import ComponentAction
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.relative_action import xyzquat_to_pose_matrix

pytestmark = [pytest.mark.unit, pytest.mark.contract]
REPO = Path(__file__).resolve().parents[1]
BASE_COMMIT = "79b6b90c953e7b965293f3cc3407529a4486cb5c"
PROFILE_DIR = "agent_gateway/profiles/mujoco/"

# Filled once from the pinned Git blobs, using this deterministic fixture.
# Direct frozen blob SHA-256 (not candidate files):
# motion_factory.py: ceedce05b5f7746baebdfc47eb6531a031a251bec19078441ea85e92ef168878
# motion_deployment.py: 32d9d1987f0b1851fec08e81cc75b93f9d1b3c9d4b6ff7f7dcc67ee14c901a93
# motion_primitive_profiles.py: ccbe3c9f1f24b0fe56dc62d37aa07bcff49add9787d99a445dcc4b1cbe77587d
# state_deployment.py: 011ba6f9dc7e01254585ef60e8ef7a88a87bfab71912ab03e53ddc294946e62b
# component_action_profile.py: 80707f7f53cfa29ad0c13b06362d339f37f00ae445c1c35012be68f1d276c965
# grasp_contact.py: afc585c1547918b6bbd84ba627278887638354e4c1b3564762bed732284f672b
# target_config.py: e6bb73994768f970c9e7a52078273f7d06429412a8e907ff982e533a8af8b284
# The other 252 imported project modules were byte-checked against the same
# commit; sorted path-to-SHA256 JSON digest:
# e14b3c840ef5474631c8ccb92c520aaf0e34661c466b69aa932b808066764579
RIGHT_EXPORT_SHA256 = {
    "False": {
        "cardinal_step_deployment_binding": "09999b4c4dc3c139697ce5200492bd4bccde76e60591cbe05d654f45851a632e",
        "cardinal_step_evaluation_plan": "d305268e11af328ef042d5747f05019c1a6fadeea1f3c1fcd81d37d07be67440",
        "cardinal_step_profile": "0c02bf0afd8b7acfec3c6658723eea2cb26c45d781aac0127267a9469481c01b",
        "definition_catalog": "f4b7f074c13ae87f6bd94c73f4d5c8cc180b157cb58a2e68ae2388b17be275d7",
        "deployment_binding": "372d10fda0804926ffff19fad0ae0aa3ca1d1d279adf5b176da0da5ff1dc2560",
        "envelopes": "3da8bbbb5889a8ba5580848d776026228985f5f3079c485432115e93fe9f64b3",
        "evaluation_plan": "ed6546371f35dd6ad810fe31b0f5294e113e403dd5eb567b4d5daa719e033c13",
        "fixed_cardinal_step_contracts": "c6ea7992bfd1ebee45fef7ba64e5b900765d48809704e7b2f93c06a35856d80e",
        "fixed_orientation_contracts": "d817307c19ca85c6d334973b4d3fcddb602d699a4ed0a07da48dcc8eda85fdd9",
        "history_deployment_binding": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
        "history_profile": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
        "history_source_binding": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
        "io_catalog": "35f673e5ee432d06827b63cc534adf82eecad914b6fc8cd0c1d49a44c81128b1",
        "motion_profile": "876203912755a998b862b3fcabced7edff68acdc31fe788f6ab38142258398c4",
        "observation_plan": "05bbddf27bc45d806ea1a09000d606c13d2af5586c9e1375e383aa68d41c350a",
        "route_manifest": "f55416192baec9029194efc2fa90533a592695bf0ce63752d78caf354fe80ca8",
        "safe_tcp_pose_contract": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
        "schema_catalog": "ab7a6bf1d14fbdbaab94871ed4b6b77e5101bde03607a472d355dde86f0cc8f6",
        "state_deployment_binding": "19e4f9075ae3630e8ee5833fac4ce6873681c6d23e166d04af893c2c25924ee1",
    },
    "True": {
        "cardinal_step_deployment_binding": "3eca82a9785b6f74f0275c3297fdf5f0db92c175c9bda630139bbb2d71e0a047",
        "cardinal_step_evaluation_plan": "ab9127b7dd1e0d096ecbd87d0b5ee0472e3f274d5cc21440968e6ffab611e0c6",
        "cardinal_step_profile": "0c02bf0afd8b7acfec3c6658723eea2cb26c45d781aac0127267a9469481c01b",
        "definition_catalog": "ded9c012a72e95854f9a948a84b75b23c335703bf8da131d582f1d3ce6dcdc0c",
        "deployment_binding": "e236e6db22e1cbee11bd6badf9fce13bf99f9c6670c72aa5b3860ee1a57a897a",
        "envelopes": "a2b10f80af683bdf5db6069a5cc8709fa94abe9480bb891226f96f2d46536cfc",
        "evaluation_plan": "a5e61f2c5ae725f8c8f0ec149be1856f190da0d2eac22c2c4427e14dc62935e5",
        "fixed_cardinal_step_contracts": "ff9ef2523884ab17e64ea31ce883e6a84ec6d4534993688eb91903c0ccc9f90a",
        "fixed_orientation_contracts": "d817307c19ca85c6d334973b4d3fcddb602d699a4ed0a07da48dcc8eda85fdd9",
        "history_deployment_binding": "1809a6bf090adb12d78d5f6f0e3575092d2ffc8ef2a938c829a3d6198b62f09d",
        "history_profile": "a9bfdf02ad01a6392b65aee35c4fef56549850b9229141a8d54f41044b9cd6e3",
        "history_source_binding": "0cab1a777693753b80bcb5504e152eb02969b32e997945f53c6da150cb29776f",
        "io_catalog": "831acfee47c2cb51842b3e71fde3a6adbcb018694f9e5180640c8eb7b155e699",
        "motion_profile": "876203912755a998b862b3fcabced7edff68acdc31fe788f6ab38142258398c4",
        "observation_plan": "05bbddf27bc45d806ea1a09000d606c13d2af5586c9e1375e383aa68d41c350a",
        "route_manifest": "c7622e10f6c6f729bc2d5126944b6cdea7609cec652c86766dd0f36f67431aa2",
        "safe_tcp_pose_contract": "1da303002543f4c5cf2366174e8dcccc268ec5875ac9213de57e8371bd1916c5",
        "schema_catalog": "0dd4a8b8ad8aff8c8a4c4e40dffe31a4531c70aaef92eef13a27f4e4d5f6fab0",
        "state_deployment_binding": "19e4f9075ae3630e8ee5833fac4ce6873681c6d23e166d04af893c2c25924ee1",
    },
}


# Explicit pose-goal catalogue migration. Preserve both original 79b6 export
# tables; only the five reviewed Safe-related extended exports change.
LEGACY_RIGHT_EXPORT_SHA256 = RIGHT_EXPORT_SHA256
RIGHT_EXPORT_SHA256 = {
    "False": LEGACY_RIGHT_EXPORT_SHA256["False"],
    "True": {**LEGACY_RIGHT_EXPORT_SHA256["True"],
        'definition_catalog': 'dd0bd376821b44f2dc447ab1461e3fb43b932f8777c120f5bf8282ec76ced8cf',
        'io_catalog': 'a6fa428bcbca79da7ba65ac60cdbd8f0916cd7e2fea43ba9d6db9dcdb269e43b',
        'route_manifest': 'd0699c7fc69bf1f3fe676bd30f8cccb6b6986c5f681202b5a0d15568acc34966',
        'safe_tcp_pose_contract': 'f398ac39fe893ec29edd7c9ceaef1e8fe3d82b3ef71eeaa7c69fb182a2e59832',
        'schema_catalog': 'c024295cb57ee0bfcf2afe82eb16c69afcfee9d2c733a928417358fe852ac6a7',
    },
}

# Preserve the historical 79b6 and pose-goal tables. Source 0236feec9's
# authorized archival removes only debug publications and their envelope route
# references. These current export pins were captured at 6b36561dc; unchanged
# motion/state/observation/history values still use the original pins above.
POSE_GOAL_RIGHT_EXPORT_SHA256 = RIGHT_EXPORT_SHA256
ARCHIVED_EXPORT_FIELDS = frozenset({
    "cardinal_step_profile", "cardinal_step_deployment_binding",
    "fixed_cardinal_step_contracts", "fixed_orientation_contracts",
    "cardinal_step_evaluation_plan",
})
ARCHIVAL_EXPORT_SHA256 = {
    "False": {
        "definition_catalog": "b0d675a18ca55750398a6234f8e55cbfe812f32c3901ad35abb038118ec6d3bc",
        "deployment_binding": "700ff2fe72ead6d84cad1a6a3058b9c71fc32cd36ccb628962f7b1722c5023cd",
        "envelopes": "6acab32bff1cdd87476f7141fb61455985b1ccdbb615796bdba593ca9669bfd5",
        "evaluation_plan": "20faaa199ce65e1d57aeea14afb76055ab3f5da2aa69730a9ed896101b74f058",
        "io_catalog": "df499eae76db1e3d925606dcf0d80cbb212a5f20505acd6ea819c76a88fcaa0c",
        "route_manifest": "d26d70240a4510b9142801675fa10cd3095bae9b8d2c973806da8c43bc7ede9d",
        "schema_catalog": "1e58d07c330dd4d384ab233f549f9b6b101b810579ee463505863ffd2167f190",
    },
    "True": {
        "definition_catalog": "799a0863dc45e854fe499874ecde94b7a93dc29eb7dd270dc11e7f1309fc2034",
        "deployment_binding": "d6f80ceba4d35419facb9757e54a78576a05c910178dc73e4fc7d918faf1ad81",
        "envelopes": "556e0f04ed29dc16c068875ae168c537fd3fe58182b9094f41dc0b42287db275",
        "evaluation_plan": "795d1f4e90a4a654d194f41325e74167a0d5f9fda9acc77933e9e881e76ff9c9",
        "io_catalog": "8824b14d06466659652e5101f347eabdabeb25411232cd24e9e2e8506daf4904",
        "route_manifest": "f9ea6d9edba44a9154731261f6e991aa7f13c5c65fc4151fdada555a8d2728a8",
        "safe_tcp_pose_contract": "8ae6e5ff65fc588ffe09b2450991ba5172da78ef0b0e3afed9b83022a0e8086e",
        "schema_catalog": "45e11050ac58924423cdba3d9841fc6acc52aafd670f4d95221940e2dca47a30",
    },
}
RIGHT_EXPORT_SHA256 = {
    extended: {**{key: value for key, value in exports.items() if key not in ARCHIVED_EXPORT_FIELDS},
        **ARCHIVAL_EXPORT_SHA256[extended]}
    for extended, exports in POSE_GOAL_RIGHT_EXPORT_SHA256.items()
}

# Derived from the actual pure definition module at BASE_COMMIT, independently
# of current catalog publication. Historical identities remain usable by readers.
HISTORICAL_DEBUG_DEFINITION_HASHES = {
    "motion.step_tcp_cardinal": "28c5d8cd1315785b4287fad3fe2e926cf17cc1650ebd0f99cee6451113550249",
    "motion.step_tcp_forward": "d23c0b7cdc09b2ea1bbf08d099e6a633c1c6b21a4ba85bf15907eb4b146d917d",
    "motion.step_tcp_backward": "a9f56b7aa65ec0d5a96699071e59835de29b73bc97bc389357a0054900a62a31",
    "motion.step_tcp_left": "ed81f6a8e7eb4d60f9924039e28d6ba4fa9120b8793eb69c793fd8b3c9743d09",
    "motion.step_tcp_right": "462f3670b5d2bf0c9d5740801718d8fb5c97fe28631a224ba5e63f32a84c581c",
    "motion.step_tcp_up": "82c39eea62541a7d78a9b6f9524e2767abd6433141c882219687e275d3361862",
    "motion.step_tcp_down": "4d1968a58cad0d7e821622a14d2b75e384348cb96e0c1da603350f271ef9e81f",
    "motion.set_tcp_level_forward": "f4cf378efdaa11ed45128b3ae7a1ba06b8f24fdfd8b8b5714ef34a213e00a954",
    "motion.set_tcp_level_forward_left_30": "54fbe9dee506a8f380620a86289d170219ac8dff1b9c58d66ca58d51834d9b9b",
    "motion.set_tcp_level_forward_right_30": "2e5b056963c9361136e4e954033bc607e4598cd69190e2d1dc07bd819417a964",
}


def _load_offline(
    monkeypatch, relative, *, omit=(), namespace=None, source=None, names=None
):
    """Execute unchanged Python bodies without importing resource-owning edges."""
    path = REPO / relative
    tree = ast.parse(path.read_bytes() if source is None else source, str(path))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module in omit)
    ]
    if names is not None:
        tree.body = [
            node
            for node in tree.body
            if getattr(node, "name", None) in names
            or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
    name = "_motion_side_contract_" + path.stem
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = relative.rsplit("/", 1)[0].replace("/", ".")
    module.__dict__.update(namespace or {})
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def _modules(monkeypatch, source_for=None):
    """Share only pure contracts; a native/resource import is a test failure."""
    original_import = builtins.__import__

    def no_native(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mujoco" or name.startswith(("mujoco.", "sim.mujoco")):
            raise AssertionError(f"Native/resource import in offline test: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_native)

    def load(relative, **kwargs):
        source = None if source_for is None else source_for(relative)
        return _load_offline(monkeypatch, relative, source=source, **kwargs)

    config = load("sim/mujoco/deploy/target_config.py")
    component = load("sim/mujoco/deploy/component_action_profile.py")
    contact = load("sim/mujoco/deploy/grasp_contact.py")
    state = load(PROFILE_DIR + "state_deployment.py")
    primitives = load(PROFILE_DIR + "motion_primitive_profiles.py")
    pure_contracts = {
        k: v for k, v in vars(primitives).items() if not k.startswith("__")
    }
    deployment = load(
        PROFILE_DIR + "motion_deployment.py",
        omit=(
            "sim.mujoco.deploy.grasp_contact",
            "visual_observation",
            "state_deployment",
            "motion_primitive_profiles",
        ),
        namespace={
            **pure_contracts,
            "MujocoGraspContactBinding": contact.MujocoGraspContactBinding,
            "MujocoPassiveTcpPoseStateDeploymentBinding": state.MujocoPassiveTcpPoseStateDeploymentBinding,
        },
    )
    factory = load(
        PROFILE_DIR + "motion_factory.py",
        omit=(
            "sim.mujoco.deploy",
            "sim.mujoco.deploy.grasp_contact",
            "sim.mujoco.deploy.target_config",
            "sim.mujoco.tasks.cube_lift.config",
            "gripper_skill",
            "motion_application",
            "motion_gateway_session",
            "motion_deployment",
            "motion_primitive_profiles",
        ),
        namespace={
            **{k: v for k, v in vars(deployment).items() if not k.startswith("__")},
            "MujocoSimHILTargetConfig": config.MujocoSimHILTargetConfig,
            "DEFAULT_STATION_POSE_CONTRACT_PATH": config.DEFAULT_STATION_POSE_CONTRACT_PATH,
        },
    )
    return SimpleNamespace(factory=factory, deployment=deployment, component=component)


@pytest.fixture
def offline(monkeypatch):
    return _modules(monkeypatch)


def _side_ready(offline):
    for fn in (
        offline.factory.build_mujoco_agent_tcp_pose_gateway_session,
        offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1,
    ):
        parameter = inspect.signature(fn).parameters.get("motion_side")
        assert parameter is not None, f"Missing fixed-side interface: {fn.__name__}"
        assert parameter.default == "right"


def _application(offline, *, side="right", explicit=False, extended=False):
    component = f"{side}_arm"
    base = f"{side}_arm_base"
    profile = offline.component.build_mujoco_pose_component_target_profile(
        SimpleNamespace(arm_specs={"left": object(), "right": object()}),
        profile_id=f"fixture.agent_tcp_pose.{component}.v1",
        arm_side_by_component={component: side},
        arm_reference_frame_by_component={component: base},
        arm_controlled_frame_by_component={component: "tool_tcp"},
        selection_group_by_component={component: "arms"},
    )
    plan_kwargs = {"motion_side": side} if explicit else {}
    plan = offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1(
        **plan_kwargs
    )
    direct = SimpleNamespace(
        producer_kind="agent_skill", producer_id="contract-fixture", producer_epoch=1
    )
    config = SimpleNamespace(
        runtime_session_id="runtime:fixed-side-fixture",
        control_domain_id="domain:fixed-side-fixture",
        embodiment_instance_id="embodiment:fixed-side-fixture",
        clock_domain_id="clock:fixed-side-fixture",
        decision_hz=15.0,
        direct_component_proposal_source=direct,
        enable_primary_policy_authority=False,
        enable_composer_authority=False,
        enable_replay_authority=False,
        initial_control_mode="agent_skill",
        execution_observation_plan=plan,
    )
    runtime = SimpleNamespace(
        config=config,
        action_schema=profile.action_schema,
        catalog=SimpleNamespace(
            action_space=profile.action_schema.schema_id,
            semantic_content_hash="component-catalog:fixture",
        ),
        execution_observation_binding=SimpleNamespace(
            binding_id="observation-binding:fixture",
            plan_hash=plan.semantic_content_hash,
            semantic_content_hash="observation-binding-hash:fixture",
        ),
        execution_observation_capability=SimpleNamespace(
            semantic_content_hash="observation-capability:fixture",
            representation_by_component={component: "cartesian_pose"},
            schema_hash_by_component={component: "pose-schema:fixture"},
        ),
        authoritative_port=SimpleNamespace(
            profile=SimpleNamespace(
                selection_group_ids=("arms",),
                authority_policy_hash="authority-policy:fixture",
                decision_schedule_hash="decision-schedule:fixture",
            )
        ),
        clock=lambda: 10.0,
    )
    for name in (
        "component_target_capability",
        "target_runtime_action_contract",
        "execution_capability",
        "reference_activation_spec",
    ):
        setattr(runtime, name, SimpleNamespace(semantic_content_hash=name + ":fixture"))
    world, station = "mujoco_sim_hil/world", "mujoco_sim_hil/station"
    graph = CoordinateFrameGraph(
        graph_id="frame-graph:fixture",
        frames=(
            CoordinateFrameSpec(frame_id=base),
            CoordinateFrameSpec(
                frame_id=world, semantic_parameters={"frame_family": "simulation_world"}
            ),
            CoordinateFrameSpec(
                frame_id=station,
                semantic_parameters={
                    "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1
                },
            ),
        ),
        transforms=tuple(
            RigidFrameTransformSpec(
                transform_id=f"fixture:{source}-to-{target}",
                source_frame_id=source,
                target_frame_id=target,
                matrix_target_source=tuple(tuple(row) for row in np.eye(4)),
            )
            for source, target in ((base, world), (world, station))
        ),
    )
    application = SimpleNamespace(
        motion_side=side,
        component_id=component,
        runtime=runtime,
        task_config=SimpleNamespace(physics_hz=1000.0),
        calibration_hash="calibration:fixture",
        target_frame_profile=SimpleNamespace(
            semantic_content_hash=f"target-frame-profile:{side}:fixture",
            reference_frame_by_component={component: base},
            canonical_reference_frame_id=station,
            frame_graph=graph,
        ),
        tool_frame_profile=SimpleNamespace(
            semantic_content_hash=f"tool-config:{side}:fixture",
            binding_by_component={
                component: SimpleNamespace(
                    controlled_frame_id="tool_tcp",
                    semantic_content_hash=f"tool-binding:{side}:fixture",
                )
            },
        ),
        target=SimpleNamespace(
            planned_tcp_motion=SimpleNamespace(
                motion_side=side,
                component_id=component,
                planning_clearance_padding_m=0.002,
            )
        ),
    )
    envelope_kwargs = {"motion_side": side} if explicit else {}
    application.tcp_pose_envelope_runtime = (
        offline.factory._build_standard_tcp_pose_envelope_runtime(
            prepared=application,
            target_config=config,
            enable_safe_tcp_pose=extended,
            **envelope_kwargs,
        )
    )
    return application


def _bundle(
    offline,
    monkeypatch,
    *,
    side="right",
    explicit=False,
    extended=False,
    history=False,
    application=None,
):
    application = application or _application(
        offline, side=side, explicit=explicit, extended=extended
    )
    monkeypatch.setattr(
        offline.deployment,
        "build_control_execution_manifest",
        lambda **kwargs: SimpleNamespace(
            semantic_content_hash="execution-manifest:fixture"
        ),
    )
    return offline.deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        application=application,
        enable_safe_tcp_pose=extended,
        enable_tcp_pose_comparison=extended,
        history_profile=offline.deployment.MujocoAgentTcpPoseHistoryProfile()
        if history
        else None,
    )


def _payload(value):
    if hasattr(value, "to_wire"):
        return _payload(value.to_wire())
    if hasattr(value, "to_ref"):
        return _payload(value.to_ref())
    if hasattr(value, "to_payload"):
        return _payload(value.to_payload())
    if is_dataclass(value):
        return {
            field.name: _payload(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


def _export_bytes(bundle, application):
    exports = {
        name: getattr(bundle, name)
        for name in (
            "definition_catalog",
            "schema_catalog",
            "io_catalog",
            "route_manifest",
            "motion_profile",
            "deployment_binding",
            "state_deployment_binding",
            "evaluation_plan",
            "safe_tcp_pose_contract",
            "history_profile",
            "history_source_binding",
            "history_deployment_binding",
        )
    }
    exports["history_profile"] = (
        None
        if bundle.history_profile is None
        else bundle.history_profile.semantic_content_hash
    )
    exports["observation_plan"] = application.runtime.config.execution_observation_plan
    exports["envelopes"] = application.tcp_pose_envelope_runtime.catalog_entries
    return {
        name: json.dumps(
            _payload(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        for name, value in exports.items()
    }


def _parameters(bundle, entry):
    profile = bundle.motion_profile
    properties = bundle.schema_catalog.entry(
        entry.route.skill_id, entry.route.skill_version
    ).parameter_schema["properties"]
    values = {
        **{
            key: getattr(profile, key)
            for key in (
                "reference_frame_id",
                "reference_frame_spec_hash",
                "frame_graph_hash",
                "tool_config_hash",
                "calibration_hash",
            )
        },
        "controlled_frame_id": "tool_tcp",
        "pose_goal_policy": "within_arrival_tolerance",
        "component_id": profile.component_id,
        "component_ids": [profile.component_id],
        "max_measurement_age_ms": 250,
        "target_operation_id": None,
        "pose_targets_by_component": {
            profile.component_id: [0.2, -0.1, 0.3, 0, 0, 0, 1]
        },
    }
    return {key: values[key] for key in properties}


def _request(bundle, entry, parameters, sequence):
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="fixture:client",
        request_id=f"fixture:{sequence}",
        idempotency_key=f"fixture:key:{sequence}",
        client_sequence=sequence,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=sequence,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters,
        observation_snapshot_id=f"fixture:snapshot:{sequence}",
        requested_mode="sim_active",
        requested_timeout_ms=min(1000, route.max_operation_duration_ms),
    )


def _operation(request, name):
    return AgentSkillOperation(
        operation_id=name,
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
        status="failed",
        reason_code="fixture",
        station_revision_at_admission=request.expected_station_revision,
        current_station_revision=2,
        started_at=10.0,
        updated_at=11.8,
        deadline_at=20.0,
        execution_outcome="failed",
        task_outcome="not_assessed",
    )


@pytest.mark.parametrize(
    "extended", [False, True], ids=["default", "safe-state2-history"]
)
def test_default_right_exports_are_pinned_with_explicit_archival(offline, monkeypatch, extended):
    application = _application(offline, extended=extended)
    bundle = _bundle(
        offline,
        monkeypatch,
        extended=extended,
        history=extended,
        application=application,
    )
    actual = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in _export_bytes(bundle, application).items()
    }
    assert actual == RIGHT_EXPORT_SHA256[str(extended)]
    assert set(actual).isdisjoint(ARCHIVED_EXPORT_FIELDS)
    assert all(not hasattr(bundle, field) for field in ARCHIVED_EXPORT_FIELDS)
    for key in set(actual) - set(ARCHIVAL_EXPORT_SHA256[str(extended)]):
        assert actual[key] == POSE_GOAL_RIGHT_EXPORT_SHA256[str(extended)][key]


@pytest.mark.parametrize("skill_id,expected_hash", sorted(HISTORICAL_DEBUG_DEFINITION_HASHES.items()))
def test_archived_definitions_keep_their_original_79b6_identity(offline, monkeypatch, skill_id, expected_hash):
    from agent_gateway.skills.motion import definition as module

    if skill_id == "motion.step_tcp_cardinal":
        definition = module.cardinal_tcp_step_skill_definition_v1(controlled_component_ids=("right_arm",))
    elif skill_id in module.FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1:
        definition = module.fixed_cardinal_tcp_step_skill_definition_v1(
            skill_id=skill_id, controlled_component_ids=("right_arm",))
    else:
        definition = module.fixed_level_tcp_orientation_skill_definition_v1(skill_id=skill_id)
    assert definition.skill_id == skill_id and definition.skill_version == "1"
    assert definition.semantic_content_hash == expected_hash
    assert set(HISTORICAL_DEBUG_DEFINITION_HASHES) == module.ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1
    bundle = _bundle(offline, monkeypatch, extended=True)
    for catalog, getter in ((bundle.route_manifest, "route"),
            (bundle.definition_catalog, "definition"), (bundle.schema_catalog, "entry"),
            (bundle.io_catalog, "contract")):
        with pytest.raises(KeyError):
            getattr(catalog, getter)(skill_id, "1")


@pytest.mark.parametrize("side", ["right", "left"])
def test_all_advertised_motion_and_state_contracts_bind_one_side(
    offline, monkeypatch, side
):
    _side_ready(offline)
    application = _application(offline, side=side, explicit=True, extended=True)
    bundle = _bundle(
        offline,
        monkeypatch,
        side=side,
        explicit=True,
        extended=True,
        application=application,
    )
    component = f"{side}_arm"
    assert application.runtime.action_schema.component_ids == (component,)
    assert application.runtime.config.execution_observation_plan.component_ids == (
        component,
    )
    assert bundle.state_deployment_binding.component_id == component
    for schema in bundle.schema_catalog.schema_entries:
        properties = schema.parameter_schema["properties"]
        if "pose_targets_by_component" in properties:
            assert set(properties["pose_targets_by_component"]["properties"]) == {
                component
            }
        elif "component_ids" in properties:
            assert tuple(properties["component_ids"]["items"]["enum"]) == (component,)
        else:
            assert tuple(properties["component_id"]["enum"]) == (component,)
        assert tuple(properties["reference_frame_id"]["enum"]) == (
            bundle.motion_profile.reference_frame_id,
        )
        assert properties["controlled_frame_id"]["const"] == "tool_tcp"
    contracts = [
        (bundle.motion_profile, bundle.deployment_binding),
        (
            bundle.safe_tcp_pose_contract.motion_profile,
            bundle.safe_tcp_pose_contract.deployment_binding,
        ),
    ]
    for profile, binding in contracts:
        assert profile.component_id == binding.component_id == component
        assert profile.execution_reference_frame_id == f"{side}_arm_base"
        assert binding.reference_frame_id == profile.reference_frame_id
        assert (
            binding.frame_graph_hash
            == profile.frame_graph_hash
            == application.target_frame_profile.frame_graph.semantic_content_hash
        )
        assert (
            binding.tool_config_hash
            == profile.tool_config_hash
            == application.tool_frame_profile.semantic_content_hash
        )
    for binding in (bundle.state_deployment_binding,):
        assert binding.reference_frame_id == bundle.motion_profile.reference_frame_id
        assert (
            binding.tool_config_hash
            == application.tool_frame_profile.semantic_content_hash
        )
    bundle.payload_validators.validate_manifest(bundle.route_manifest)
    right = _bundle(offline, monkeypatch, extended=True)
    authority_fields = (
        "control_effect_class",
        "privilege_tier",
        "supported_modes",
        "target_kind",
        "may_move_robot",
        "may_open_publication",
        "requires_operator_confirmation",
        "requires_deadman",
        "max_operation_duration_ms",
        "cancellation_policy",
        "requires_task_evaluation",
    )
    for entry in bundle.route_manifest.skill_routes:
        original = right.route_manifest.route(
            entry.route.skill_id, entry.route.skill_version
        ).route
        assert tuple(getattr(entry.route, name) for name in authority_fields) == tuple(
            getattr(original, name) for name in authority_fields
        )
    assert (
        bundle.route_manifest.route(
            "state.read_tcp_pose", "1"
        ).route.semantic_content_hash
        != bundle.route_manifest.route(
            "state.read_tcp_pose", "2"
        ).route.semantic_content_hash
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_definitions_match_selected_side_preserving_legacy_right_bytes(
    offline, monkeypatch, side
):
    _side_ready(offline)
    bundle = _bundle(offline, monkeypatch, side=side, explicit=True, extended=True)
    mismatches = {}
    for definition in bundle.definition_catalog.skill_definitions:
        expected = (f"{side}_arm",)
        if definition.controlled_component_ids != expected:
            mismatches[definition.skill_id] = definition.controlled_component_ids
    assert not mismatches


@pytest.mark.parametrize("side", ["left", "right"])
def test_every_advertised_route_rejects_wrong_component_frame_and_tool(
    offline, monkeypatch, side
):
    _side_ready(offline)
    bundle = _bundle(offline, monkeypatch, side=side, explicit=True, extended=True)
    other = "right_arm" if side == "left" else "left_arm"
    seen = set()
    for entry in bundle.route_manifest.skill_routes:
        route = entry.route
        seen.add((route.skill_id, route.skill_version))
        good = _parameters(bundle, entry)
        validator = bundle.payload_validators.validator_for(route)
        validator.validate_parameters(good)
        if "pose_targets_by_component" in good:
            wrong = {other: next(iter(good["pose_targets_by_component"].values()))}
            mutations = [
                ("pose_targets_by_component", wrong),
                (
                    "pose_targets_by_component",
                    {**wrong, **good["pose_targets_by_component"]},
                ),
            ]
        elif "component_id" in good:
            mutations = [("component_id", other)]
        else:
            mutations = [
                ("component_ids", [other]),
                ("component_ids", [f"{side}_arm", other]),
            ]
        mutations += [
            (key, "foreign:" + str(good[key]))
            for key in (
                "reference_frame_id",
                "reference_frame_spec_hash",
                "controlled_frame_id",
                "tool_config_hash",
                "frame_graph_hash",
                "calibration_hash",
            )
            if key in good
        ]
        for key, value in mutations:
            with pytest.raises((TypeError, ValueError)):
                validator.validate_parameters({**good, key: value})
    expected = {
        ("motion.move_tcp_to_pose", "1"),
        ("motion.move_tcp_to_pose_safe", "1"),
        ("state.read_tcp_pose", "1"),
        ("state.read_tcp_pose", "2"),
    }
    assert seen == expected


def test_left_state2_uses_one_fresh_cut_and_only_the_retained_target_component(
    offline, monkeypatch
):
    _side_ready(offline)
    application = _application(offline, side="left", explicit=True, extended=True)
    bundle = _bundle(
        offline,
        monkeypatch,
        side="left",
        explicit=True,
        extended=True,
        application=application,
    )
    motion = bundle.route_manifest.route("motion.move_tcp_to_pose_safe", "1")
    source_request = _request(bundle, motion, _parameters(bundle, motion), 0)
    source_operation = _operation(source_request, "fixture:retained-left")
    state = bundle.route_manifest.route("state.read_tcp_pose", "2")
    parameters = {
        **_parameters(bundle, state),
        "target_operation_id": source_operation.operation_id,
    }
    request = _request(bundle, state, parameters, 1)
    binding = bundle.state_deployment_binding
    measured = {
        **{
            key: getattr(binding, key)
            for key in (
                "reference_frame_id",
                "reference_frame_spec_hash",
                "frame_graph_hash",
                "tool_config_hash",
                "calibration_hash",
                "observation_binding_id",
                "observation_plan_hash",
            )
        },
        "component_ids": ["left_arm"],
        "controlled_frame_id": "tool_tcp",
        "observation_cut_id": "fixture:cut",
        "observation_cut_hash": "1" * 64,
        "cutoff_at": 12.0,
        "cutoff_clock_domain_id": application.runtime.config.clock_domain_id,
        "measurement_id": "fixture:measurement",
        "measurement_observation_hash": "2" * 64,
        "source_measurement_hash": "3" * 64,
        "measured_at": 11.9,
        "available_at": 11.95,
        "clock_domain_id": application.runtime.config.clock_domain_id,
        "pose_by_component": {"left_arm": [0.21, -0.1, 0.3, 0, 0, 0, 1]},
        "pose_schema_hash_by_component": {"left_arm": binding.pose_schema_hash},
        "representation_by_component": {"left_arm": "cartesian_pose"},
    }
    reads = []

    def capture(parameters, operation_id):
        reads.append((parameters, operation_id))
        return PassiveReadResult(dict(measured), ("fixture:one-cut",))

    executor = build_tcp_pose_comparison_executor(
        registration=bundle.tcp_pose_comparison_registration,
        capture_state=capture,
        lookup_target=lambda operation_id: (source_request, source_operation),
        route_manifest=bundle.route_manifest,
        payload_validators=bundle.payload_validators,
        clock_domain_id=application.runtime.config.clock_domain_id,
        current_station_revision=lambda: 2,
        clock=lambda: 12.0,
        max_retained_operations=4,
    )
    result = executor._read(
        request, _operation(request, "fixture:current-read")
    ).execution_result
    assert len(reads) == 1
    assert {
        key: value for key, value in result.items() if key != "target_comparison"
    } == measured
    comparison = result["target_comparison"]
    for key in (
        "target_pose_by_component",
        "translation_error_m_by_component",
        "rotation_error_rad_by_component",
    ):
        assert set(comparison[key]) == {"left_arm"}
    assert comparison["target_request_hash"] == source_request.semantic_content_hash
    assert comparison["target_status"] == "failed"
    assert comparison["target_task_outcome"] == "not_assessed"
    assert comparison["translation_error_m_by_component"]["left_arm"] == pytest.approx(
        0.01
    )
    # Freshness remains a property of this cut, not of a retained terminal time.
    measured["measured_at"] = 11.0
    with pytest.raises(ValueError):
        executor._read(request, _operation(request, "fixture:stale-read"))


def test_explicit_right_is_byte_identical_to_default(offline, monkeypatch):
    _side_ready(offline)
    for extended in (False, True):
        default = _application(offline, extended=extended)
        explicit = _application(offline, explicit=True, extended=extended)
        old = _bundle(
            offline,
            monkeypatch,
            extended=extended,
            history=extended,
            application=default,
        )
        new = _bundle(
            offline,
            monkeypatch,
            explicit=True,
            extended=extended,
            history=extended,
            application=explicit,
        )
        assert _export_bytes(old, default) == _export_bytes(new, explicit)


@pytest.mark.parametrize("side", ["left", "right"])
def test_selected_envelope_keeps_scalar_limits_and_real_binding(offline, side):
    _side_ready(offline)
    right = _application(offline, explicit=True, extended=True)
    selected = _application(offline, side=side, explicit=True, extended=True)
    runtime = selected.tcp_pose_envelope_runtime
    assert runtime.reference_frame_by_side == {side: f"{side}_arm_base"}
    assert runtime.tool_frame_binding_hash_by_side == {
        side: f"tool-binding:{side}:fixture"
    }
    for old, new in zip(
        right.tcp_pose_envelope_runtime.catalog_entries,
        runtime.catalog_entries,
        strict=True,
    ):
        assert old.allowed_route_ids == new.allowed_route_ids
        assert set(new.profile.side_config_by_side) == {side}
        assert _payload(old.profile.side_config_by_side["right"]) == _payload(
            new.profile.side_config_by_side[side]
        )
        if side == "left":
            assert (
                old.profile.target_frame_binding_hash
                != new.profile.target_frame_binding_hash
            )
            assert old.semantic_content_hash != new.semantic_content_hash


@pytest.mark.parametrize("exposed", [("left", "right"), ("left",)])
def test_default_factory_rejects_both_or_wrong_side_before_prepare(
    offline, monkeypatch, exposed
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid configuration reached Scene preparation")

    monkeypatch.setattr(
        offline.factory, "prepare_mujoco_sim_hil_target", forbidden, raising=False
    )
    config = offline.factory.MujocoSimHILTargetConfig(
        task_config_path=Path("never-opened.json"),
        realtime_worker=False,
        allow_scripted_task_reset=False,
        exposed_pose_sides=exposed,
        execution_observation_plan=offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1(),
    )
    with pytest.raises(ValueError):
        offline.factory.build_mujoco_agent_tcp_pose_gateway_session(
            target_config=config, allowed_client_ids=("fixture",)
        )


@pytest.mark.parametrize("bad", ["both", "", "LEFT", "left ", None, True])
def test_invalid_explicit_side_is_rejected_before_prepare(offline, monkeypatch, bad):
    _side_ready(offline)
    monkeypatch.setattr(
        offline.factory,
        "prepare_mujoco_sim_hil_target",
        lambda *a, **k: pytest.fail("Scene preparation"),
        raising=False,
    )
    config = offline.factory.MujocoSimHILTargetConfig(
        task_config_path=Path("never-opened.json"),
        allow_scripted_task_reset=False,
        exposed_pose_sides=("right",),
        execution_observation_plan=offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1(),
    )
    with pytest.raises((TypeError, ValueError)):
        offline.factory.build_mujoco_agent_tcp_pose_gateway_session(
            target_config=config, allowed_client_ids=("fixture",), motion_side=bad
        )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mismatch", ["observation", "component", "tool"])
def test_contract_rejects_cross_side_or_tool_mismatch(
    offline, monkeypatch, side, mismatch
):
    _side_ready(offline)
    application = _application(offline, side=side, explicit=True)
    other = "right" if side == "left" else "left"
    if mismatch == "observation":
        application.runtime.config.execution_observation_plan = (
            offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1(
                motion_side=other
            )
        )
    elif mismatch == "component":
        application.runtime.action_schema = _application(
            offline, side=other, explicit=True
        ).runtime.action_schema
    else:
        application.tool_frame_profile.binding_by_component[
            f"{side}_arm"
        ].controlled_frame_id = "flange"
    with pytest.raises((TypeError, ValueError, KeyError)):
        _bundle(offline, monkeypatch, side=side, explicit=True, application=application)


@pytest.mark.parametrize("side", ["left", "right"])
def test_explicit_side_rejects_exposure_or_observation_mismatch_before_prepare(
    offline, monkeypatch, side
):
    _side_ready(offline)
    other = "right" if side == "left" else "left"
    monkeypatch.setattr(
        offline.factory,
        "prepare_mujoco_sim_hil_target",
        lambda *a, **k: pytest.fail("Scene preparation"),
        raising=False,
    )
    config = offline.factory.MujocoSimHILTargetConfig(
        task_config_path=Path("never-opened.json"),
        allow_scripted_task_reset=False,
        exposed_pose_sides=(side,),
        execution_observation_plan=offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1(
            motion_side=side
        ),
    )
    for bad in (
        replace(config, exposed_pose_sides=(other,)),
        replace(config, exposed_pose_sides=("left", "right")),
        replace(config, allow_scripted_task_reset=True),
        replace(
            config,
            execution_observation_plan=offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1(
                motion_side=other
            ),
        ),
    ):
        with pytest.raises(ValueError):
            offline.factory.build_mujoco_agent_tcp_pose_gateway_session(
                target_config=bad, allowed_client_ids=("fixture",), motion_side=side
            )


@pytest.mark.parametrize("bad_side", [None, "left", "right"])
def test_left_tool_metadata_is_checked_before_prepare_including_inactive_arm(
    offline, monkeypatch, bad_side
):
    _side_ready(offline)
    calls = []
    task_loads = []
    # The shared loader owns both named profiles and path resolution.
    task = SimpleNamespace(
        left_tool_profile_path="fixture_left_tool_profile",
        right_tool_profile_path="./fixture/right_tool_profile.json",
    )

    def load_task(path):
        task_loads.append(path)
        return task

    monkeypatch.setattr(
        offline.factory,
        "CubeLiftTaskConfig",
        SimpleNamespace(load=load_task),
        raising=False,
    )

    def tool(path, *, side):
        assert type(path) is str
        calls.append((path, side))
        return {
            "arm": ("right" if side == "left" else "left") if side == bad_side else side
        }

    monkeypatch.setattr(offline.factory, "load_rm65_tool_profile", tool)

    class ReachedPrepare(Exception):
        pass

    def prepare(config):
        raise ReachedPrepare

    monkeypatch.setattr(
        offline.factory, "prepare_mujoco_sim_hil_target", prepare, raising=False
    )
    config = offline.factory.MujocoSimHILTargetConfig(
        task_config_path=Path("never-opened.json"),
        allow_scripted_task_reset=False,
        exposed_pose_sides=("left",),
        execution_observation_plan=offline.factory.mujoco_agent_tcp_pose_execution_observation_plan_v1(
            motion_side="left"
        ),
    )
    with pytest.raises(ReachedPrepare if bad_side is None else ValueError):
        offline.factory.build_mujoco_agent_tcp_pose_gateway_session(
            target_config=config, allowed_client_ids=("fixture",), motion_side="left"
        )
    assert task_loads == [config.task_config_path]
    expected_sides = ["left"] if bad_side == "left" else ["left", "right"]
    assert calls == [
        (getattr(task, f"{side}_tool_profile_path"), side) for side in expected_sides
    ]


@pytest.mark.parametrize("side", ["left", "right"])
def test_factory_passes_selected_side_and_measured_inactive_support_to_existing_ports(
    offline, monkeypatch, side
):
    """Spy at resource ports; run the actual factory and contract construction."""
    _side_ready(offline)
    app = _application(offline, side=side, explicit=True, extended=True)
    scene = SimpleNamespace(
        arm_specs={"left": object(), "right": object()}, gripper_specs={}
    )
    poses = {
        key: xyzquat_to_pose_matrix([x, -0.1, 0.3, 0, 0, 0, 1])
        for key, x in (("left", 0.2), ("right", 0.4))
    }
    prepared = SimpleNamespace(
        scene=scene,
        task_config=app.task_config,
        task_initialization_info={},
        target_frame_profile=app.target_frame_profile,
        tool_frame_profile=app.tool_frame_profile,
    )
    app.runtime.config.pose_frame_contract_hash = "fixture:pose-contract"
    app.runtime.rig = SimpleNamespace(capture_scene=lambda callback: callback(scene))
    config = offline.factory.MujocoSimHILTargetConfig(
        task_config_path=Path("never-opened.json"),
        allow_scripted_task_reset=False,
        exposed_pose_sides=(side,),
        execution_observation_plan=app.runtime.config.execution_observation_plan,
        clock_domain_id=app.runtime.config.clock_domain_id,
    )
    captured = {}
    monkeypatch.setattr(
        offline.factory,
        "prepare_mujoco_sim_hil_target",
        lambda config: prepared,
        raising=False,
    )
    monkeypatch.setattr(
        offline.factory,
        "CubeLiftTaskConfig",
        SimpleNamespace(
            load=lambda path: SimpleNamespace(
                left_tool_profile_path="fixture:left",
                right_tool_profile_path="fixture:right",
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        offline.factory, "load_rm65_tool_profile", lambda path, *, side: {"arm": side}
    )
    for name in (
        "build_mujoco_pose_component_target_profile",
        "MujocoComponentActionRobotActionV1Materializer",
    ):
        monkeypatch.setattr(
            offline.factory, name, getattr(offline.component, name), raising=False
        )
    monkeypatch.setattr(
        offline.factory,
        "build_mujoco_component_execution_binding",
        lambda profile, materializer: SimpleNamespace(
            action_schema=profile.action_schema, materializer=materializer
        ),
        raising=False,
    )
    monkeypatch.setattr(
        offline.factory,
        "MujocoExecutionDifferenceObserver",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        offline.factory,
        "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
        lambda *args: poses,
        raising=False,
    )

    def target(**kwargs):
        captured["target"] = kwargs
        app.runtime.action_schema = kwargs["component_execution_binding"].action_schema
        app.target = SimpleNamespace(
            runtime=app.runtime,
            task_config=app.task_config,
            scene=scene,
            target_frame_profile=app.target_frame_profile,
            tool_frame_profile=app.tool_frame_profile,
            tcp_pose_envelope_runtime=kwargs["tcp_pose_envelope_runtime"],
            planned_tcp_motion=app.target.planned_tcp_motion,
        )
        return app.target

    def application(**kwargs):
        captured["application"] = kwargs
        app.tcp_pose_envelope_runtime = kwargs["target"].tcp_pose_envelope_runtime
        app.calibration_hash = kwargs["calibration_hash"]
        return app

    monkeypatch.setattr(
        offline.factory, "build_mujoco_agent_skill_motion_target", target, raising=False
    )
    monkeypatch.setattr(
        offline.factory, "MujocoAgentSkillMotionApplication", application, raising=False
    )
    monkeypatch.setattr(
        offline.deployment,
        "build_control_execution_manifest",
        lambda **kwargs: SimpleNamespace(
            semantic_content_hash="execution-manifest:fixture"
        ),
    )
    monkeypatch.setattr(
        offline.factory,
        "assemble_mujoco_agent_tcp_pose_gateway_session",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    session = offline.factory.build_mujoco_agent_tcp_pose_gateway_session(
        target_config=config,
        allowed_client_ids=("fixture",),
        motion_side=side,
        enable_safe_tcp_pose=True,
        enable_tcp_pose_comparison=True,
    )
    assert (
        captured["target"]["motion_side"]
        == captured["application"]["motion_side"]
        == side
    )
    assert session.application is app
    assert session.contract_bundle.motion_profile.component_id == f"{side}_arm"
    materializer = captured["target"]["component_execution_binding"].materializer
    inactive = "right" if side == "left" else "left"
    assert materializer.support_arm_sides == (inactive,)
    assert dict(materializer.profile.arm_side_by_component) == {f"{side}_arm": side}
    support = materializer.support_arm_pose_provider()
    assert set(support) == {inactive}
    np.testing.assert_array_equal(support[inactive], poses[inactive])
    command = ComponentAction(
        materializer.profile.action_schema,
        {f"{side}_arm": [0.25, -0.1, 0.3, 0, 0, 0, 1]},
    )
    translated = materializer(command, timestamp=10.0, source="fixture")
    np.testing.assert_allclose(
        translated.arm_pose_by_side[inactive][:3], poses[inactive][:3, 3]
    )
    assert app.tcp_pose_envelope_runtime.snapshot().safety_reference_hash is not None


@pytest.mark.parametrize("side", ["left", "right"])
def test_inactive_joint_reference_and_original_drift_guard_are_preserved(
    offline, monkeypatch, side
):
    """Execute the real resolver with arrays/fake oracle, never a native Scene."""
    _side_ready(offline)
    planning = _load_offline(
        monkeypatch,
        "sim/mujoco/deploy/planned_tcp_motion.py",
        names={"MujocoTcpPlanRejected", "MujocoPlannedTcpMotion", "_scene_serialized"},
        namespace={
            "np": np,
            "math": math,
            "threading": threading,
            "time": time,
            "wraps": wraps,
            "xyzquat_to_matrix": xyzquat_to_pose_matrix,
            "IKResult": SimpleNamespace,
        },
    )
    addresses = {"left": np.arange(6), "right": np.arange(6, 12)}
    inactive = "right" if side == "left" else "left"
    qpos = np.arange(12, dtype=float) / 100
    scene = SimpleNamespace(data=SimpleNamespace(qpos=qpos.copy()))
    rig = SimpleNamespace(scene=scene, capture_scene=lambda callback: callback(scene))
    planner = planning.MujocoPlannedTcpMotion(
        rig=rig,
        task_config=object(),
        pose_projector=object(),
        reference_hz=60.0,
        motion_side=side,
    )
    reference_pose = [0.2, -0.1, 0.3, 0, 0, 0, 1]
    active_q = qpos[addresses[side]] + 0.002
    planner._active = SimpleNamespace(
        stages=(),
        reference_poses=(reference_pose,),
        trajectory=SimpleNamespace(q_rad=[active_q]),
        path=SimpleNamespace(
            start_qpos_by_component={
                key: qpos[addr].copy() for key, addr in addresses.items()
            }
        ),
        source_model_qpos=qpos.copy(),
        trajectory_hash="fixture:trajectory",
        payload_binding=None,
        contact_binding=None,
    )
    planner._reference_index = 0
    planner._live_stage = planner._active
    check = SimpleNamespace(safe=True)
    planner._live_oracle = SimpleNamespace(
        source=SimpleNamespace(qpos=qpos.copy()),
        addresses=addresses,
        configuration_constraint=None,
        check=lambda pair: check,
        projector=SimpleNamespace(
            project=lambda *a, **k: SimpleNamespace(modified=False, check=check)
        ),
    )
    action = SimpleNamespace(arm_pose_by_side={side: reference_pose})
    for drift in (0.0, 0.0009):
        scene.data.qpos[addresses[inactive][0]] = qpos[addresses[inactive][0]] + drift
        result = planner.resolve(action)
        np.testing.assert_array_equal(
            result.qpos_by_side[inactive], qpos[addresses[inactive]]
        )
        np.testing.assert_array_equal(result.qpos_by_side[side], active_q)
    scene.data.qpos[addresses[inactive][0]] = qpos[addresses[inactive][0]] + 0.0011
    with pytest.raises(
        planning.MujocoTcpPlanRejected, match="safe_tcp_passive_arm_changed"
    ):
        planner.resolve(action)
