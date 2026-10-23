"""Current native body measurements on a private copy under the existing lock.

No physics step, grasp establishment, release history or command inference. A
future bound release producer may supply the optional history through the
shared profile; this native single-cut source deliberately reports null.
"""

import numpy as np

from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw


class MujocoObjectStateSource:
    def __init__(self, *, application, profile, contact_binding):
        if profile.observation_source != "mujoco_native_physics" or profile.release_supported:
            raise ValueError("Native single-cut source cannot claim release-window evidence")
        if profile.object_id != contact_binding.target_body_name:
            raise ValueError("Observed object differs from the bound contact target")
        graph = application.target_frame_profile.frame_graph
        if (profile.frame_graph_hash != graph.semantic_content_hash
                or profile.reference_frame_spec_hash != graph.frame_by_id[profile.reference_frame_id].semantic_content_hash
                or profile.clock_domain_id != "mujoco-native-physics:" + application.runtime.config.runtime_session_id):
            raise ValueError("Object-state source differs from its deployed frame/clock")
        graph.resolve(MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, profile.reference_frame_id)
        self.application, self.profile, self.contact_binding = application, profile, contact_binding
        self._model = application.scene.model
        self._body = int(self._model.body(profile.object_id).id)
        self._support = int(self._model.body(application.task_config.table_body_name).id)
        self._sensors = {int(self._model.geom(name).id) for name in contact_binding.sensor_geom_names}
        self._target_geoms = set(np.flatnonzero((self._model.geom_bodyid == self._body)
            & ((self._model.geom_contype != 0) | (self._model.geom_conaffinity != 0))).tolist())
        if not self._target_geoms:
            raise ValueError("Observed target has no native collision geometry")
        self._scratch = None

    def capture(self):
        def project(scene):
            mj, model = scene.mujoco, scene.model
            if model is not self._model:
                raise ValueError("Object-state source model changed")
            if self._scratch is None:
                self._scratch = mj.MjData(model)
            data = self._scratch
            mj.mj_copyData(data, model, scene.data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.time):
                raise ValueError("Object-state native measurement is nonfinite")
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            mj.mj_collision(model, data)
            world_pose = np.eye(4)
            world_pose[:3, :3], world_pose[:3, 3] = data.xmat[self._body].reshape(3, 3), data.xpos[self._body]
            graph = self.application.target_frame_profile.frame_graph
            transform = graph.resolve(MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, self.profile.reference_frame_id)
            public_from_world = transform.matrix_target_source
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, jacr, self._body)
            touching = set()
            for contact in data.contact[:int(data.ncon)]:
                if contact.dist > 0:
                    continue
                first, second = int(contact.geom1), int(contact.geom2)
                if first in self._target_geoms and second not in self._target_geoms:
                    touching.add(second)
                elif second in self._target_geoms and first not in self._target_geoms:
                    touching.add(first)
            return {
                "measured_at": float(data.time), "cutoff_at": float(data.time),
                "pose_xyz_quat_xyzw": list(pose_to_xyz_quat_xyzw(public_from_world @ world_pose)),
                "linear_velocity_xyz_m_s": (public_from_world[:3, :3] @ jacp @ data.qvel).tolist(),
                "angular_velocity_xyz_rad_s": (public_from_world[:3, :3] @ jacr @ data.qvel).tolist(),
                "support_contact": any(int(model.geom_bodyid[g]) == self._support for g in touching),
                "gripper_contact": any(g in self._sensors for g in touching),
                "other_contact": any(g not in self._sensors and int(model.geom_bodyid[g]) != self._support for g in touching),
                "release_observation": None,
            }

        cut = self.application.capture_passive_application_cut(scene_projector=project)
        if cut.lifecycle_state not in {"running", "held"} or cut.scene_projection is None:
            raise RuntimeError("Object-state read requires the existing running or held application")
        return cut.scene_projection
