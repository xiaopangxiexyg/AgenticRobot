import time, os, sys, cv2
import open3d as o3d
from linker_hand_controller import LinkerHandController
import pinocchio as pin
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from dex_retargeting.robot_wrapper import RobotWrapper
import numpy as np
import rospy
import pyrealsense2 as rs
import multiprocessing
from hardware.realsense.color import color_format_value, color_image_to_bgr

def produce_frame(queue: multiprocessing.Queue):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, color_format_value(rs, "yuyv"), 30)
    pipeline.start(config)
    while True:#cap.isOpened():
        #success, image = cap.read()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = color_image_to_bgr(np.asanyarray(color_frame.get_data()), "yuyv")
        #if not success:
        #    continue
        frame = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if queue.full():
            queue.get()
        queue.put(image)
        time.sleep(1 / 30.0)
        cv2.imshow("demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def start_retargeting(queue: multiprocessing.Queue, use_sapien=True):
    from pathlib import Path
    import importlib
    from dex_retargeting.example.single_hand_detector import SingleHandDetector
    from dex_retargeting.constants import (
        RobotName,
        RetargetingType,
        HandType,
        get_default_config_path,
    )
    from dex_retargeting.retargeting_config import RetargetingConfig
    import sapien
    from sapien.asset import create_dome_envmap
    from sapien.utils import Viewer
    from queue import Empty

    robot_dir = Path(importlib.util.find_spec('dex_retargeting').origin).absolute().parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    override = dict(add_dummy_free_joint=True)
    config_path = get_default_config_path(RobotName.linker, RetargetingType.dexpilot, HandType.right)
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    detector = SingleHandDetector(hand_type="Right", selfie=False)

    if use_sapien:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        scene = sapien.Scene()
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.06, 0.08, 0.12, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.9
        render_mat.specular = 0.8
        scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])
        # Lighting
        scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)
        # Camera
        cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
        cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))
        # Viewer
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())

        # Load robot and set it to a good pose to take picture
        loader = scene.create_urdf_loader()
        filepath = Path(config.urdf_path)
        robot_name = filepath.stem
        loader.load_multiple_collisions_from_file = True
        filepath = str(filepath)
        robot = loader.load(filepath)
        robot.set_pose(sapien.Pose([0, 0, -0.1]))

        # Different robot loader may have different orders for joints
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        retargeting_joint_names = retargeting.joint_names
        retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    hand_controller = LinkerHandController(hand_side="right")
    visualizer = HandVisualizer()

    while not rospy.is_shutdown():
        start_time = time.time()
        try:
            rgb = queue.get(timeout=5)
        except Empty:
            raise Exception("Fail to fetch image from camera in 5 secs. Please check your web camera device.")

        _, joint_pos, _, _ = detector.detect(rgb)
        #print(joint_pos)
        if joint_pos is not None:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            #print(retargeting_type)
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            if use_sapien:
                robot.set_qpos(qpos[retargeting_to_sapien])
                viewer.render()
            #print(retargeting.optimizer.target_joint_names, retargeting.joint_limits)
            actuator_qpos = qpos[retargeting.optimizer.idx_pin2target][6:]
            all_qpos = qpos[6:]
            force = hand_controller.get_3D_force()[:,:3]
            hand_controller.act(actuator_qpos)
            visualizer.compute_link_pos(all_qpos)
            visualizer.render(force_3d=force)
        
        sleep_time = max(0.05 - (time.time() - start_time), 0)
        time.sleep(sleep_time)

class HandVisualizer:
    def __init__(self, urdf="../../assets/robots/hands/linker_hand/linker_hand_right.urdf"):
        self.robot = RobotWrapper(urdf)
        self.link_names = self.robot.link_names
        self.joint_names = self.robot.joint_names
        print(len(self.link_names), self.link_names, 
              len(self.joint_names), self.joint_names)
        self.fingertip_names = ["thumb_Link5", "index_Link4", "middle_Link4", "ring_Link4", 
                                "little_Link4"]

        ### Initialize Open3D Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)
        # link points
        self.pcd_links = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_links)
        # fingertip force arrows
        self.fingertip_force_lines, self.fingertip_coords = [], [] 
        for i in range(5):
            line_set = o3d.geometry.LineSet()
            points = np.zeros((2, 3))  # 2个点，初始设置为(0, 0, 0)
            lines = [[0, 1]]  # 线段连接点0和点1
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            self.fingertip_force_lines.append(line_set)
            self.vis.add_geometry(line_set)
            # fingertip coordinate
            # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[0, 0, 0])
            # self.fingertip_coords.append(coord)
            # self.vis.add_geometry(coord)

        # base coordinate
        base_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.vis.add_geometry(base_coord)
        view_control = self.vis.get_view_control()
        view_control.set_lookat([0, 0, 0.2])  # 观察点
        view_control.set_front([1, 0, 0.3])  # 前视方向
        view_control.set_up([0, 0, 1])  # 上方向设置为z轴方向
        view_control.set_zoom(1)  # 设置缩放比例以适应场景
    
    def compute_link_pos(self, joint_pos=[0]*21):
        self.robot.compute_forward_kinematics(np.asarray(joint_pos))
        self.link_pos = [self.robot.get_link_pose(i)[:3,3] for i in range(len(self.link_names))]
        self.fingertip_pose = [self.robot.get_link_pose(self.link_names.index(name)) for name in self.fingertip_names]
        #print(self.fingertip_pose)

    def render(self, force_3d=np.zeros((5,3))):
        # link positions
        points = np.array(self.link_pos)
        self.pcd_links.points = o3d.utility.Vector3dVector(points)

        # fingertip coords
        # for transform, coord in zip(self.fingertip_pose, self.fingertip_coords):
        #     print(dir(coord))
        #     coord.transform(transform)
        #     self.vis.update_geometry(coord)

        # force arrows
        for i, (transform, force_local) in enumerate(zip(self.fingertip_pose, force_3d)):
            fingertip_position = transform[:3, 3]
            force_world = transform[:3, :3] @ force_local / 10.  # 旋转矩阵 * 力向量 1N->10cm
            force_end_point = fingertip_position + force_world
            line_set = self.fingertip_force_lines[i]
            points = np.array([fingertip_position, force_end_point])
            line_set.points = o3d.utility.Vector3dVector(points)
            self.vis.update_geometry(line_set)

        self.vis.update_geometry(self.pcd_links)
        self.vis.poll_events()
        self.vis.update_renderer()
    

if __name__ == "__main__":
    queue = multiprocessing.Queue(maxsize=2)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue,))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

    
