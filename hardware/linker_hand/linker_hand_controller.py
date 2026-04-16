import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float32MultiArray
import time, sys, os
import numpy as np
#sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
#from dex_retargeting.robot_wrapper import RobotWrapper

class LinkerHandController:
    def __init__(self, hand_side="right"):
        """
        初始化控制器
        :param hand_side: "left" 或 "right"，表示控制的灵巧手。
        """
        self.hand_side = hand_side
        self.control_topic = f"/cb_{hand_side}_hand_control_cmd"
        self.state_topic = f"/cb_{hand_side}_hand_state"
        self.touch_topic = f"/cb_{hand_side}_hand_touch"

        # 初始化ROS节点
        rospy.init_node('dexterous_hand_controller', anonymous=True)

        # 创建发布者用于控制手部
        self.control_pub = rospy.Publisher(self.control_topic, JointState, queue_size=10)

        # 创建订阅者用于获取手部状态
        rospy.Subscriber(self.state_topic, JointState, self.state_callback)
        
        # 更新触觉数据的订阅者
        rospy.Subscriber(self.touch_topic, Float32MultiArray, self.touch_callback)

        self.prepare_urdf_to_hardware()

        self.state_data = None
        self.touch_data = None
        while self.state_data is None or self.touch_data is None:
            time.sleep(0.1)

    def prepare_urdf_to_hardware(self):
        self.hardware_joint_names = ["thumb_joint2", "index_joint1", "middle_joint1", "ring_joint1", "little_joint1", "thumb_joint0", "index_joint0", "middle_joint0", "ring_joint0", "little_joint0", 
                                     "thumb_joint1", None, None, None, None, "thumb_joint3", "index_joint2", "middle_joint2", "ring_joint2", "little_joint2"]
        self.hardware_joint_limits_low = np.array([0, 0, 0, 0, 0, 0, 80, 80, 80, 80,
                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.hardware_joint_limits_high = np.array([255, 255, 255, 255, 255, 255, 176, 176, 176, 176,
                                            255, 255, 255, 255, 255, 255, 255, 255, 255, 255])
        self.hardware_joint_direction_sign = np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1, 1,
                                              1, None, None, None, None, -1, -1, -1, -1, -1])
        
        self.urdf_joint_names = ['index_joint0', 'index_joint1', 'index_joint2', 
                                 'little_joint0', 'little_joint1', 'little_joint2', 
                                 'middle_joint0', 'middle_joint1', 'middle_joint2', 
                                 'ring_joint0', 'ring_joint1', 'ring_joint2', 
                                 'thumb_joint0', 'thumb_joint1', 'thumb_joint2', 'thumb_joint3']
        self.urdf_joint_limits_low = np.array([-0.26, 0., -0.01204277, -0.26, 0., -0.01204277, -0.26, 0., -0.01204277, -0.26, 0., -0.01204277, -0.49131, -1.69, -0.05550147, -0.01204277])
        self.urdf_joint_limits_high = np.array([0.26, 1.57, 1.30969507, 0.26, 1.57, 1.30969507, 0.26, 1.57, 1.30969507, 0.26, 1.57, 1.30969507, 1.05016, 0., 0.78418, 1.30969507])
        self.urdf_to_hardware_indices = [self.hardware_joint_names.index(name) for name in self.urdf_joint_names]
        # self.hardware_action = np.array([255,255,255,255,255,
        #                                  255,128,128,128,128,
        #                                  255,0,0,0,0, 
        #                                  255,255,255,255,255],dtype=float)
    
    def urdf_to_hardware_action(self, action):
        action = np.clip(action, self.urdf_joint_limits_low, self.urdf_joint_limits_high)
        action_percentile = (action - self.urdf_joint_limits_low) / (self.urdf_joint_limits_high - self.urdf_joint_limits_low)
        action_percentile_hardware = np.where(self.hardware_joint_direction_sign[self.urdf_to_hardware_indices] > 0, action_percentile, 1-action_percentile)
        action_hardware = action_percentile_hardware * (self.hardware_joint_limits_high[self.urdf_to_hardware_indices] - self.hardware_joint_limits_low[self.urdf_to_hardware_indices]) \
            + self.hardware_joint_limits_low[self.urdf_to_hardware_indices]
        ret = np.zeros(20)
        ret[self.urdf_to_hardware_indices] = action_hardware
        return ret

    def normalize_hardware_state(self, state):
        return (state - self.hardware_joint_limits_low) / (self.hardware_joint_limits_high - self.hardware_joint_limits_low)

    def state_callback(self, msg):
        """
        订阅并处理手部状态数据（关节位置等）
        :param msg: 来自 /cb_right_hand_state 或 /cb_left_hand_state 的消息
        """
        self.state_data = msg
        #rospy.loginfo(f"Received state data: {msg.position}, {msg.effort}")

    def touch_callback(self, msg):
        """
        订阅并处理触觉数据
        :param msg: 来自 /cb_right_hand_touch 或 /cb_left_hand_touch 的消息
        """
        self.touch_data = msg.data
        #rospy.loginfo(f"Received touch data: {self.touch_data}")

    def send_command_hardware(self, positions):
        """
        发送控制命令来设置关节位置。
        :param positions: 目标关节角度的列表
        """
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = []  # 根据需要填充关节名称
        joint_state.position = positions
        joint_state.velocity = []
        joint_state.effort = []

        # 发布控制命令
        self.control_pub.publish(joint_state)
        #rospy.loginfo(f"Sent command to {self.hand_side} hand: {positions}")

    def get_state_hardware(self):
        """
        获取手部状态
        :return: 返回手部关节的状态数据
        """
        return list(self.state_data.position)

    def get_touch_data(self):
        """
        获取手部的触觉数据
        :return: 返回触觉数据
        """
        return list(self.touch_data)
    
    def get_3D_force(self, with_proximity=True):
        """
        返回(5,3)或(5,4)数据，表示五指的三维力(指尖link坐标系，x法向向内，z向上，y向侧)，单位N
        + 接近觉 0或1
        """
        data = np.array(self.get_touch_data())
        force_3d = np.zeros((5,3))
        force_3d[:,0] = -data[0:5]/255.*20. # normal force
        
        force_tangent = data[5:10]/255.*20.
        force_tangent_dir = np.clip(data[10:15], 0, 127)/127.*2*np.pi
        force_3d[:,1] = force_tangent * np.sin(force_tangent_dir)
        force_3d[:,2] = force_tangent * np.cos(force_tangent_dir)
        
        if not with_proximity:
            return force_3d
        else:
            proximity = np.where(data[15:20]>0, 1, 0)
            return np.concatenate([force_3d, proximity[:,None]], axis=1)

    def act(self, action):
        action_hardware = self.urdf_to_hardware_action(action)
        self.send_command_hardware(action_hardware)

    def get_state(self):
        return self.normalize_hardware_state(np.array(self.get_state_hardware()))

if __name__ == "__main__":
    # 初始化控制器对象
    hand_controller = LinkerHandController(hand_side="right")
    # 等待一些时间以确保ROS订阅器正常工作
    rospy.sleep(1)
    rate = rospy.Rate(10)

    action = np.array([255,255,255,255,255,
                       0,128,128,128,128,
                       255,0,0,0,0, 255,255,255,255,255],dtype=float)
    t = 0
    while not rospy.is_shutdown():
        state = hand_controller.get_state() #hand_controller.get_state_hardware()
        #touch_data = hand_controller.get_touch_data()
        force = hand_controller.get_3D_force(with_proximity=True)
        #action[16] = 180 + 50 * np.sin(t/10)
        hand_controller.send_command_hardware(action)
        print(state, force)
        rate.sleep()
        t += 1