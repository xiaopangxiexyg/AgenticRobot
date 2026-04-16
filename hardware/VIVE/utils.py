
import numpy as np
try:
    from .VDGloves.VDHand import DataStream, MocapHandDataProcessor
except ImportError:
    pass

from typing import Dict, Optional
from .mocap.vive_data_receiver import VIVEDataReceiver

class VIVEWristProcessor:
    """
    处理来自 VIVE Tracker 的数据，并将其转换为手腕的 4x4 位姿矩阵。
    
    该类会启动一个后台线程来持续接收 VIVE 发送的数据。
    """
    def __init__(self, port: int = 6666, num_floats: int = 36):
        """
        初始化 VIVE 数据接收器。

        Args:
            port (int): 监听 VIVE 数据的端口。
            num_floats (int): VIVE 发送的浮点数数量。
        """
        self.receiver = VIVEDataReceiver(port=port, num_floats=num_floats)
        self.receiver.start()
        print("VIVE data receiver started.")

    def get_wrist_pose(self) -> np.ndarray | None:
        """
        获取最新的手腕位姿。

        从 VIVE 的数据流中提取特定追踪器（默认为索引 [12:24] 的数据）
        的位置和旋转信息，并构建一个 4x4 的位姿矩阵。

        Returns:
            np.ndarray | None: 一个 4x4 的齐次变换矩阵，表示手腕在追踪器
                              坐标系下的位姿。如果没有接收到数据，则返回 None。
        """
        latest_data = self.receiver.get_latest_data()
        
        if latest_data is None:
            return None

        # 从原始数据中提取平移向量和旋转矩阵
        # 暂时只提取单个作为右手
        try:
            trans = np.array(latest_data[12:24])[:3]
            rot_matrix = np.array(latest_data[12:24])[3:].reshape(3, 3)
        except (IndexError, ValueError) as e:
            print(f"Error parsing VIVE data: {e}")
            return None

        # 构建 4x4 齐次变换矩阵
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = trans
        
        return pose_matrix
        
    def stop(self):
        """停止数据接收线程。"""
        self.receiver.stop()
        print("VIVE data receiver stopped.")

class VDGloveProcessor:
    """
    处理来自 Virtual Dynamics (VD) 手套的动捕数据，并将其转换为
    左右手的三维关键点。
    """
    def __init__(self, ip: str = "192.168.20.126", broadcast_port: int = 9998, local_port: int = 9999):
        """
        初始化 VD 手套数据流和处理器。

        Args:
            ip (str): 广播 IP 地址。
            broadcast_port (int): 广播端口。
            local_port (int): 本地监听端口。
        """
        self.stream = DataStream()
        self.stream.set_ip(ip)
        self.stream.set_broascast_port(broadcast_port)
        self.stream.set_local_port(local_port)
        
        self.processor = MocapHandDataProcessor()

    def connect(self) -> bool:
        """
        连接到 VD 手套的数据流。

        Returns:
            bool: 如果连接成功，返回 True，否则返回 False。
        """
        print("Connecting to VD Glove data stream...")
        is_connected = self.stream.connect()
        if is_connected:
            print("Successfully connected to VD Glove.")
        else:
            print("[ERROR] Failed to connect to VD Glove.")
        return is_connected

    def get_mocap_data(self):
        while True:
            self.stream.request_mocap_data()
            mocap_data = self.stream.get_mocap_data()

            if mocap_data.isUpdate:
                break
        return mocap_data

    def get_hand_keypoints(self) -> Optional[Dict[str, np.ndarray]]:
        """
        请求并获取最新的手部关键点数据。

        Returns:
            Optional[Dict[str, np.ndarray]]: 
                如果接收到新数据，返回一个包含 'left' 和 'right' 键的字典，
                值为对应的手部关键点 (21, 3) NumPy 数组。
                如果没有新数据，则返回 None。
        """
        self.stream.request_mocap_data()
        mocap_data = self.stream.get_mocap_data()

        if not mocap_data.isUpdate:
            return None # 没有新数据

        hand_data_dict = self.processor.process(mocap_data)
        
        keypoints = {}
        # 根据原始数据结构，分别转换左右手数据
        if "left_hand" in hand_data_dict:
            keypoints["left"] = self.processor.convert_left_hand_data_to_keypoints(hand_data_dict["left_hand"])
        
        if "right_hand" in hand_data_dict:
            keypoints["right"] = self.processor.convert_right_hand_data_to_keypoints(hand_data_dict["right_hand"])

        return keypoints