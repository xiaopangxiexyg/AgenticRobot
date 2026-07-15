import numpy as np
try:
    from .VDGloves.VDHand import DataStream, MocapHandDataProcessor
except ImportError:
    pass

from typing import Dict, Optional

class VIVEWristProcessor:
    """
    旧版 float UDP VIVE wrapper。

    当前 VIVE 主链路已经收敛到：
    - 新 sender: hardware/VIVE/mocap/vive_data_sender.py
    - 稳定 receiver 兼容 sender: hardware/VIVE/mocap/vive_data_sender_legacy.py
    - receiver: teleop_runtime/sources/pose/vive.py
    """
    def __init__(self, port: int = 6666, num_floats: int = 36):
        raise RuntimeError(
            "VIVEWristProcessor used the removed legacy float UDP receiver. "
            "Use teleop_runtime.sources.pose.vive.VivePoseSource for receiving, "
            "and hardware/VIVE/mocap/vive_data_sender.py for current local sending "
            "(or vive_data_sender_legacy.py for a stable receiver)."
        )

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
