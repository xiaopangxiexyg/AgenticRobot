import time
import threading
import numpy as np


from unitree_dds_wrapper.idl import unitree_go
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription


class H1HandController:
    def __init__(self, topic_hand_cmd="rt/inspire/cmd", topic_hand_state="rt/inspire/state"):
        self.cmd = unitree_go.msg.dds_.MotorCmds_()
        self.state = unitree_go.msg.dds_.MotorStates_()
        self.labels = {
            "open": np.ones(6),
            "close": np.zeros(6),
            "half": np.full(6, 0.5)
        }
        self.lock = threading.Lock()
        self.init_dds(topic_hand_cmd, topic_hand_state)

    def init_dds(self, topic_hand_cmd, topic_hand_state):
        self.hand_cmd = Publisher(unitree_go.msg.dds_.MotorCmds_, topic_hand_cmd)
        self.hand_state = Subscription(unitree_go.msg.dds_.MotorStates_, topic_hand_state)
        self.cmd.cmds = [unitree_go.msg.dds_.MotorCmd_() for _ in range(12)]
        self.state.states = [unitree_go.msg.dds_.MotorState_() for _ in range(12)]

        self.report_rpy_thread = threading.Thread(target=self.subscribe_state)
        self.report_rpy_thread.start()

    def subscribe_state(self):
        while True:
            if self.hand_state.msg:
                self.state = self.hand_state.msg
            time.sleep(0.01)

    def set_hand_state(self, label):
        """
        :param label: str, "open", "close", "half"
        """
        if label in self.labels:
            self._ctrl(self.labels[label], self.labels[label])
        else:
            print(f"Invalid label: {label}")

    def set_qpos_target(self, right_angles, left_angles):
        self._ctrl(right_angles, left_angles)

    def _ctrl(self, right_angles, left_angles):
        for i in range(6):
            self.cmd.cmds[i].q = right_angles[i]
            self.cmd.cmds[i+6].q = left_angles[i]
        self.hand_cmd.msg.cmds = self.cmd.cmds
        self.hand_cmd.write()

    def get_hand_state(self):
        with self.lock:
            q = np.array([self.state.states[i].q for i in range(12)])
            return q
        
    def get_right_q(self):
        with self.lock:
            q = np.array([self.state.states[i].q for i in range(6)])
            return q

    def get_left_q(self):
        with self.lock:
            q = np.array([self.state.states[i+6].q for i in range(6)])
            return q