import time
import copy
import threading
import struct
import numpy as np

from unitree_dds_wrapper.idl import unitree_hg
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription

from utils import *


class H1ArmController:
    def __init__(self, topic_low_cmd="rt/lowcmd", topic_low_state="rt/lowstate", left_arm_down=False):
        """
        :param topic_low_cmd: dds_control_topic, should in ['rt/lowcmd', 'rt/arm_sdk']
        :param topic_low_state: state topic, default is rt/lowstate
        """
        assert topic_low_cmd in ['rt/lowcmd', 'rt/arm_sdk']
        self.low_cmd = True if topic_low_cmd == 'rt/lowcmd' else False
            
        print("Initialize ArmController...")
        self.q_des_list = np.zeros(num_motors)
        if left_arm_down:
            self.q_des_list[13:20] = [0.2,  0.2, -0.1,  0.7, 0, 0 , 0]
            #print(self.q_des_list)

        self.q_tau_ff = np.zeros(num_motors)
        self.msg = unitree_hg.msg.dds_.LowCmd_()

        if self.low_cmd:
            self._pack_fmt_low_cmd = '<2B2x' + 'B3x5fI' * 35 + '5I'
            self.msg.head = [0xFE, 0xEF]
        self.low_cmd_publisher = Publisher(unitree_hg.msg.dds_.LowCmd_, topic_low_cmd)
        self.low_state_subscriber = Subscription(unitree_hg.msg.dds_.LowState_, topic_low_state)

        self.motor_state_buffer = DataBuffer()
        self.motor_command_buffer = DataBuffer()
        self.base_state_buffer = DataBuffer()
        self.kp_low = 140.0
        self.kd_low = 7.5

        self.kp_high = 200.0
        self.kd_high = 5.0

        self.kp_wrist = 35.0
        self.kd_wrist = 6.0

        self.control_dt = 0.01
        self.hip_pitch_init_pos = -0.5
        self.knee_init_pos = 1.0
        self.ankle_init_pos = -0.5
        self.shoulder_pitch_init_pos = -1.4
        self.time = 0.0
        self.init_duration = 10.0
        self.report_dt = 0.1
        self.ratio = 0.0
        self.q_target = []

        while not self.low_state_subscriber.msg:
            print("low_state_subscriber is not ok! Please check dds.")
            time.sleep(0.01)
        
        self.msg.mode_pr = self.low_state_subscriber.msg.mode_pr if self.low_cmd else 0
        self.msg.mode_machine = self.low_state_subscriber.msg.mode_machine if self.low_cmd else 0
        
        duration = 1000
        init_q = np.array([self.low_state_subscriber.msg.motor_state[id].q for id in JointIndex])
        
        #self.report_rpy_thread = threading.Thread(target=self.subscribe_state)
        #self.report_rpy_thread.start()
        #time.sleep(5)
        #state = self.get_motor_state()
        #print(state)
        #sys.exit(0)


        if self.low_cmd:
            for id in JointIndex:
                self.msg.motor_cmd[id].q = self.low_state_subscriber.msg.motor_state[id].q
                self.q_target.append(self.msg.motor_cmd[id].q)
                print(f"Init q_pose is :{self.q_target}")
                print("Lock Leg...")
            for i in range(duration):
                time.sleep(0.001)
                q_t = init_q + (self.q_target - init_q) * i / duration
                for i, id in enumerate(JointIndex):
                    self.msg.motor_cmd[id].mode = 1
                    if id not in ArmJointIndex:
                        self.msg.motor_cmd[id].kp = 200
                        self.msg.motor_cmd[id].kd = 5
                        self.msg.motor_cmd[id].q = q_t[i]
                self._pack_crc()
                self.low_cmd_publisher.msg = self.msg
                self.low_cmd_publisher.write()
            print("Lock Leg OK!")

        #print(self.q_des_list)
        self.set_qpos_target(self.q_des_list, self.q_tau_ff)

        self.report_rpy_thread = threading.Thread(target=self.subscribe_state)
        self.report_rpy_thread.start()

        self.control_thread = threading.Thread(target=self.control)
        self.control_thread.start()

        command_writer = self.low_command_writer if self.low_cmd else self.arm_sdk_writer
        self.command_writer_thread = threading.Thread(target=command_writer)
        self.command_writer_thread.start()
        print("Initialize ArmController OK!")

    def low_state_handler(self, message):
        low_state = message
        self.record_motor_state(low_state)
        self.record_base_state(low_state)

    def set_qpos_target(self, q_des_list, q_tau_ff=np.zeros(len(JointIndex))):
        self.q_des_list = q_des_list
        self.q_tau_ff = q_tau_ff

    def _trans(self, pack_data):
        calc_data = []
        calc_len = ((len(pack_data) >> 2)-1)

        for i in range(calc_len):
            d = ((pack_data[i*4+3] << 24) | (pack_data[i*4+2] << 16) | (pack_data[i*4+1] << 8) | (pack_data[i*4]))
            calc_data.append(d)

        return calc_data

    def _crc32(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04c11db7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for _ in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1

        return crc

    def _pack_crc(self):
        original_data = []
        original_data.append(self.msg.mode_pr)
        original_data.append(self.msg.mode_machine)

        for i in range(num_motors):
            original_data.append(self.msg.motor_cmd[i].mode)
            original_data.append(self.msg.motor_cmd[i].q)
            original_data.append(self.msg.motor_cmd[i].dq)
            original_data.append(self.msg.motor_cmd[i].tau)
            original_data.append(self.msg.motor_cmd[i].kp)
            original_data.append(self.msg.motor_cmd[i].kd)
            original_data.append(self.msg.motor_cmd[i].reserve)

        original_data.extend(self.msg.reserve)
        original_data.append(self.msg.crc)
        calc_data = struct.pack(self._pack_fmt_low_cmd, *original_data)
        calc_data = self._trans(calc_data)
        self.msg.crc = self._crc32(calc_data)

    def low_command_writer(self):
        while True:
            mc_tmp_ptr = self.motor_command_buffer.data
            if mc_tmp_ptr:
                for i in ArmJointIndex:
                    self.msg.motor_cmd[i].tau = mc_tmp_ptr.tau_ff[i]
                    self.msg.motor_cmd[i].q = mc_tmp_ptr.q_ref[i]
                    self.msg.motor_cmd[i].dq = mc_tmp_ptr.dq_ref[i]
                    self.msg.motor_cmd[i].kp = mc_tmp_ptr.kp[i]
                    self.msg.motor_cmd[i].kd = mc_tmp_ptr.kd[i]
                self._pack_crc()
                self.low_cmd_publisher.msg = self.msg
                self.low_cmd_publisher.write()
            time.sleep(0.002)
    
    def arm_sdk_writer(self):
        while True:
            mc_tmp_ptr = self.motor_command_buffer.data
            if mc_tmp_ptr:
                for i in JointIndex:
                    self.msg.motor_cmd[i].tau = mc_tmp_ptr.tau_ff[i]
                    self.msg.motor_cmd[i].q = mc_tmp_ptr.q_ref[i] if i != arm_sdk_used_joint_index else 1
                    self.msg.motor_cmd[i].dq = mc_tmp_ptr.dq_ref[i]
                    self.msg.motor_cmd[i].kp = mc_tmp_ptr.kp[i]
                    self.msg.motor_cmd[i].kd = mc_tmp_ptr.kd[i]
                self.low_cmd_publisher.msg = self.msg
                self.low_cmd_publisher.write()
            time.sleep(0.002)

    def control(self):
        while True:
            ms_tmp_ptr = self.motor_state_buffer.data
            if ms_tmp_ptr:
                tem_q_desList = copy.deepcopy(self.q_des_list)
                tem_q_tau_ff = copy.deepcopy(self.q_tau_ff)
                motor_command_tmp = MotorCommand()
                self.time += self.control_dt
                self.time = min(max(self.time, 0.0), self.init_duration)
                self.ratio = self.time / self.init_duration
                for i in range(num_motors):
                    if self.is_weak_motor(i):
                        motor_command_tmp.kp[i] = self.kp_low
                        motor_command_tmp.kd[i] = self.kd_low
                    elif self.is_wrist_motor(i):
                        motor_command_tmp.kp[i] = self.kp_wrist
                        motor_command_tmp.kd[i] = self.kd_wrist
                    else:
                        motor_command_tmp.kp[i] = self.kp_high
                        motor_command_tmp.kd[i] = self.kd_high
                    motor_command_tmp.dq_ref[i] = 0.0
                    motor_command_tmp.tau_ff[i] = tem_q_tau_ff[i]
                    q_des = tem_q_desList[i]

                    q_des = (q_des - ms_tmp_ptr.q[i]) * self.ratio + ms_tmp_ptr.q[i]
                    motor_command_tmp.q_ref[i] = q_des
                self.motor_command_buffer.data = motor_command_tmp
            time.sleep(0.002)

    def get_motor_state(self):
        ms_tmp_ptr = self.motor_state_buffer.data
        if ms_tmp_ptr:
            return ms_tmp_ptr.q[13:27], ms_tmp_ptr.dq[13:27]
        else:
            return None, None

    def subscribe_state(self):
        while True:
            if self.low_state_subscriber.msg:
                self.low_state_handler(self.low_state_subscriber.msg)
            time.sleep(0.002)

    def record_motor_state(self, msg):
        ms_tmp = MotorState()
        for i in range(num_motors):
            ms_tmp.q[i] = msg.motor_state[i].q
            ms_tmp.dq[i] = msg.motor_state[i].dq
        self.motor_state_buffer.data = ms_tmp

    def record_base_state(self, msg):
        bs_tmp = BaseState()
        bs_tmp.omega = msg.imu_state.gyroscope
        bs_tmp.rpy = msg.imu_state.rpy
        self.base_state_buffer.data = bs_tmp

    def is_weak_motor(self, motor_index):
        weak_motors = [
            JointIndex.left_ankle,
            JointIndex.right_ankle,
            # Left arm
            JointIndex.left_shoulder_pitch,
            JointIndex.left_shoulder_roll,
            JointIndex.left_shoulder_yaw,
            JointIndex.left_elbow_pitch,
            # Right arm
            JointIndex.right_shoulder_pitch,
            JointIndex.right_shoulder_roll,
            JointIndex.right_shoulder_yaw,
            JointIndex.right_elbow_pitch,
        ]
        return motor_index in weak_motors

    def is_wrist_motor(self, motor_index):
        wrist_motors = [
            JointIndex.left_elbow_roll,
            JointIndex.left_wrist_pitch,
            JointIndex.left_wrist_yaw,
            JointIndex.right_elbow_roll,
            JointIndex.right_wrist_pitch,
            JointIndex.right_wrist_yaw,
        ]
        return motor_index in wrist_motors


if __name__ == '__main__':
    print(len(ArmJointIndex))
    arm_controller = H1ArmController()

    
