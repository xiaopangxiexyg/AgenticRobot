from .robotic_arm_package.robotic_arm import * 
import time
import numpy as np
#import pickle
#from tqdm import trange
import multiprocessing
import signal
from copy import deepcopy
import sys
from .utils import interpolate_vectors_linear
from hardware.inspire_tac.inspire_tac_modbus import InspireTacModbus
from dex_retargeting.robot_wrapper import RobotWrapper

class PinRobot:
    def __init__(self, path="../../assets/rm65_inspire_tac/rm65_arm.urdf"):
        path = os.path.join(os.path.dirname(__file__), path)
        self.robot = RobotWrapper(path)
        self.ee_idx = self.robot.get_link_index("hand_base")

    def get_min_singular_value(self, q):
        #print(self.robot.joint_names, self.robot.joint_limits)
        #self.robot.compute_forward_kinematics(np.array(q))
        ee_jacobian = self.robot.compute_single_link_local_jacobian(np.array(q), self.ee_idx)
        _, S, _ = np.linalg.svd(ee_jacobian)
        #print(S)
        return np.min(S)

"""
### 底层控制进程
对手臂：维护一个插值后的轨迹点队列，以一定底层控制频率不断透传下一个点，策略每做一次决策，控制器插值出从当前到目标的路径点，清空队列、替换成新的路径点
对灵巧手：每次策略决策后，直接将动作透传给灵巧手，不做插值
"""
def run_arm_hand_controller(arm_ip, hand_ip, max_fingertip_force, tgt_now, tgt_changed, current_tgt_done, lock, maxd, control_dt, 
                            stop_event, controller_ready_event, force_pos_control=False):
    signal.signal(signal.SIGINT, signal.SIG_IGN) # 忽略子进程的SIGINT信号, 由主进程发异常处理event
    try:
        # create arm connection
        robot = Arm(RM65, arm_ip)
        res, joint_now, pose, arm_err, sys_err = robot.Get_Current_Arm_State() # 40fps call
        try:
            assert res==0
        except:             # modified 2025/06/12
            print(f"error code: {res}")
        #print("Robot created. Initial state:", joint_now)
        tgt_now[:6] = [q/180.0*np.pi for q in joint_now] # update real arm state

        # create hand connection
        hand = InspireTacModbus(ip=hand_ip, max_fingertip_force=max_fingertip_force)
        if force_pos_control:
            res = robot.Clear_Force_Data(True)
            res = robot.Start_Force_Position_Move(True)
        
            # handle singularity
            pin_robot = PinRobot()
            force_mode = True
        
        waypoints = []
        waypoint_ptr = 0
        last_waypoint = deepcopy(tgt_now[:6])
        controller_ready_event.set()
        
        # low-level control loop
        while not stop_event.is_set():
            start_time = time.time()
            with lock:
                # the high-level policy takes one step
                if tgt_changed.value:
                    ## update the arm target
                    waypoints = interpolate_vectors_linear(
                        np.array(last_waypoint), np.array(tgt_now[:6]), maxd)
                    waypoint_ptr = 0
                    tgt_changed.value = False
                    #print("tgt now:", tgt_now[:])
                    current_tgt_done.value = False
                    ## control the hand
                    hand.set_joint_pos(np.array(tgt_now[6:]))
        
            # take an arm's low-level step
            if waypoint_ptr<len(waypoints):
                if force_pos_control:
                    # handle singularity
                    res, joint_now, _, _, _ = robot.Get_Current_Arm_State()
                    min_s = pin_robot.get_min_singular_value(np.array(joint_now)/180.0*np.pi) 
                    print("min_s:", min_s)
                    if min_s<=0.02:
                        if force_mode:
                            robot.Stop_Force_Position_Move(block=True)
                            force_mode = False
                        robot.Movej_Cmd(waypoints[-1][0:6]/np.pi*180, 5, block=True)
                        waypoint_ptr = len(waypoints)-1
                    else: # switch to force control
                        if not force_mode:
                            res = robot.Start_Force_Position_Move(True)
                            force_mode = True
                        res = robot.Force_Position_Move_Joint(waypoints[waypoint_ptr][0:6]/np.pi*180.0, 
                                                              sensor=1, mode=0, dir=0, force=0, follow=False)
                else:
                    robot.Movej_CANFD(waypoints[waypoint_ptr][0:6]/np.pi*180.0, False)
                last_waypoint = waypoints[waypoint_ptr]
                waypoint_ptr += 1
            else:
                current_tgt_done.value=True
            now_time = time.time()
            #print(control_dt - now_time+start_time)
            time.sleep(max(control_dt - now_time + start_time, 0))
    except Exception as e:
        print('Arm process exception:', e)
    finally:    
        # 断开连接
        robot.RM_API_UnInit()
        robot.Arm_Socket_Close()
        print('Arm connection closed.')
        hand.close()
        print('Hand connection closed.')


class RM65_Inspire_Tac_Controller:
    def __init__(self, 
                 arm_ip="192.168.1.20", 
                 hand_ip="192.168.11.210",
                 max_fingertip_force=800,
                 max_angvel=[0.2]*6, # max angvel for each arm joint, in rad/s
                 control_dt=0.02, # low-level control frequency for the arm
                 ):
        assert np.max(max_angvel)<1.58 # safety
        self.maxd = np.asarray(max_angvel) * control_dt
        self.control_dt = control_dt
        
        # shared vars
        self.tgt_now = multiprocessing.Array('f', [0.]*6+[1000.]*6) # arm-hand target
        self.tgt_changed = multiprocessing.Value('b', False) # signal to switch to a new target
        self.current_tgt_done = multiprocessing.Value('b', True) # current target finished
        self.lock = multiprocessing.Lock()
        self.stop_event = multiprocessing.Event()
        self.controller_ready_event = multiprocessing.Event()
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # mp
        self.controller_process = multiprocessing.Process(target=run_arm_hand_controller, 
            args=(arm_ip, hand_ip, max_fingertip_force, self.tgt_now, self.tgt_changed, self.current_tgt_done, 
                self.lock, self.maxd, self.control_dt, self.stop_event, 
                self.controller_ready_event))
        #self.controller_process.daemon = True
        self.controller_process.start()
        while not self.controller_ready_event.is_set():
            time.sleep(0.1)
        
        # create another robot in the main process for state listening
        self.listener = ArmHandListener(arm_ip, hand_ip, max_fingertip_force)

    def signal_handler(self, sig, frame):
        print("Ctrl+C detected. Shutting down...")
        self.stop()
    
    def stop(self):
        self.stop_event.set()
        self.listener.close()
        try:
            self.controller_process.join(timeout=3)       # Ensure the subprocess has terminated
        except multiprocessing.TimeoutError:
            print("Join timed out, force terminating the controller process.")
            self.controller_process.terminate()  # Force terminate if join fails
            self.controller_process.join()  # Ensure it is terminated
        print('Controller stopped')
        sys.exit(0)

    def move_to(self, q_tgt, block=True):
        self.set_target(q_tgt)
        if block:
            time.sleep(0.5)
            while not self.current_tgt_done.value:
                time.sleep(0.1)
            #print("movej done")

    def set_target(self, q_tgt):
        #print(q_tgt)
        with self.lock:
            self.tgt_now[:] = deepcopy(q_tgt)
            self.tgt_changed.value=True
            self.current_tgt_done.value = False
    
    def get_state(self, **kwargs):
        return self.listener.get_state(**kwargs)


class ArmHandListener:
    def __init__(self, arm_ip, hand_ip, max_fingertip_force):
        # create arm connection
        self.arm = Arm(RM65, arm_ip)
        self.hand = InspireTacModbus(ip=hand_ip, max_fingertip_force=max_fingertip_force)

    def close(self):
        self.arm.RM_API_UnInit()
        self.arm.Arm_Socket_Close()
        self.hand.close()
        print("Listener closed")
    
    def get_state(self, require_arm_6d_force=False):
        ret = {}
        get_current_arm_state = time.time()
        res, joint_now, pose, arm_err, sys_err = self.arm.Get_Current_Arm_State() 
        try:
            assert res==0
        except:             # modified 2025/06/12
            # print(f"error code: {res}")
            pass
        # print(f"get_current_arm_state: {time.time() - get_current_arm_state}")
        q_arm = np.asarray(joint_now) / 180.0 * np.pi
        ret["arm_joint_pos"] = q_arm

        get_hand_joint_and_tactile = time.time()
        q_hand = self.hand.read_joint_pos()
        tac_vec, tac_dict = self.hand.read_tac_data()             # modified 2025/06/20 
        # breakpoint()                                              # modified 2025/06/20 
        # tac_vec = np.zeros((1062,))                     # modified 2025/06/20 
        # tac_dict = {
        #     'little_1': np.zeros((3, 3)), 
        #     'little_2': np.zeros((12, 8)), 
        #     'little_3': np.zeros((10, 8)), 
        #     'ring_1': np.zeros((3, 3)), 
        #     'ring_2': np.zeros((12, 8)), 
        #     'ring_3': np.zeros((10, 8)), 
        #     'middle_1': np.zeros((3, 3)), 
        #     'middle_2': np.zeros((12, 8)), 
        #     'middle_3': np.zeros((10, 8)), 
        #     'index_1': np.zeros((3, 3)), 
        #     'index_2': np.zeros((12, 8)), 
        #     'index_3': np.zeros((10, 8)), 
        #     'thumb_1': np.zeros((3, 3)), 
        #     'thumb_2': np.zeros((12, 8)), 
        #     'thumb_3': np.zeros((3, 3)), 
        #     'thumb_4': np.zeros((12, 8)), 
        #     'palm': np.zeros((8, 14))
        # }                             # modified 2025/06/20 
        # print(f"get_hand_joint_and_tactile: {time.time() - get_hand_joint_and_tactile}")
        ret["hand_joint_pos"] = q_hand
        ret["hand_tac_vec"] = tac_vec
        ret["hand_tac_dict"] = tac_dict

        ret["eef_6d_pose"] = pose       # added 2025/06/13
        # [0.03761899843811989, 0.32555699348449707, 0.29548200964927673, -3.0460000038146973, 0.07500000298023224, -1.6519999504089355]

        get_force_time = time.time()
        if require_arm_6d_force:
            ret_code, force, _, _, _ = self.arm.Get_Force_Data()
            try:
                assert ret_code==0
            except:             # modified 2025/06/13
                # print(f"error code: {ret_code}")
                pass
            ret['arm_6d_force'] = force
        # print(f"get_force_time: {time.time() - get_force_time}")
        return ret