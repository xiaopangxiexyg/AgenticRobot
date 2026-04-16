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

"""
### 底层控制进程
对手臂：维护一个插值后的轨迹点队列，以一定底层控制频率不断透传下一个点，策略每做一次决策，控制器插值出从当前到目标的路径点，清空队列、替换成新的路径点
对灵巧手：每次策略决策后，直接将动作透传给灵巧手，不做插值
"""
def run_arm_hand_controller(robot_ip, tgt_now, tgt_changed, current_tgt_done, lock, maxd, control_dt, 
                            stop_event, controller_ready_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN) # 忽略子进程的SIGINT信号, 由主进程发异常处理event
    try:
        # create arm connection
        robot = Arm(RM65, robot_ip)
        res, joint_now, pose, arm_err, sys_err = robot.Get_Current_Arm_State() # 40fps call
        # assert res==0
        #print("Robot created. Initial state:", joint_now)
        tgt_now[:6] = [q/180.0*np.pi for q in joint_now] # update real arm state

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
                    robot.Set_Hand_Angle(np.array(tgt_now[6:]).astype(int), block=False)
        
            # take an arm's low-level step
            if waypoint_ptr<len(waypoints):
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


class RM65_Inspire_Controller:
    def __init__(self, 
                 ip="192.168.1.19", 
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
            args=(ip, self.tgt_now, self.tgt_changed, self.current_tgt_done, 
                self.lock, self.maxd, self.control_dt, self.stop_event, 
                self.controller_ready_event))
        #self.controller_process.daemon = True
        self.controller_process.start()
        while not self.controller_ready_event.is_set():
            time.sleep(0.1)
        
        # create another robot in the main process for state listening
        self.listener = ArmListener(ip)

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
    
    def get_dof_pos(self, **kwargs):
        return self.listener.get_dof_pos(**kwargs)


class ArmListener:
    def __init__(self, ip):
        # create arm connection
        self.arm = Arm(RM65, ip)

    def close(self):
        self.arm.RM_API_UnInit()
        self.arm.Arm_Socket_Close()
        print("Listener closed")
    
    def get_dof_pos(self, require_force=False):
        res, joint_now, pose, arm_err, sys_err = self.arm.Get_Current_Arm_State() 
        # assert res==0
        q_arm = np.asarray(joint_now) / 180.0 * np.pi

        info = {}
        # other info
        if require_force:
            ret, force, _, _, _ = self.arm.Get_Force_Data()
            assert ret==0
            info['force'] = force
        return q_arm, info


