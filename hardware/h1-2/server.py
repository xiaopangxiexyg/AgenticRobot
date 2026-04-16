import argparse
import socket
import pickle

from arm_controller import H1ArmController
from hand_controller import H1HandController
from utils import *
from copy import deepcopy
import os, time

class Controller:
    def __init__(self, args):
        self.arm_controller = H1ArmController(topic_low_cmd='rt/lowcmd' if args.lowcmd else 'rt/arm_sdk',
                left_arm_down=args.left_arm_down)
        self.hand_controller = H1HandController()

    def set_arm_target(self, qpos, tau_ff=np.zeros(len(JointIndex))):
        self.arm_controller.set_qpos_target(qpos, tau_ff)

    def set_hand_target(self, left_hand_qpos, right_hand_qpos):
        self.hand_controller.set_qpos_target(left_hand_qpos, right_hand_qpos)

    def get_arm_state(self):
        return self.arm_controller.get_motor_state()

    def get_hand_state(self):
        return self.hand_controller.get_hand_state()

class StateBuffer:
    def __init__(self, start_episode=0):
        self.n_episode = start_episode
        self.save_dir = "data_{}".format(time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.save_dir, exist_ok=True)
        print("states save_dir:", self.save_dir)
        self.clear()
    
    def clear(self):
        self.buffer = []

    def add_state_dict(self, state):
        arm_qpos, arm_qvel = state["arm"]
        hand_qpos = state["hand"]
        s = np.zeros(26)
        s[0:14] = np.asarray(arm_qpos)
        s[14:26] = np.asarray(hand_qpos)
        s = np.concatenate([s, state["head"]])
        self.buffer.append(s)

    def save_episode(self):
        with open(os.path.join(self.save_dir, f"episode_{self.n_episode}.observation.state.pkl"), 'wb') as f:
            pickle.dump(self.buffer, f)
        self.n_episode += 1
        self.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.200.166") # wired network ip
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--save_state", action="store_true") # save episodes locally, not sending states to the client
    parser.add_argument("--episode", type=int, default=0) # starting episode number, for --save_state
    parser.add_argument("--lowcmd", action="store_true") # when the robot sits down, use lowcmd topic
    parser.add_argument("--left_arm_down", action="store_true") # set the left arm default pos towards downside
    args = parser.parse_args()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((args.ip, args.port))
    print('Server binded to', args.ip, args.port)
    robot_controller = Controller(args)
    if args.save_state:
        episode_state_buffer = StateBuffer(start_episode=args.episode)

    try:
        while True:
            #print("Waiting for data...")
            data, addr = server_socket.recvfrom(1024)
            if data:
                received_dict = pickle.loads(data)
                #print("Received dict:", received_dict)

                ### get_state request: the dict is {"request":"get_state"}
                if "request" in received_dict:
                    if received_dict["request"] == "get_state":
                        if args.save_state:
                            state = {
                                "arm": robot_controller.get_arm_state(), # [qpos,qvel]
                                "hand": robot_controller.get_hand_state(), # [qpos]
                                "head": received_dict["head_qpos"], # [qpos]
                            }
                            episode_state_buffer.add_state_dict(state)
                        else:
                            state = {
                                "arm": robot_controller.get_arm_state(), # [qpos,qvel]
                                "hand": robot_controller.get_hand_state(), # [qpos]
                            }
                            server_socket.sendto(pickle.dumps(state), addr)
                    ## when episode done, send the epsiode states to the client
                    elif received_dict["request"] == "save_episode":
                        if args.save_state:
                            episode_state_buffer.save_episode()
                        else:
                            raise Exception("Server cannot save episode when --save_state is not set.")
                    else:
                        raise NotImplementedError
                    
                ### send action request: the dict is {"arm":[qpos,tau_ff], "hand":[right_qpos, left_qpos]}
                else:
                    for key, value in received_dict.items():
                        if key == "arm":
                            robot_controller.set_arm_target(value[0], value[1])
                        elif key == "hand":
                            robot_controller.set_hand_target(value[0], value[1])
                        else:
                            print(f"Invalid key: {key}")

    except KeyboardInterrupt:
        print("Server stopped.")
        server_socket.close()


if __name__ == '__main__':
    main()
