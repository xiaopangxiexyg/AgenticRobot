import time
import sys
import openvr
import math
import json

from functools import lru_cache
import transforms3d as tfs
import numpy as np

# Function to print out text but instead of starting a new line it will overwrite the existing line
def update_text(txt):
    sys.stdout.write('\r'+txt)
    sys.stdout.flush()

def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

def _normalize_serial(serial):
    return _to_text(serial).strip()

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Euler angles (in degrees)
def convert_to_euler(pose_mat):
    yaw = 180 / math.pi * math.atan2(pose_mat[1][0], pose_mat[0][0])
    pitch = 180 / math.pi * math.atan2(pose_mat[2][0], pose_mat[0][0])
    roll = 180 / math.pi * math.atan2(pose_mat[2][1], pose_mat[2][2])
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x,y,z,yaw,pitch,roll]

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Quaternion
def convert_to_quaternion(pose_mat):
    # Per issue #2, adding a abs() so that sqrt only results in real numbers
    r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
    r_x = (pose_mat[2][1]-pose_mat[1][2])/(4*r_w)
    r_y = (pose_mat[0][2]-pose_mat[2][0])/(4*r_w)
    r_z = (pose_mat[1][0]-pose_mat[0][1])/(4*r_w)

    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x,y,z,r_w,r_x,r_y,r_z]

def convert_to_quaternion_robust(pose_mat):
    """
    一个数值稳定的将 3x3 旋转矩阵转换为四元数 [w, x, y, z] 的函数。
    """
    # 位置信息
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]

    # 旋转信息
    m00, m01, m02 = pose_mat[0][0], pose_mat[0][1], pose_mat[0][2]
    m10, m11, m12 = pose_mat[1][0], pose_mat[1][1], pose_mat[1][2]
    m20, m21, m22 = pose_mat[2][0], pose_mat[2][1], pose_mat[2][2]

    trace = m00 + m11 + m22

    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    # 返回 [x, y, z, w, x, y, z]
    return [x, y, z, qw, qx, qy, qz]

def ctypes_arrays_to_ndarray(c_arrays_list):
    # 确定数组形状
    n_arrays = len(c_arrays_list)
    array_length = len(c_arrays_list[0])
    
    # 创建空的 NumPy 数组
    result = np.empty((n_arrays, array_length), dtype=np.float32)
    
    # 逐个复制数据
    for i, c_array in enumerate(c_arrays_list):
        # 使用 np.ctypeslib 直接访问内存
        np_array_view = np.ctypeslib.as_array(c_array)
        result[i] = np_array_view
    
    return result

# 使用函数
# ndarray_result = ctypes_arrays_to_ndarray(c_arrays_list)

def convert_to_quaternion_tfs(pose_mat):
    # print(pose_mat)
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    # print(pose_mat)
    pose_mat = ctypes_arrays_to_ndarray(pose_mat)
    pose_mat_3x3 = pose_mat[0:3, 0:3]
    # print(pose_mat_3x3.shape)
    qw, qx, qy, qz = tfs.quaternions.mat2quat(pose_mat_3x3)
    return [x, y, z, qw, qx, qy, qz]

def matrix_to_flat_list(pose_mat):
    pose_mat = ctypes_arrays_to_ndarray(pose_mat)
    x = pose_mat[0,3]
    y = pose_mat[1,3]
    z = pose_mat[2,3]
    pose_mat_3x3 = pose_mat[0:3, 0:3]
    flat_list = pose_mat_3x3.flatten().tolist()
    return [x, y, z] + flat_list
    

#Define a class to make it easy to append pose matricies and convert to both Euler and Quaternion for plotting
class pose_sample_buffer():
    def __init__(self):
        self.i = 0
        self.index = []
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.r_w = []
        self.r_x = []
        self.r_y = []
        self.r_z = []

    def append(self,pose_mat,t):
        self.time.append(t)
        self.x.append(pose_mat[0][3])
        self.y.append(pose_mat[1][3])
        self.z.append(pose_mat[2][3])
        self.yaw.append(180 / math.pi * math.atan(pose_mat[1][0] /pose_mat[0][0]))
        self.pitch.append(180 / math.pi * math.atan(-1 * pose_mat[2][0] / math.sqrt(pow(pose_mat[2][1], 2) + math.pow(pose_mat[2][2], 2))))
        self.roll.append(180 / math.pi * math.atan(pose_mat[2][1] /pose_mat[2][2]))
        r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
        self.r_w.append(r_w)
        self.r_x.append((pose_mat[2][1]-pose_mat[1][2])/(4*r_w))
        self.r_y.append((pose_mat[0][2]-pose_mat[2][0])/(4*r_w))
        self.r_z.append((pose_mat[1][0]-pose_mat[0][1])/(4*r_w))

def get_pose(vr_obj):
    return vr_obj.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)


class vr_tracked_device():
    def __init__(self,vr_obj,index,device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj

    @lru_cache(maxsize=None)
    def get_serial(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_SerialNumber_String)

    def get_model(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_ModelNumber_String)

    def get_battery_percent(self):
        return self.vr.getFloatTrackedDeviceProperty(self.index, openvr.Prop_DeviceBatteryPercentage_Float)

    def is_charging(self):
        return self.vr.getBoolTrackedDeviceProperty(self.index, openvr.Prop_DeviceIsCharging_Bool)


    def sample(self,num_samples,sample_rate):
        interval = 1/sample_rate
        rtn = pose_sample_buffer()
        sample_start = time.time()
        for i in range(num_samples):
            start = time.time()
            pose = get_pose(self.vr)
            rtn.append(pose[self.index].mDeviceToAbsoluteTracking,time.time()-sample_start)
            sleep_time = interval- (time.time()-start)
            if sleep_time>0:
                time.sleep(sleep_time)
        return rtn

    def get_pose_euler(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_euler(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def get_pose_matrix(self, pose=None):
        # if self.index == 3:
        #     print("HERE")
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].mDeviceToAbsoluteTracking
        else:
            return None

    def get_velocity(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vVelocity
        else:
            return None

    def get_angular_velocity(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vAngularVelocity
        else:
            return None

    def get_pose_quaternion(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None
    
    def get_pose_quaternion_robust(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion_robust(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None
    
    def get_pose_quaternion_tfs(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion_tfs(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def controller_state_to_dict(self, pControllerState):
        # This function is graciously borrowed from https://gist.github.com/awesomebytes/75daab3adb62b331f21ecf3a03b3ab46
        # docs: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetControllerState
        d = {}
        d['unPacketNum'] = pControllerState.unPacketNum
        # on trigger .y is always 0.0 says the docs
        d['trigger'] = pControllerState.rAxis[1].x
        # 0.0 on trigger is fully released
        # -1.0 to 1.0 on joystick and trackpads
        d['trackpad_x'] = pControllerState.rAxis[0].x
        d['trackpad_y'] = pControllerState.rAxis[0].y
        # These are published and always 0.0
        # for i in range(2, 5):
        #     d['unknowns_' + str(i) + '_x'] = pControllerState.rAxis[i].x
        #     d['unknowns_' + str(i) + '_y'] = pControllerState.rAxis[i].y
        d['ulButtonPressed'] = pControllerState.ulButtonPressed
        d['ulButtonTouched'] = pControllerState.ulButtonTouched
        # To make easier to understand what is going on
        # Second bit marks menu button
        d['menu_button'] = bool(pControllerState.ulButtonPressed >> 1 & 1)
        # 32 bit marks trackpad
        d['trackpad_pressed'] = bool(pControllerState.ulButtonPressed >> 32 & 1)
        d['trackpad_touched'] = bool(pControllerState.ulButtonTouched >> 32 & 1)
        # third bit marks grip button
        d['grip_button'] = bool(pControllerState.ulButtonPressed >> 2 & 1)
        # System button can't be read, if you press it
        # the controllers stop reporting
        return d

    def get_controller_inputs(self):
        result, state = self.vr.getControllerState(self.index)
        return self.controller_state_to_dict(state)

    def trigger_haptic_pulse(self, duration_micros=1000, axis_id=0):
        """
        Causes devices with haptic feedback to vibrate for a short time.
        """
        self.vr.triggerHapticPulse(self.index ,axis_id, duration_micros)

class vr_tracking_reference(vr_tracked_device):
    def get_mode(self):
        return _to_text(self.vr.getStringTrackedDeviceProperty(self.index,openvr.Prop_ModeLabel_String)).upper()
    def sample(self,num_samples,sample_rate):
        print("Warning: Tracking References do not move, sample isn't much use...")

class triad_openvr():
    def __init__(self, configfile_path=None):
        # Initialize OpenVR in the
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()

        # Initializing object to hold indexes for various tracked objects
        self.object_names = {"Tracking Reference":[],"HMD":[],"Controller":[],"Tracker":[]}
        self.devices = {}
        self.device_index_map = {}
        self.configfile_path = configfile_path
        self.config_mode = configfile_path is not None
        self.config = None
        self.config_devices_by_serial = {}
        self.device_serial_map = {}
        self.serial_device_name_map = {}
        self.refresh_interval = 0.5
        self._last_refresh_time = 0.0

        # Loading config file
        if configfile_path:
            try:
                with open(configfile_path, 'r') as json_data:
                    self.config = json.load(json_data)
            except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
                print('config.json not found.')
                exit(1)

            for device in self.config.get('devices', []):
                serial = _normalize_serial(device.get('serial'))
                device_name = device.get('name')
                device_type = device.get('type')
                if not serial or not device_name or device_type not in self.object_names:
                    print("Skipping invalid triad_openvr config device: {}".format(device))
                    continue
                if serial in self.config_devices_by_serial:
                    print("Duplicate triad_openvr serial in config: {}".format(serial))
                self.config_devices_by_serial[serial] = {
                    'name': device_name,
                    'type': device_type,
                    'serial': serial,
                }
                self.serial_device_name_map[serial] = device_name

        self.refresh_discovered_devices()

    def __del__(self):
        try:
            openvr.shutdown()
        except Exception:
            pass

    def get_pose(self):
        return get_pose(self.vr)

    def poll_vr_events(self):
        """
        Used to poll VR events and find any new tracked devices or ones that are no longer tracked.
        """
        changes = 0
        event = openvr.VREvent_t()
        while self.vrsystem.pollNextEvent(event):
            if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                if self.add_tracked_device(event.trackedDeviceIndex):
                    changes += 1
            elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                #If we were already tracking this device, quit tracking it.
                if self.remove_tracked_device(event.trackedDeviceIndex):
                    changes += 1
        if changes or time.monotonic() - self._last_refresh_time >= self.refresh_interval:
            changes += self.refresh_discovered_devices()
        return changes

    def refresh_discovered_devices(self, poses=None):
        """
        Rebuild active device indexes from the current OpenVR pose table.

        SteamVR occasionally drops activation/deactivation events when trackers
        sleep, lose radio, or reconnect with a new device index.  A refresh pass
        keeps the public devices/object_names maps consistent even if events were
        missed.
        """
        if poses is None:
            poses = self.get_pose()

        connected_indexes = set()
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bDeviceIsConnected:
                connected_indexes.add(i)

        changes = 0
        for tracked_device_index in list(self.device_index_map):
            if tracked_device_index not in connected_indexes:
                if self.remove_tracked_device(tracked_device_index):
                    changes += 1

        for tracked_device_index in sorted(connected_indexes):
            if self.add_tracked_device(tracked_device_index):
                changes += 1

        self._last_refresh_time = time.monotonic()
        return changes

    def add_tracked_device(self, tracked_device_index):
        registration = self._resolve_device_registration(tracked_device_index)
        if registration is None:
            return self.remove_tracked_device(tracked_device_index)

        device_name, device_type, serial = registration
        previous_name = self.device_index_map.get(tracked_device_index)
        if previous_name == device_name and device_name in self.devices:
            self.devices[device_name].index = tracked_device_index
            self.device_serial_map[tracked_device_index] = serial
            return False

        if previous_name and previous_name != device_name:
            self.remove_tracked_device(tracked_device_index)

        if device_name in self.devices:
            previous_index = self.devices[device_name].index
            if previous_index != tracked_device_index:
                self.remove_tracked_device(previous_index)

        if device_name not in self.object_names[device_type]:
            self.object_names[device_type].append(device_name)

        device_cls = vr_tracking_reference if device_type == "Tracking Reference" else vr_tracked_device
        self.devices[device_name] = device_cls(self.vr, tracked_device_index, device_type)
        self.device_index_map[tracked_device_index] = device_name
        self.device_serial_map[tracked_device_index] = serial
        if serial:
            self.serial_device_name_map[serial] = device_name

        if device_type == "Tracker":
            print("Added tracker: "+ device_name)
            print(" Index: {}".format(tracked_device_index))
        return True

    def remove_tracked_device(self, tracked_device_index):
        if tracked_device_index not in self.device_index_map:
            return False

        device_name = self.device_index_map.pop(tracked_device_index)
        self.device_serial_map.pop(tracked_device_index, None)
        device = self.devices.pop(device_name, None)
        if device is not None:
            names = self.object_names.get(device.device_class, [])
            if device_name in names:
                names.remove(device_name)
        return True

    def _resolve_device_registration(self, tracked_device_index):
        device_type = self._device_type_from_index(tracked_device_index)
        if device_type is None:
            return None

        serial = self._get_device_serial(tracked_device_index)
        if self.config_mode:
            configured = self.config_devices_by_serial.get(serial)
            if configured is not None:
                return configured['name'], configured['type'], serial
            # The sender may intentionally accept any two GenericTrackers.
            # Keep configured serials stable, but do not hide an otherwise
            # valid replacement tracker merely because its serial is new.
            if device_type != "Tracker":
                return None
            device_name = self.serial_device_name_map.get(serial) if serial else None
            if not device_name:
                device_name = self._next_device_name(device_type)
            return device_name, device_type, serial

        device_name = self.serial_device_name_map.get(serial) if serial else None
        if not device_name:
            previous_name = self.device_index_map.get(tracked_device_index)
            previous_device = self.devices.get(previous_name)
            if previous_device is not None and previous_device.device_class == device_type:
                device_name = previous_name
        if not device_name:
            device_name = self._next_device_name(device_type)
        return device_name, device_type, serial

    def _device_type_from_index(self, tracked_device_index):
        device_class = self.vr.getTrackedDeviceClass(tracked_device_index)
        if device_class == openvr.TrackedDeviceClass_Controller:
            return "Controller"
        if device_class == openvr.TrackedDeviceClass_HMD:
            return "HMD"
        if device_class == openvr.TrackedDeviceClass_GenericTracker:
            return "Tracker"
        if device_class == openvr.TrackedDeviceClass_TrackingReference:
            return "Tracking Reference"
        return None

    def _get_device_serial(self, tracked_device_index):
        try:
            serial = self.vr.getStringTrackedDeviceProperty(
                tracked_device_index,
                openvr.Prop_SerialNumber_String,
            )
        except Exception:
            serial = ""
        return _normalize_serial(serial)

    def _next_device_name(self, device_type):
        prefixes = {
            "Controller": "controller",
            "HMD": "hmd",
            "Tracker": "tracker",
            "Tracking Reference": "tracking_reference",
        }
        prefix = prefixes[device_type]
        used_names = set(self.devices)
        used_names.update(self.object_names[device_type])
        used_names.update(self.serial_device_name_map.values())
        i = 1
        while "{}_{}".format(prefix, i) in used_names:
            i += 1
        return "{}_{}".format(prefix, i)

    def rename_device(self,old_device_name,new_device_name):
        self.devices[new_device_name] = self.devices.pop(old_device_name)
        for i in range(len(self.object_names[self.devices[new_device_name].device_class])):
            if self.object_names[self.devices[new_device_name].device_class][i] == old_device_name:
                self.object_names[self.devices[new_device_name].device_class][i] = new_device_name
        for tracked_device_index, device_name in list(self.device_index_map.items()):
            if device_name == old_device_name:
                self.device_index_map[tracked_device_index] = new_device_name
        for serial, device_name in list(self.serial_device_name_map.items()):
            if device_name == old_device_name:
                self.serial_device_name_map[serial] = new_device_name

    def print_discovered_objects(self):
        for device_type in self.object_names:
            plural = device_type
            if len(self.object_names[device_type])!=1:
                plural+="s"
            print("Found "+str(len(self.object_names[device_type]))+" "+plural)
            for device in self.object_names[device_type]:
                if device_type == "Tracking Reference":
                    print("  "+device+" ("+_to_text(self.devices[device].get_serial())+
                          ", Mode "+_to_text(self.devices[device].get_mode())+
                          ", "+_to_text(self.devices[device].get_model())+
                          ")")
                else:
                    print("  "+device+" ("+_to_text(self.devices[device].get_serial())+
                          ", "+_to_text(self.devices[device].get_model())+")")
