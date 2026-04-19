from pymodbus.client.sync import ModbusTcpClient #pip3 install pymodbus==2.5.3
import time
import cv2
import numpy as np


def open_modbus(ip, port):
    client = ModbusTcpClient(ip, port)
    client.connect()
    return client

def write_register(client, address, values):
    # Modbus 写入寄存器，传入寄存器地址和要写入的值列表
    client.write_registers(address, values)

def read_register(client, address, count):
    # Modbus 读取寄存器
    response = client.read_holding_registers(address, count)
    return response.registers if response.isError() is False else []

def write6(client, reg, val):
    val_reg = []
    for i in range(6):
        val_reg.append(val[i] & 0xFFFF)  # 取低16位
    write_register(client, reg, val_reg)

def read6(client, reg, n_short=6, use_byte=False): # 1 short = 2 bytes
    val = read_register(client, reg, n_short)
    if len(val) < n_short:
        raise Exception('没有读到数据')

    if use_byte:
        results = []
        # 将每个寄存器的高位和低位分开存储
        for i in range(len(val)):
            # 读取当前寄存器和下一个寄存器
            low_byte = val[i] & 0xFF            # 低八位
            high_byte = (val[i] >> 8) & 0xFF     # 高八位
            results.append(low_byte)  # 存储低八位
            results.append(high_byte)  # 存储高八位
    else:
        results = val
    return results


class InspireTacModbus:
    def __init__(self, ip = '192.168.12.210', port = 6000, max_fingertip_force=800):
        self.client = open_modbus(ip, port)

        # 寄存器地址
        self.regdict = {
            'ID': 1000,
            'baudrate': 1001,
            'clearErr': 1004,
            'forceClb': 1009,
            'defaultSpeedSet': 1032,
            'defaultForceSet': 1044,
            'angleSet': 1486,
            'forceSet': 1498,
            'speedSet': 1522,
            'angleAct': 1546,
            'forceAct': 1582,
            'errCode': 1606,
            'statusCode': 1612,
            'temp': 1618,
            'actionSeq': 2320,
            'actionRun': 2322,
        }

        # tac data
        self.tac_chunks_start = [(3000+i*114) for i in range(18)] + [5052]
        self.tac_chunks_count = [57]*18 + [36]
        #for s, c in zip(chunks_start, chunks_count):
        #   print(s, s+c*2)
        self.n_tac_regions = 17
        self.tac_region_names = ["little_1", "little_2", "little_3",
                                 "ring_1", "ring_2", "ring_3",
                                 "middle_1", "middle_2", "middle_3",
                                 "index_1", "index_2", "index_3",
                                 "thumb_1", "thumb_2", "thumb_3", "thumb_4",
                                 "palm"]
        self.tac_region_sizes = [(3,3), (12,8), (10,8),
                                 (3,3), (12,8), (10,8),
                                 (3,3), (12,8), (10,8),
                                 (3,3), (12,8), (10,8),
                                 (3,3), (12,8), (3,3), (12,8),
                                 (8,14)]
        self.tac_region_start_idxs = [0] + [sum([s[0]*s[1] for s in self.tac_region_sizes[:i]]) for i in range(1, len(self.tac_region_sizes)+1)]

        # Precompute region to chunk mapping
        self.chunk_descriptors = []
        current_idx = 0
        for i, (start_addr, count) in enumerate(zip(self.tac_chunks_start, self.tac_chunks_count)):
            self.chunk_descriptors.append({
                'chunk_idx': i,
                'reg_start': start_addr,
                'count': count,
                'buf_start': current_idx,
                'buf_end': current_idx + count
            })
            current_idx += count
        
        self.region_to_chunks = {}
        for i, name in enumerate(self.tac_region_names):
            r_start = self.tac_region_start_idxs[i]
            r_end = self.tac_region_start_idxs[i+1]
            
            # Find chunks that overlap with [r_start, r_end)
            overlapping_chunks = []
            for chunk in self.chunk_descriptors:
                # Check overlap: start1 < end2 and start2 < end1
                if r_start < chunk['buf_end'] and chunk['buf_start'] < r_end:
                    overlapping_chunks.append(chunk['chunk_idx'])
            self.region_to_chunks[name] = overlapping_chunks

        # initialize
        self.get_default_speed_force()
        self.set_force([max_fingertip_force]*6)

    def get_ID(self):
        ret = read_register(self.client, self.regdict['ID'], 1)[0]
        print('ID:', ret)

    def get_default_speed_force(self):
        speed = read6(self.client, self.regdict['defaultSpeedSet'], n_short=6, use_byte=False)
        force = read6(self.client, self.regdict['defaultForceSet'], n_short=6, use_byte=False)
        print('Default speed set:', speed)
        print('Default force set:', force)
    
    def clear_error(self):
        raise NotImplementedError('TODO')
        write_register(self.client, self.regdict['clearErr'], [1])
        time.sleep(1)
    
    def calib_joint_force(self):
        write_register(self.client, self.regdict['forceClb'], [1])
        time.sleep(1)

    def read_tac_data(self, target_regions=None):
        '''
        target_regions: list of region names to read. If None, read all.
        return:
            tac_vec: (1062,) tactile data, range [0,1]. Unread regions are 0.
            tac_dict: dict of tactile data, each region is a small matrix
        '''
        # Determine which chunks to read
        if target_regions is None:
            chunks_to_read_indices = range(len(self.chunk_descriptors))
        else:
            chunks_to_read_indices = set()
            for name in target_regions:
                if name in self.region_to_chunks:
                    chunks_to_read_indices.update(self.region_to_chunks[name])
                else: 
                     print(f"Warning: Region {name} not found")
            chunks_to_read_indices = sorted(list(chunks_to_read_indices))

        # buffer for all data (init with zeros)
        full_results = np.zeros(1062, dtype=float)
        
        # Read chunks
        for i in chunks_to_read_indices:
            chunk = self.chunk_descriptors[i]
            val = read_register(self.client, chunk['reg_start'], chunk['count'])
            if len(val) == chunk['count']:
                 full_results[chunk['buf_start']:chunk['buf_end']] = val
            # else:
            #      print(f"Error reading chunk {i}")

        tac_vec = full_results / 4096.

        tac_dict = {}
        # Only populate tac_dict for regions that were requested (or all if None)
        regions_to_process = target_regions if target_regions is not None else self.tac_region_names
        
        for name in regions_to_process:
            if name not in self.tac_region_names: continue
            
            i = self.tac_region_names.index(name)
            region_size = self.tac_region_sizes[i]
            r_start = self.tac_region_start_idxs[i]
            r_end = self.tac_region_start_idxs[i+1]
            
            region_data = tac_vec[r_start:r_end].reshape(region_size)
            if name.endswith("_1") or name=="thumb_3" or name == 'palm':
                # 3*3 sensors and palm sensors are rotated 90 deg
                region_data = region_data.reshape(region_size[1], region_size[0]).T
            elif name=="thumb_4":
                # thumb_4 sensors are up-side-down
                region_data = region_data[::-1, ::-1]
            tac_dict[name] = region_data
        #print(tac_dict)
        return tac_vec, tac_dict

    def read_joint_pos(self):
        '''
        return: 
            joint_pos: (6,) joint positions, scaled in range [0,1]
        '''
        data = read6(self.client, self.regdict['angleAct'], n_short=6, use_byte=False)
        joint_pos = np.array(data) / 1000.
        return joint_pos

    def read_joint_force(self):
        '''
        return: 
            joint_force: (6,) joint forces, scaled in range [-1,1]
        '''
        raise NotImplementedError('TODO: check the data range')
        data = read6(self.client, self.regdict['forceAct'], n_short=6, use_byte=False)
        joint_force = np.array(data) / 4000.
        return joint_force

    def set_joint_pos(self, joint_pos):
        '''
        joint_pos: (6,) joint positions, scaled in range [0,1]
        '''
        data = (np.asarray(joint_pos) * 1000).astype(int)
        write6(self.client, self.regdict['angleSet'], data)

    def set_force(self, force):
        '''
        Set max force for the six joints
        force: (6,), range [0,3000] (g)
        '''
        data = np.asarray(force).astype(int)
        write6(self.client, self.regdict['forceSet'], data)
        
    def close(self):
        self.client.close()



class InspireSimulatedForceControl:
    def __init__(self, ip='192.168.11.210', max_force_rating=3000, kp=2000, min_force_threshold=300):
        """
        初始化灵巧手拟力控接口
        :param ip: 灵巧手IP
        :param max_force_rating: 灵巧手最大力矩设定值 (通常是 g 或者毫牛米等内部单位，参考底层驱动)
        :param kp: 阻抗控制的刚度系数 (Impedance Control Stiffness), 防止误差越大导致的力越大
        """
        self.driver = InspireTacModbus(ip=ip, max_fingertip_force=max_force_rating)
        self.max_force_rating = max_force_rating
        self.min_force_threshold = min_force_threshold
        self.kp = kp
        self.current_pos = np.ones(6)  # 默认初始化为全开状态, 后续通过get_state更新
        self.last_update_time = time.time()
        self.last_force_cmds = [-1] * 6
        self.last_target_pos = [-999] * 6

    def get_state(self):
        """
        获取当前关节位置
        :return: 6维数组, 0-1之间
        """
        try:
            self.current_pos = self.driver.read_joint_pos()
        except Exception as e:
            print(f"Read Error: {e}")
        return self.current_pos

    def step(self, target_pos, target_close_force_ratio = [0., 0., 0., 0., 0., 0.]):
        """
        控制步进函数
        :param target_pos: 目标位置, 6维数组 or list, 范围 [0, 1] (0: tight/close, 1: open)
        :param target_close_force_ratio: 目标闭合力比例, 6维数组 or list, 范围 [0, 1]. 
               1.0 表示使用硬件允许的最大力.
        :return: 6维数组, target_pos - current_pos
        """
        # 1. 获取最新状态(如果不希望每一步都读，可以在外部调用get_state)
        # current_pos = self.get_state() 
        current_pos = self.current_pos # 使用缓存的状态(假设外部循环中会更新或者在step前更新)

        # 2. 计算控制指令
        # 硬件协议通常是: set_joint_pos 设置目标位置, set_force 设置力矩限制
        
        force_cmds = []
          # 硬件所需的最小有效力值, 防止设为0不动
        
        for i in range(6):
            t_pos = target_pos[i]
            c_pos = current_pos[i]
            
            # 计算期望的力矩值 (映射从 0-1 到 0-max_force_rating)
            desired_force_val = target_close_force_ratio[i] * self.max_force_rating
            
            # 逻辑判定: 是 "张开" 还是 "闭合/保持" ?
            # 0 is closed (tight), 1 is open.
            # 如果 target > current, 说明要向张开方向运动 (from 0.1 to 0.5)
            # 如果 target < current, 说明要向闭合方向运动 (from 0.5 to 0.1)
            
            # 使用一个小的死区 epsilon 防止震荡
            epsilon = 0.01
            
            if t_pos > c_pos + epsilon:
                # Case 1: 张开阶段 (Opening)
                cmd_force = self.min_force_threshold
            else:
                # Case 2: 闭合或保持阶段 (Closing / Grasping)
                
                # Impedance Control (阻抗控制)
                # 偏差越大 (target < current, deviation = current - target), 力越大
                # 当 action (target) 比 state (current) 大时(即Opening), 上面if分支已经处理，保持默认(Max)
                # 这里处理 Closing / Holding.
                # deviation > 0 means we are blocked or closing gap.
                deviation = max(0.0, c_pos - t_pos)
                impedance_force = deviation * self.kp
                
                total_force = desired_force_val + impedance_force
                
                cmd_force = min(max(total_force, self.min_force_threshold), self.max_force_rating)
            
            force_cmds.append(cmd_force)

        # 3. 下发指令
        # 先下发力配置，再下发位置，或者反之皆可， Modbus通常是独立的寄存器写入
        # 注意: 频繁写寄存器可能会占用通信带宽，如果 force_cmds 变化不大可以做个滤波或判重
        
        if np.max(np.abs(np.array(force_cmds) - np.array(self.last_force_cmds))) > 100:
            self.driver.set_force(force_cmds)
            self.last_force_cmds = list(force_cmds)

        if np.max(np.abs(np.array(target_pos) - np.array(self.last_target_pos))) > 0.01:
            self.driver.set_joint_pos(target_pos)
            self.last_target_pos = list(target_pos)
            
        return np.array(target_pos) - np.array(current_pos)

    def close(self):
        self.driver.close()



class TactileVisualizer:
    def __init__(self, canvas_size=(600, 800, 3), cell_size=10):
        """
        初始化触觉数据可视化类
        :param canvas_size: 画布大小 (height, width, channels)
        :param cell_size: 每个矩阵单元格的像素大小
        """
        self.canvas_size = canvas_size
        self.cell_size = cell_size
        self.canvas = np.zeros(self.canvas_size, dtype=np.uint8)
        
        # 定义每个触觉区域的位置
        self.region_positions = {
            "little_1": (50, 50), "little_2": (50, 150), "little_3": (50, 300),
            "ring_1": (200, 50), "ring_2": (200, 150), "ring_3": (200, 300),
            "middle_1": (350, 50), "middle_2": (350, 150), "middle_3": (350, 300),
            "index_1": (500, 50), "index_2": (500, 150), "index_3": (500, 300),
            "thumb_1": (650, 50), "thumb_2": (650, 150), "thumb_3": (650, 300), "thumb_4": (650, 450),
            "palm": (200, 450)
        }
        
    def update(self, tac_dict, minus_dict = None):
        """
        更新触觉数据并可视化
        :param tac_dict: 包含触觉数据的字典
        """
        # 清空画布
        self.canvas.fill(0)
        
        for region_name, region_data in tac_dict.items():
            if region_name not in self.region_positions:
                continue

            # 获取矩阵的左上角起始位置
            start_x, start_y = self.region_positions[region_name]
            h, w = region_data.shape
            if minus_dict is None:
                minus_data = np.zeros_like(region_data)
            else:
                minus_data = minus_dict.get(region_name, np.zeros_like(region_data))
            print_region_data = region_data - minus_data

            # 遍历矩阵中的每个值，并在画布上绘制对应的矩形
            for i in range(h):
                for j in range(w):
                
                    value = print_region_data[i, j]

                    # # 将值归一化到0-1范围内，并映射到颜色
                    color_value = int(value*1000*255)
                    color = (color_value , color_value / 255, color_value/ 255 / 255)  # 蓝色通道代表触觉值（深浅）
                    top_left = (start_x + j * self.cell_size, start_y + i * self.cell_size)
                    bottom_right = (start_x + (j + 1) * self.cell_size, start_y + (i + 1) * self.cell_size)
                    cv2.rectangle(self.canvas, top_left, bottom_right, color, -1)

            # 标注区域名称
            cv2.putText(self.canvas, region_name, (start_x, start_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 显示画布
        cv2.imshow('Tactile Data Visualization', self.canvas)
        cv2.waitKey(1)


# class TactileVisualizer:
#     def __init__(self, canvas_size=(1400, 1800, 3), cell_size=40):
#         """
#         初始化触觉数据可视化类
#         :param canvas_size: 画布大小 (height, width, channels)。已增大以容纳更大的单元格。
#         :param cell_size: 每个矩阵单元格的像素大小。已增大以便显示数值。
#         """
#         self.canvas_size = canvas_size
#         self.cell_size = cell_size
#         self.canvas = np.zeros(self.canvas_size, dtype=np.uint8)
        
#         # 重新定义每个触觉区域的位置 (y, x)，以适应更大的 cell_size 并防止重叠
#         self.region_positions = {
#             "little_1": (50, 50), "little_2": (50, 220), "little_3": (50, 590),
#             "ring_1":   (250, 50), "ring_2":   (250, 220), "ring_3":   (250, 590),
#             "middle_1": (450, 50), "middle_2": (450, 220), "middle_3": (450, 590),
#             "index_1":  (650, 50), "index_2":  (650, 220), "index_3":  (650, 590),
#             "thumb_1":  (850, 50), "thumb_2":  (850, 220), "thumb_3":  (850, 590), "thumb_4": (850, 760),
#             "palm":     (50, 1150)
#         }
        
#     def update(self, tac_dict, minus_dict = None):
#         """
#         更新触觉数据并可视化
#         :param tac_dict: 包含触觉数据的字典
#         :param minus_dict: 用于相减的基线数据字典 (可选)
#         """
#         # 清空画布
#         self.canvas.fill(0)
        
#         for region_name, region_data in tac_dict.items():
#             if region_name not in self.region_positions:
#                 continue

#             # 获取矩阵的左上角起始位置
#             start_y, start_x = self.region_positions[region_name]
#             h, w = region_data.shape
            
#             if minus_dict is None:
#                 minus_data = np.zeros_like(region_data)
#             else:
#                 minus_data = minus_dict.get(region_name, np.zeros_like(region_data))
                
#             print_region_data = region_data - minus_data

#             # --- 主要修改部分 ---
#             # 遍历矩阵中的每个值
#             for i in range(h):
#                 for j in range(w):
#                     value = print_region_data[i, j]

#                     # 1. 绘制背景矩形 (颜色依然代表数值大小)
#                     # 将原始浮点值映射到0-255的灰度值，以便可视化
#                     # 我们使用 np.clip 确保值在 0-255 之间
#                     # 乘以一个较大的系数 (如 1000) 是因为原始值通常很小
#                     gray_value = int(np.clip(value * 1000, 0, 255))
#                     color = (gray_value, gray_value, gray_value)  # (B, G, R) 灰度
                    
#                     top_left = (start_x + j * self.cell_size, start_y + i * self.cell_size)
#                     bottom_right = (start_x + (j + 1) * self.cell_size, start_y + (i + 1) * self.cell_size)
#                     cv2.rectangle(self.canvas, top_left, bottom_right, color, -1)

#                     # 2. 准备并绘制数值文本
#                     # 根据背景亮度动态选择文本颜色，确保可读性
#                     if gray_value > 128:
#                         text_color = (0, 0, 0)  # 亮背景 -> 黑字
#                     else:
#                         text_color = (255, 255, 255) # 暗背景 -> 白字

#                     # 将浮点数格式化为保留两位小数的字符串
#                     # text_to_draw = f"{value:.2f}"
#                     text_to_draw = f"{int(value):d}"
#                     font_scale = 0.4
#                     font_thickness = 1

#                     # 3. 计算文本位置，使其在单元格内居中
#                     (text_width, text_height), baseline = cv2.getTextSize(text_to_draw, 
#                                                                         cv2.FONT_HERSHEY_SIMPLEX, 
#                                                                         font_scale, font_thickness)
                    
#                     text_x = top_left[0] + (self.cell_size - text_width) // 2
#                     text_y = top_left[1] + (self.cell_size + text_height) // 2
                    
#                     # 4. 在画布上绘制文本
#                     cv2.putText(self.canvas,
#                                 text_to_draw,
#                                 (text_x, text_y),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 font_scale,
#                                 text_color,
#                                 font_thickness,
#                                 cv2.LINE_AA)
#             # --- 修改结束 ---

#             # 标注区域名称
#             cv2.putText(self.canvas, region_name, (start_x, start_y - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

#         # 显示画布
#         cv2.imshow('Tactile Data Visualization', self.canvas)
#         cv2.waitKey(1)
        
if __name__ == '__main__':
    from tqdm import trange
    import numpy as np
    test = InspireTacModbus(max_fingertip_force=800)
    # test.set_joint_pos([1, 1, 1, 1, 1, 1])
    
    # test.set_joint_pos(np.ones(6))
    # time.sleep(5.0)
    # time.sleep(1)

    test.set_joint_pos([0, 0, 0, 0, 1, 1])
    joint_pos = test.read_joint_pos()
    tac_vec, tac_dict = test.read_tac_data()
    test.set_joint_pos([1, 1, 1, 1, 1, 1])
    # test.set_joint_pos([0.5, 0.5, 0.5, 0.5, 1, 1])
    # test.set_joint_pos([0, 0, 0, 0, 1, 1])

    # #test.get_default_speed_force()
    # visualizer = TactileVisualizer()
    # last_time = time.time()
    # while True:
    print('Joint pos:', np.round(joint_pos, 3))
    print('Tactile data:', tac_vec)
    #     # tac_vec, tac_dict = test.read_tac_data()
    #     # test.set_force([300]*6)
    #     test.set_joint_pos([0, 0, 0, 0, 0.5, 0.5])
    #     # test.set_joint_pos([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    #     # visualizer.update(tac_dict)
    #     #test.set_joint_pos([np.cos(t/100)/2+0.5]*6)
    #     #print(joint_pos)


    #     interval = time.time() - last_time
    #     last_time = time.time()
    #     print(f'time: {interval*1000:.1f} ms')

    test.close()


