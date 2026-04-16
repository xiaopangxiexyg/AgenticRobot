import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# 动态添加路径以确保可以导入同目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from .vive_data_receiver import VIVEDataReceiver
except ImportError:
    print(f"Error: 无法在 {current_dir} 中找到 `vive_data_receiver.py` 或无法导入 `VIVEDataReceiver` 类。")
    print("请确保该文件存在且没有语法错误。")
    sys.exit(1)

class ViveVisualizer:
    def __init__(self, port=9999, axis_length=0.1):
        self.port = port
        self.axis_length = axis_length
        self.receiver = VIVEDataReceiver(port=self.port, num_floats=36)
        
        # 初始化 Matplotlib 图形
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(f"Vive Trackers Visualization (Port: {port})")
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 初始化坐标轴范围（根据实际空间调整，Vive一般在 -2m 到 2m 之间）
        self.limit = 1.0
        self.ax.set_xlim(-self.limit, self.limit)
        self.ax.set_ylim(-self.limit, self.limit)
        self.ax.set_zlim(0, 2.0) # Z轴通常向上，高度0-2米
        
        # 存储绘图对象以便更新
        self.quivers = [] 
        self.texts = []
        
        # 世界坐标系原点
        # self.draw_coordinate_frame(np.eye(4), "World", length=0.2, linestyle='--')
        
        self.is_running = True

    def start(self):
        print("启动接收器...")
        self.receiver.start()
        
        print("开始可视化... 按 Ctrl+C 退出。")
        # 使用 FuncAnimation 进行动态更新
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        print("\n正在停止...")
        self.receiver.stop()
        sys.exit(0)

    def draw_frame(self, position, rotation_matrix, label, color_prefix):
        """
        绘制一个坐标系（RGB对应XYZ）
        """
        origin = position
        
        # 定义局部坐标系的基向量 (X, Y, Z)
        # 旋转矩阵的列向量即为局部坐标轴在世界坐标系下的方向
        axis_vectors = [
            rotation_matrix[:, 0], # X轴
            rotation_matrix[:, 1], # Y轴
            rotation_matrix[:, 2]  # Z轴
        ]
        
        colors = ['r', 'g', 'b'] # X=红, Y=绿, Z=蓝
        
        lines = []
        for i, vector in enumerate(axis_vectors):
            # 计算轴的终点
            endpoint = origin + vector * self.axis_length
            # 绘制线段
            line, = self.ax.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=colors[i],
                linewidth=2
            )
            lines.append(line)
            
        # 绘制原点文字
        text = self.ax.text(origin[0], origin[1], origin[2], label, color='k')
        
        return lines, text

    def update_plot(self, frame):
        # 获取最新数据
        data = self.receiver.get_latest_data()
        
        if data is None:
            return
        
        # data 是 36个浮点数 (3个tracker * 12 floats)
        # 清除上一帧的动态元素 (保留静态的世界坐标系等)
        # 注意：matplotlib 3D清除重绘开销较大，这里尝试清除特定对象或者全部清除重绘
        # 简单起见，清除当前坐标轴内容并重设范围（虽然效率稍低，但代码最稳健）
        self.ax.cla()
        
        # 重设标签和范围，因为 cla() 会清除它们
        self.ax.set_title(f"Vive Trackers Visualization (Port: {self.port})")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-self.limit, self.limit)
        self.ax.set_ylim(-self.limit, self.limit)
        self.ax.set_zlim(0, 2.0)
        
        # 绘制世界坐标系
        self.ax.plot([0, 0.2], [0, 0], [0, 0], 'r--') # X
        self.ax.plot([0, 0], [0, 0.2], [0, 0], 'g--') # Y
        self.ax.plot([0, 0], [0, 0], [0, 0.2], 'b--') # Z
        self.ax.text(0, 0, 0, "World", color='k')

        # 解析并绘制每个 Tracker
        tracker_names = ["Tracker_1", "Tracker_2", "Tracker_3"]
        
        for i in range(3):
            base_idx = i * 12
            # 提取数据
            # 假设格式: [Px, Py, Pz, R00, R01, R02, R10, R11, R12, R20, R21, R22]
            # 这是基于发送端 print(data[0:3]) 为 position 的逻辑推断
            
            chunk = data[base_idx : base_idx+12]
            
            pos = np.array(chunk[0:3])
            rot_flat = np.array(chunk[3:12])
            
            # 检查数据有效性（防止全0）
            if np.all(pos == 0) and np.all(rot_flat == 0):
                continue
                
            try:
                rot = rot_flat.reshape(3, 3)
            except ValueError:
                continue

            # 绘制
            self.draw_frame(pos, rot, tracker_names[i], i)

if __name__ == "__main__":
    # 配置
    port = 9999
    
    # 简单的命令行参数支持
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Usage: python vive_visualizer.py [port]")
            sys.exit(1)
            
    viz = ViveVisualizer(port=port)
    viz.start()
