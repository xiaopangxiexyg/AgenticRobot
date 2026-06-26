import os
import sys
import numpy as np
import cv2
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from teleop.mocap.vive_data_receiver import VIVEDataReceiver
except ImportError:
    # 如果上面的导入失败，尝试使用原始文件中的路径添加方式
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from teleop.mocap.vive_data_receiver import VIVEDataReceiver

def draw_axes_manual(img, camera_matrix, dist_coeffs, R, t, length):
    """
    手动绘制坐标轴，避免使用 cv2.Rodrigues 带来的 180 度奇异性问题。
    直接使用旋转矩阵变换坐标轴端点。
    """
    # 定义局部坐标系下的轴端点: 原点, X, Y, Z
    # shape: (4, 3)
    points_local = np.array([
        [0, 0, 0],      # Origin
        [length, 0, 0], # X
        [0, length, 0], # Y
        [0, 0, length]  # Z
    ], dtype=np.float32)
    
    # 变换到相机坐标系: P_cam = R @ P_local + t
    # R: (3,3), points_local.T: (3,4) -> (3,4)
    points_cam = (R @ points_local.T).T + t.reshape(1, 3)
    
    # 投影到 2D 像素坐标
    # 因为点已经在相机坐标系下，所以 projectPoints 的 rvec, tvec 设为 0
    points_2d, _ = cv2.projectPoints(points_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    points_2d = np.int32(points_2d).reshape(-1, 2)
    
    origin = tuple(points_2d[0])
    pt_x = tuple(points_2d[1])
    pt_y = tuple(points_2d[2])
    pt_z = tuple(points_2d[3])
    
    # 绘制线段 (BGR 颜色空间)
    # X轴: 红色 (0, 0, 255)
    # Y轴: 绿色 (0, 255, 0)
    # Z轴: 蓝色 (255, 0, 0)
    cv2.line(img, origin, pt_x, (0, 0, 255), 2)
    cv2.line(img, origin, pt_y, (0, 255, 0), 2)
    cv2.line(img, origin, pt_z, (255, 0, 0), 2)

def visualize_tracker_only():
    """
    单纯根据 Tracker 的 12 维信息进行可视化渲染。
    不依赖相机流，使用虚拟相机和黑色背景。
    """
    # 1. 初始化 VIVE 接收器
    print("正在连接 VIVE 接收器 (Port: 9999)...")
    receiver = VIVEDataReceiver(port=9999, num_floats=36)
    receiver.start()
    
    # 2. 定义虚拟相机参数 (Intrinsics)
    width, height = 1600, 1200
    focal_length = 200 # 焦距，决定视野大小
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1)) # 虚拟相机无畸变

    print("开始可视化。按 'q' 退出。")
    print("说明: 坐标轴显示 Tracker 的姿态 (固定在屏幕中心前方)。")
    print("      红色: X轴, 绿色: Y轴, 蓝色: Z轴")

    try:
        while True:
            # 3. 获取数据
            vive_data = receiver.get_latest_data()
            
            # 如果没有数据，显示等待
            if vive_data is None:
                img = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(img, "Connecting to VIVE...", (50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Tracker Visualization", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.01)
                continue
                
            # 提取 Tracker 数据 (12维: 3 pos + 9 rot)
            # 根据 run_calibrate_handeye.py: tracker_pose_data = vive_data[12:24]
            tracker_data = vive_data[12:24] # 默认是 tracker1
            
            # 检查数据有效性 (全0通常表示未追踪)
            if all(v == 0 for v in tracker_data):
                img = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(img, "Tracker Not Visible / Data Invalid", (50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Tracker Visualization", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 解析位姿
            # 位置 (原始数据通常是米)
            pos = np.array(tracker_data[0:3]) 
            # 旋转矩阵 (3x3)
            rot_matrix = np.array(tracker_data[3:12]).reshape(3, 3)

            # 4. 渲染
            # 创建黑色背景
            img = np.zeros((height, width, 3), dtype=np.uint8)

            # --- 定义虚拟相机视角 (World -> Camera) ---
            # 我们固定相机，观察运动的 Tracker
            # 假设相机位于世界坐标系原点后方，看向 Z 正方向
            # 并且为了适应屏幕坐标系 (Y向下)，我们将 Y 轴翻转
            # 这样世界坐标系中的 Y 向上 在屏幕上就是 向上 (即屏幕坐标减小)
            
            # 视图平移 (将世界原点推到相机前方 1.5m 处，并稍微向下平移一点以便居中)
            view_offset = np.array([0.0, 0.3, 1.5]) 
            
            # 视图旋转 (翻转 Y 轴，使世界系的 +Y 对应屏幕的上方)
            view_rot = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=np.float32)

            # --- 1. 绘制原点 (固定不动) ---
            # 原点在世界系中位姿: R=I, t=0
            t_origin_world = np.zeros((3, 1))
            R_origin_world = np.eye(3)
            
            # 变换到相机系: P_cam = R_view * P_world + t_view
            t_origin_cam = view_rot @ t_origin_world + view_offset.reshape(3, 1)
            R_origin_cam = view_rot @ R_origin_world
            
            # 绘制原点坐标轴 (长度 0.3m)
            # rvec_origin, _ = cv2.Rodrigues(R_origin_cam)
            # cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec_origin, t_origin_cam, 0.3)
            draw_axes_manual(img, camera_matrix, dist_coeffs, R_origin_cam, t_origin_cam, 0.3)
            
            # 标记原点文字
            origin_pt_2d, _ = cv2.projectPoints(t_origin_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
            origin_pt_2d = tuple(np.int32(origin_pt_2d).reshape(-1, 2)[0])
            cv2.putText(img, "World Origin", (origin_pt_2d[0] + 10, origin_pt_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- 2. 绘制 Tracker (运动) ---
            t_tracker_world = pos.reshape(3, 1)
            R_tracker_world = rot_matrix
            
            # 变换到相机系
            t_tracker_cam = view_rot @ t_tracker_world + view_offset.reshape(3, 1)
            R_tracker_cam = view_rot @ R_tracker_world
            
            # 绘制 Tracker 坐标轴
            # rvec_tracker, _ = cv2.Rodrigues(R_tracker_cam)
            # cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec_tracker, t_tracker_cam, 0.15)
            draw_axes_manual(img, camera_matrix, dist_coeffs, R_tracker_cam, t_tracker_cam, 0.15)
            
            # --- 3. 绘制连线 (Origin -> Tracker) ---
            tracker_pt_2d, _ = cv2.projectPoints(t_tracker_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
            tracker_pt_2d = tuple(np.int32(tracker_pt_2d).reshape(-1, 2)[0])
            
            cv2.line(img, origin_pt_2d, tracker_pt_2d, (100, 100, 100), 1)
            cv2.putText(img, "Tracker", (tracker_pt_2d[0] + 10, tracker_pt_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- 显示数值信息 ---
            # 绘制背景框以提高文字可读性
            cv2.rectangle(img, (0, 0), (300, 250), (30, 30, 30), -1)

            # 位置信息
            text_y = 30
            cv2.putText(img, f"Tracker Position (Station Frame):", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            text_y += 25
            cv2.putText(img, f"X: {pos[0]:.4f} m", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1) # Red-ish
            text_y += 25
            cv2.putText(img, f"Y: {pos[1]:.4f} m", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1) # Green-ish
            text_y += 25
            cv2.putText(img, f"Z: {pos[2]:.4f} m", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1) # Blue-ish

            # 旋转矩阵信息
            text_y += 35
            cv2.putText(img, f"Rotation Matrix:", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            for i in range(3):
                text_y += 25
                row_str = f"[{rot_matrix[i][0]:.2f}, {rot_matrix[i][1]:.2f}, {rot_matrix[i][2]:.2f}]"
                cv2.putText(img, row_str, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # 底部提示
            cv2.putText(img, "Visualizing Orientation (Fixed Depth)", (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.imshow("Tracker Visualization", img)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        receiver.stop()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    visualize_tracker_only()
