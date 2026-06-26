import socket
import struct
import threading
import time
import numpy as np

class VIVEDataReceiver:
    """
    一个线程安全的UDP数据接收器，用于高频数据流。
    它会启动一个后台线程来清空网络缓冲区并只保留最新的数据包。
    """
    def __init__(self, host="0.0.0.0", port=9999, num_floats=36):
        self.host = host
        self.port = port
        self.num_floats = num_floats
        
        # 定义与发送端完全一致的数据格式
        self.format_string = f'<{self.num_floats}f'
        self.buffer_size = struct.calcsize(self.format_string)
        
        # 初始化socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        
        # 共享变量，用于存储最新数据
        self.latest_data = None
        # 线程锁，用于保护对共享变量的访问
        self.lock = threading.Lock()
        
        # 标志位，用于控制后台线程的运行
        self.is_running = False
        
        # 后台接收线程
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)

    def _receive_loop(self):
        """后台线程运行的循环，专门用于接收数据。"""
        while self.is_running:
            try:
                data, _ = self.sock.recvfrom(self.buffer_size)
                
                # 确保收到的数据长度正确
                if len(data) == self.buffer_size:
                    unpacked_data = struct.unpack(self.format_string, data)
                    
                    # --- 核心逻辑 ---
                    # 获取锁，然后更新最新数据
                    with self.lock:
                        self.latest_data = unpacked_data
                        
            except socket.error as e:
                # 在停止时，关闭socket会引发异常，可以忽略
                if self.is_running:
                    print(f"Socket error: {e}")

    def get_latest_data(self):
        """
        主程序调用的函数，用于获取最新数据。
        此函数非阻塞，并立即返回当前可用的最新数据。
        """
        with self.lock:
            # 返回一个数据的副本
            data = self.latest_data
        return data

    def start(self):
        """启动后台接收线程。"""
        if not self.is_running:
            self.is_running = True
            self.receive_thread.start()
            print(f"数据接收器已在 {self.host}:{self.port} 启动...")

    def stop(self):
        """停止后台接收线程并关闭socket。"""
        if self.is_running:
            self.is_running = False
            # 关闭socket会让recvfrom调用立即返回一个错误，从而使线程退出循环
            self.sock.close()
            # 等待线程完全终止
            self.receive_thread.join()
            print("数据接收器已停止。")


# --- 如何使用 VIVEDataReceiver 类 ---
if __name__ == "__main__":
    # 1. 初始化并启动接收器
    #    这会立即在后台开始接收数据
    receiver = VIVEDataReceiver(port=9999, num_floats=36)
    receiver.start()

    try:
        # 2. 在你的主程序循环中，按你自己的频率调用 get_latest_data()
        #    这里我们模拟一个较慢的消费者，例如每秒只处理10次数据
        while True:
            latest_data = receiver.get_latest_data()
            
            if latest_data:
                # 如果已经收到过数据
                # 在这里进行你的数据处理...
                print(f"主程序获取到最新数据: [{latest_data[0]:.2f}, {latest_data[1]:.2f}, ...]")
            else:
                # 如果还没有收到任何数据
                print("等待第一份数据中...")

            latest_data = np.array(latest_data) if latest_data else None
            print(latest_data[0:7].sum() if latest_data is not None else "No data")
            print(latest_data[7:14].sum() if latest_data is not None else "No data")
            print(latest_data[14:].sum() if latest_data is not None else "No data")
            trans = np.array(latest_data[12:24])[:3] if latest_data is not None else None
            rot = np.array(latest_data[12:24])[3:].reshape(3, 3) if latest_data is not None else None
            # print(f"平移: {trans}, 旋转: {rot}")
            print(f"平移: {trans}")

            # 模拟主程序的其他工作，例如处理、渲染等
            time.sleep(0.5) # 以10Hz的频率运行

    except KeyboardInterrupt:
        print("主程序准备退出...")
    finally:
        # 3. 确保在程序结束时停止接收器
        receiver.stop()