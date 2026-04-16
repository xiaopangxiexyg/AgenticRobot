import socket
import json

class UDPReceiver:
    def __init__(self, host='0.0.0.0', port=12345):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        print(f"Listening on {host}:{port}")
    
    def receive(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                print(f"Received {len(data)} bytes from {addr}")
                
                # 尝试多种处理方式
                processed = self.process_data(data)
                if processed:
                    yield processed
                    
            except socket.error as e:
                print(f"Socket error: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
    
    def process_data(self, data):
        # 尝试JSON解码
        try:
            json_data = json.loads(data.decode('utf-8'))
            return {'type': 'json', 'data': json_data}
        except:
            pass
        
        # 尝试文本解码
        try:
            text_data = data.decode('utf-8')
            return {'type': 'text', 'data': text_data}
        except UnicodeDecodeError:
            # 作为二进制数据处理
            return {'type': 'binary', 'data': data, 'hex': data.hex()}

# 使用
receiver = UDPReceiver(port=9999)
for message in receiver.receive():
    print(f"Message type: {message['type']}")
    print(f"Data: {message['data']}")