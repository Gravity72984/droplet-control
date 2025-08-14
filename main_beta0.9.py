import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import threading
import socket
import numpy as np
from queue import Queue
import cv2
import yaml
import csv
import os
from video_panel import VideoPanel
from flow_predictor import FlowPredictor

class DataRecorder:
    """数据记录器类"""
    def __init__(self):
        self.recording = False
        self.data_buffer = []
        self.start_time = None
        self.output_file = None
        self.lock = threading.Lock()
        
    def start_recording(self):
        """开始数据记录"""
        with self.lock:
            self.recording = True
            self.data_buffer = []
            self.start_time = time.time()
            print("数据记录已启动")
    
    def stop_recording(self):
        """停止数据记录并保存文件"""
        with self.lock:
            if not self.recording:
                return None
            
            self.recording = False
            
            #弹出文件保存对话框
            file_path = filedialog.asksaveasfilename(
                title="保存数据记录",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"droplet_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if file_path and self.data_buffer:
                try:
                    self._save_to_csv(file_path)
                    print(f"数据已保存到: {file_path}")
                    return file_path
                except Exception as e:
                    print(f"保存数据失败: {str(e)}")
                    return None
            return None
    
    def add_data_point(self, continuous_pressure, dispersed_pressure, droplet_diameter,
                      continuous_flow, dispersed_flow, model1_pred, model3_pred, 
                      model4_pred, final_pred):
        """添加数据点"""
        with self.lock:
            if not self.recording:
                return
            
            current_time = time.time() - self.start_time
            
            data_point = {
                'time': current_time,
                'continuous_pressure': continuous_pressure,
                'dispersed_pressure': dispersed_pressure,
                'droplet_diameter': droplet_diameter,
                'continuous_flow': continuous_flow,
                'dispersed_flow': dispersed_flow,
                'model1_prediction': model1_pred,
                'model3_prediction': model3_pred,
                'model4_prediction': model4_pred,
                'final_prediction': final_pred
            }
            self.data_buffer.append(data_point)
    def _save_to_csv(self, file_path):
        """保存数据到CSV文件"""
        if not self.data_buffer:
            return
        fieldnames = [
            'time', 'continuous_pressure', 'dispersed_pressure', 'droplet_diameter',
            'continuous_flow', 'dispersed_flow', 'model1_prediction', 
            'model3_prediction', 'model4_prediction', 'final_prediction'
        ]
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data_buffer)

class PIDController:
    """改进的PID控制器（带抗积分饱和）"""
    def __init__(self, Kp, Ki, Kd, deadband=1, output_limits=(0, 2000)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.lock = threading.Lock()
        self.deadband = deadband
        self.output_limits = output_limits  # 添加输出限制范围
        self.reset()
    
    def reset(self):
        with self.lock:
            self.last_error = 0.0
            self.integral = 0.0
            self.last_time = time.time()
            self.saturated = False  # 添加饱和状态标志

    def compute(self, setpoint, measured_value):
        with self.lock:
            now = time.time()
            dt = now - self.last_time
            
            error = setpoint - measured_value
            if abs(error) < self.deadband:
                self.last_time = now  # 更新时间防止dt过大
                return 0.0
            
            # 比例项
            P = self.Kp * error
            
            # 积分项（带抗饱和）
            if not self.saturated:  #仅在非饱和状态更新积分
                self.integral += error * dt
            I = self.Ki * self.integral
            
            # 微分项
            if dt > 0:
                derivative = (error - self.last_error) / dt
            else:
                derivative = 0
            D = self.Kd * derivative
            
            # 计算原始输出
            raw_output = P + I + D
            
            # 检测饱和状态
            min_output, max_output = self.output_limits
            is_saturated = raw_output <= min_output or raw_output >= max_output
            
            # 当从饱和状态转为非饱和状态时重置积分
            if self.saturated and not is_saturated:
                self.integral = 0# 重置积分项
                self.saturated = False
            # 保存饱和状态
            self.saturated = is_saturated
            # 保存状态
            self.last_error = error
            self.last_time = now
            
            # 限制输出
            return max(min_output, min(max_output, raw_output))

class VideoInputManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.cap = None  # 确保初始化时设置cap属性
        self.load_config()
        self._init_input_source()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            self.config = {}
        self.input_type = self.config.get('input', {}).get('type', 'camera')
        
    def _init_input_source(self):
        """初始化输入源（确保先释放旧资源）"""
        # 释放旧资源
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        # 初始化新资源
        try:
            if self.input_type == "video":
                video_path = self.config['input']['video_path']
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap.isOpened():
                    raise RuntimeError(f"无法打开视频文件: {video_path}")
                self.loop = self.config['input']['loop']
                print(f"成功打开视频文件: {video_path}")
                
            elif self.input_type == "camera":
                camera_index = self.config['hardware']['camera_index']
                api_name = self.config['hardware']['capture_api'].upper()
                api_code = getattr(cv2, f"CAP_{api_name}", cv2.CAP_ANY)
                self.cap = cv2.VideoCapture(camera_index, api_code)
                if not self.cap.isOpened():
                    raise RuntimeError(f"无法打开摄像头: {camera_index}")
                print(f"成功打开摄像头: {camera_index}")
                
            else:
                raise ValueError(f"不支持的输入类型: {self.input_type}")

            self.frame_skip = self.config['input'].get('frame_skip', 0)
            self._skip_counter = 0
            return True  # 初始化成功
        except Exception as e:
            print(f"初始化输入源失败: {str(e)}")
            self.cap = None  # 确保设置为None
            return False# 初始化失败
    def reload_config(self, config_path=None):
        """重新加载配置并重新初始化输入源"""
        if config_path:
            self.config_path = config_path
        self.load_config()
        return self._init_input_source()  # 返回初始化结果
    
    def get_frame(self):
        """获取视频帧（自动处理循环和跳帧）"""
        # 检查cap是否有效
        if self.cap is None:
            # 尝试重新初始化
            print("输入源未初始化，尝试重新初始化...")
            if not self._init_input_source():
                # 返回黑色帧
                return np.zeros((480, 640, 3), dtype=np.uint8)
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.input_type == "video" and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # 验证重置是否成功
                    if self.cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                        print("视频循环失败")
                    else:
                        print("视频已重置到开头")
                    continue
                else:
                    error_msg = "视频播放结束" if self.input_type == "video" else "摄像头读取失败"
                    print(error_msg)
                    # 返回黑色帧
                    return np.zeros((480, 640, 3), dtype=np.uint8)
            
            #跳帧处理
            if self._skip_counter< self.frame_skip:
                self._skip_counter += 1
                continue
                
            self._skip_counter = 0
            return frame

    def release(self):
        """释放资源"""
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

class PressureController:
    """压力控制器（网络通信）"""
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f).get('pressure_server', {})
        # 从配置获取服务器信息
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 5001)
        self.device_id = self.config.get('device_id', "PC10200020022009002")
        
        # 通道映射配置
        self.channel_mapping = {
            'continuous': self.config.get('continuous_channel', 1),
            'dispersed': self.config.get('dispersed_channel', 2)
        }
        
        self.sock = None
        self.connected = False
        self.last_command_time = 0
        self.min_command_interval = 0.1# 最小命令间隔（秒）
        
    def connect(self):
        """连接到压力服务器"""
        try:
            if self.sock:
                self.sock.close()
                
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)  # 设置超时
            self.sock.connect((self.host, self.port))
            self.connected = True
            print(f"成功连接到压力服务器{self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接压力服务器失败: {str(e)}")
            self.connected = False
            return False
        
    def send_pressure_command(self, continuous_pressure, dispersed_pressure):
        """发送压力命令到服务器"""
        if not self.connected:
            if not self.connect():
                return False, "未连接"
                
        # 限制发送频率
        current_time = time.time()
        if current_time - self.last_command_time < self.min_command_interval:
            return True, "跳过（频率限制）"
            
        try:
            # 构建压力字典
            pressures = {
                self.channel_mapping['continuous']: continuous_pressure,
                self.channel_mapping['dispersed']: dispersed_pressure
            }
            
            # 构建命令
            command = f"{self.device_id}:"
            command += ",".join([f"P{ch},{pressure:.2f}" for ch, pressure in pressures.items()])
            command += "\n"
            
            # 发送命令
            self.sock.sendall(command.encode())
            
            # 接收响应（可选）
            try:
                response = self.sock.recv(1024).decode().strip()
            except socket.timeout:
                response = "无响应"
                
            self.last_command_time = current_time
            return True, response
        except Exception as e:
            self.connected = False
            return False, f"发送失败: {str(e)}"
            
    def close(self):
        """关闭连接"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False

class FlowController:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 5000
        self.device_id = "flow"
        self.sock = None
        self.connected = False
        self.last_read_time =0
        self.min_read_interval = 0.5  # 最小读取间隔（秒）
        self.last_cont_flow = 0.0
        self.last_disp_flow = 0.0
        self._data_buffer = ""  # 添加数据缓冲区

    def connect(self):
        """连接到流量传感器"""
        try:
            if self.sock:
                self.sock.close()
                
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(0.1)  # 短超时时间
            self.sock.connect((self.host, self.port))
            self.connected = True
            print(f"成功连接到流量传感器 {self.host}:{self.port}")
            
            #尝试发送初始命令
            try:
                self.sock.sendall(b"flow:?\n")
            except:
                pass
                
            return True
        except Exception as e:
            print(f"连接流量传感器失败: {str(e)}")
            self.connected = False
            return False
    
    def _parse_flow_data(self, raw_data):
        """解析流量数据"""
        try:
            # 去除首尾空白字符
            data = raw_data.strip()
            
            # 处理flow:前缀
            if data.startswith('flow:'):
                data = data[5:]# 去除'flow:'前缀
            # 分割数据
            parts = data.split(',')
            
            # 确保有足够的部分
            if len(parts) >= 4:
                # 直接按位置解析：F1, value1, F2, value2
                f1_label = parts[0].strip()
                f1_value = float(parts[1].strip())
                f2_label = parts[2].strip()
                f2_value = float(parts[3].strip())
                
                # 验证标签
                if f1_label == 'F1' and f2_label == 'F2':
                    return f1_value, f2_value
            
            print(f"数据格式不匹配: {data}")
            return None, None
            
        except Exception as e:
            print(f"解析流量数据失败: {e}, 原始数据: {repr(raw_data)}")
            return None, None
    def read_flows(self):
        """读取两相流量数据（支持主动推送模式）"""
        # 如果未连接，返回默认值
        if not self.connected:
            return self.last_cont_flow, self.last_disp_flow
        
        # 限制读取频率
        current_time = time.time()
        if current_time - self.last_read_time < self.min_read_interval:return self.last_cont_flow, self.last_disp_flow
        
        try:
            #尝试接收数据（非阻塞方式）
            try:
                new_data = self.sock.recv(1024).decode('utf-8', errors='ignore')
                if new_data:
                    self._data_buffer += new_data
            except socket.timeout:
                # 超时是正常的，使用缓冲区中的数据
                pass
            
            # 处理缓冲区中的数据
            if self._data_buffer:
                # 查找完整的行
                lines = self._data_buffer.split('\n')
                # 处理完整的行
                for line in lines[:-1]:  # 不处理最后一个可能不完整的行
                    line = line.strip()
                    if line and 'flow:' in line:
                        # 解析这行数据
                        cont_flow, disp_flow = self._parse_flow_data(line)
                        
                        if cont_flow is not None and disp_flow is not None:
                            self.last_cont_flow = cont_flow
                            self.last_disp_flow = disp_flow
                            self.last_read_time = current_time
                
                # 保留最后一个可能不完整的行
                self._data_buffer = lines[-1] if lines else ""
            # 如果长时间没有新数据，尝试发送查询命令
            if current_time - self.last_read_time > 5.0:# 5秒没有数据
                try:
                    self.sock.sendall(b"flow:?\n")
                except:
                    pass
            
            return self.last_cont_flow, self.last_disp_flow
            
        except Exception as e:
            print(f"读取流量失败: {str(e)}")
            self.connected = False
            return self.last_cont_flow, self.last_disp_flow
    
    def close(self):
        """关闭连接"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False
        print("流量传感器连接已关闭")


class SettingsWindow(tk.Toplevel):
    """配置设置窗口"""
    def __init__(self, parent, config_path="config.yaml"):
        super().__init__(parent)
        self.title("系统设置")
        self.geometry("800x600")
        self.config_path = config_path
        self.parent = parent
        self.vars = {}  # 存储所有变量
        
        # 加载配置
        self.config = self.load_config()
        
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建带滚动条的画布
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        # 配置滚动区域
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 创建配置表单
        self.create_form()
        
        # 创建按钮区域
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="保存", command=self.save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            return {}
    
    def create_form(self):
        """创建配置表单"""
        # 硬件配置
        hardware_frame = ttk.LabelFrame(self.scrollable_frame, text="硬件配置")
        hardware_frame.pack(fill=tk.X, padx=5, pady=5)
        self.add_entry(hardware_frame, "相机索引", "hardware.camera_index", 0)
        self.add_entry(hardware_frame, "采集API", "hardware.capture_api", "DSHOW")
        self.add_entry(hardware_frame, "像素比例", "hardware.pixel_ratio",1.96)
        
        # 模型配置
        model_frame = ttk.LabelFrame(self.scrollable_frame, text="模型配置")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        self.add_entry(model_frame, "权重路径", "model.weights_path", "")
        self.add_entry(model_frame, "置信度阈值", "model.confidence", 0.25)
        self.add_entry(model_frame, "设备类型", "model.device", "cpu")
        self.add_entry(model_frame, "图像尺寸", "model.imgsz", 640)
        
        #处理配置
        processing_frame = ttk.LabelFrame(self.scrollable_frame, text="处理配置")
        processing_frame.pack(fill=tk.X, padx=5, pady=5)
        self.add_entry(processing_frame, "历史大小", "processing.history_size", 5)
        self.add_entry(processing_frame, "最小直径(μm)", "processing.min_diameter", 5)
        self.add_entry(processing_frame, "最大直径(μm)", "processing.max_diameter", 500)
        self.add_entry(processing_frame, "边界阈值(像素)", "processing.boundary_threshold", 5)
        self.add_entry(processing_frame, "重复阈值(像素)", "processing.duplicate_threshold", 15)
        # 输入源配置
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="输入源配置")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        self.add_entry(input_frame, "输入类型", "input.type", "camera")
        self.add_entry(input_frame, "视频路径", "input.video_path", "")
        self.add_checkbox(input_frame, "循环播放", "input.loop", True)
        self.add_entry(input_frame, "跳帧设置", "input.frame_skip", 0)
        
        # 压力服务器配置
        server_frame = ttk.LabelFrame(self.scrollable_frame, text="压力服务器配置")
        server_frame.pack(fill=tk.X, padx=5, pady=5)
        self.add_entry(server_frame, "服务器地址", "pressure_server.host", "localhost")
        self.add_entry(server_frame, "服务器端口", "pressure_server.port", 5001)
        self.add_entry(server_frame, "设备ID", "pressure_server.device_id", "")
        self.add_entry(server_frame, "连续相通道", "pressure_server.continuous_channel", 1)
        self.add_entry(server_frame, "分散相通道", "pressure_server.dispersed_channel", 2)
        
        #系统配置
        system_frame = ttk.LabelFrame(self.scrollable_frame, text="系统配置")
        system_frame.pack(fill=tk.X, padx=5, pady=5)
        self.add_entry(system_frame, "最小压力(mbar)", "system.pressure_limits[0]", 0)
        self.add_entry(system_frame, "最大压力(mbar)", "system.pressure_limits[1]", 1500)
        # 添加模型路径配置区域
        model_frame = ttk.LabelFrame(self.scrollable_frame, text="模型路径配置")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.add_entry(model_frame, "模型目录", "model_paths.model_dir", "rf_models")
        self.add_entry(model_frame, "模型1路径", "model_paths.model_1", "model_1_rf.pkl")
        self.add_entry(model_frame, "模型3路径", "model_paths.model_3", "model_3_comprehensive_rf.pkl")
        self.add_entry(model_frame, "模型4路径", "model_paths.model_4", "model_4_cp_droplet_rf.pkl")
        self.add_entry(model_frame, "标准化器1", "model_paths.scaler_1", "model_1_scaler.pkl")
        self.add_entry(model_frame, "标准化器3", "model_paths.scaler_3", "model_3_scaler.pkl")
        self.add_entry(model_frame, "标准化器4", "model_paths.scaler_4", "model_4_scaler.pkl")

    def add_entry(self, parent, label, key, default):
        """添加文本输入框"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(frame, text=label, width=20).pack(side=tk.LEFT)
        
        # 从配置中获取值或使用默认值
        value = self.get_nested_value(key, default)
        var = tk.StringVar(value=str(value))
        self.vars[key] = var
        
        entry = ttk.Entry(frame, textvariable=var, width=30)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def add_checkbox(self, parent, label, key, default):
        """添加复选框"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(frame, text=label, width=20).pack(side=tk.LEFT)
        
        # 从配置中获取值或使用默认值
        value = self.get_nested_value(key, default)
        var = tk.BooleanVar(value=value)
        self.vars[key] = var
        
        checkbox = ttk.Checkbutton(frame, variable=var)
        checkbox.pack(side=tk.LEFT)
    
    def get_nested_value(self, key, default):
        """从嵌套字典中获取值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if '[' in k and ']' in k:
                # 处理列表索引
                base_key = k.split('[')[0]
                index = int(k.split('[')[1].split(']')[0])
                if base_key in value and isinstance(value[base_key], list) and len(value[base_key]) > index:
                    value = value[base_key][index]
                else:
                    return default
            else:
                if k in value:
                    value = value[k]
                else:return default
        return value
    
    def save_config(self):
        """保存配置到文件"""
        # 收集所有值
        new_config = self.config.copy()
        
        for key, var in self.vars.items():
            keys = key.split('.')
            current = new_config
            
            for i, k in enumerate(keys[:-1]):
                if '[' in k and ']' in k:
                    # 处理列表索引
                    base_key = k.split('[')[0]
                    index = int(k.split('[')[1].split(']')[0])
                    if base_key not in current:
                        current[base_key] = []
                    while len(current[base_key]) <= index:
                        current[base_key].append(None)
                    current = current[base_key][index]
                else:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
            
            last_key = keys[-1]
            if '[' in last_key and ']' in last_key:
                base_key = last_key.split('[')[0]
                index = int(last_key.split('[')[1].split(']')[0])
                if base_key not in current:
                    current[base_key] = []
                while len(current[base_key]) <= index:
                    current[base_key].append(None)
                
                # 转换值类型
                if isinstance(var, tk.BooleanVar):
                    value = var.get()
                else:
                    value_str = var.get()
                    try:
                        # 尝试转换为整数
                        value = int(value_str)
                    except ValueError:
                        try:
                            # 尝试转换为浮点数
                            value = float(value_str)
                        except ValueError:
                            # 保持为字符串
                            value = value_str
                
                current[base_key][index] = value
            else:
                # 转换值类型
                if isinstance(var, tk.BooleanVar):
                    current[last_key] = var.get()
                else:    
                    value_str = var.get()
                    try:
                        # 尝试转换为整数
                        current[last_key] = int(value_str)
                    except ValueError:
                        try:
                            # 尝试转换为浮点数            
                            current[last_key] = float(value_str)        
                        except ValueError:
                            # 保持为字符串
                            current[last_key] = value_str
        
        # 保存到文件
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, allow_unicode=True)
            
            # 通知主程序重新加载配置
            self.parent.reload_config(new_config)
            tk.messagebox.showinfo("保存成功", "配置已成功保存并应用！")
            self.destroy()
        except Exception as e:
            tk.messagebox.showerror("保存失败", f"保存配置时出错: {str(e)}")

class ControlSystem(tk.Tk):
    """主控制系统(集成GUI和通信接口)"""
    def __init__(self, data_queue: Queue):
        super().__init__()
        self.data_queue = data_queue
        self.pid_enabled = False
        self.title("微滴智能控制系统beta0.8")
        self.geometry("1024x768")
        
        # 初始化数据记录器
        self.data_recorder = DataRecorder()
        self.recording_enabled = False
        
        #初始化视频源
        self.video_source = VideoInputManager()
        self.show_video = tk.BooleanVar(value=True)
        self.input_type = self.video_source.input_type
        
        #初始化控制器
        self.pressure_controller = PressureController()
        self.flow_controller = FlowController()
        
        # 初始化参数
        self.target_size = 100.0
        self.continuous_pressure = 100.0
        self.dispersed_pressure = 100.0
        self.cont_flow = 0.0
        self.disp_flow = 0.0
        
        # 性能监控
        self.fps = 0.0
        self.processing_latency = 0.0
        # 预测结果显示变量
        self.predicted_flow = tk.StringVar(value="预测流量: -")
        self.prediction_status = tk.StringVar(value="状态: 未启用")
        self.generation_frequency = tk.StringVar(value="生成频率: -")

        # 新增流量预测器
        self.flow_predictor = FlowPredictor()

        # 分析器
        from analyzer import DropletAnalyzer
        self.analyzer = DropletAnalyzer(
            data_queue=self.data_queue,
            config_path="config.yaml"
        )
        
        # PID控制器（只初始化连续相）
        self.continuous_pid = PIDController(
            0.00, 0.01, 0.00,
            deadband=1,
            output_limits=(-20, 20)
        )
        
        # 创建UI
        self.create_widgets()

        # 添加训练模式标志
        self.training_mode = False

        # 压力范围
        self.pressure_limits = (0.0, 2000.0)
        
        # 线程控制标志
        self.video_running = True
        self.pressure_running = True
        self.flow_running = True
        self.data_recording_running = True
        
        # 共享数据
        self.current_diameters = None
        self.current_size = 0.0
        self.current_diameters = []  # 当前帧检测到的直径
        self.last_diameter_update = time.time()# 上次更新时间

        # 训练模块
        from train_module import ModelTrainer
        self.trainer = ModelTrainer(self, self.pressure_controller, self.analyzer)
        
        # 初始连接
        self.connect_to_pressure_server()
        self.connect_to_flow_sensor()
        # 启动线程
        self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
        self.video_thread.start()
        
        self.pressure_thread = threading.Thread(target=self.pressure_control_loop, daemon=True)
        self.pressure_thread.start()
        
        self.flow_thread = threading.Thread(target=self.flow_reading_loop, daemon=True)
        self.flow_thread.start()
        
        # 启动数据记录线程
        self.data_recording_thread = threading.Thread(target=self.data_recording_loop, daemon=True)
        self.data_recording_thread.start()# 窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def data_recording_loop(self):
        """数据记录循环- 每0.2秒记录一次数据"""
        last_record_time = time.time()
        
        while self.data_recording_running:
            try:
                current_time = time.time()
                
                # 检查是否需要记录数据（每0.2秒）
                if (self.recording_enabled and 
                    current_time - last_record_time >= 0.2):
                    # 获取当前预测结果
                    prediction_data = {
                        'model1': 0.0,
                        'model3': 0.0,
                        'model4': 0.0,
                        'final': 0.0
                    }    
                    if hasattr(self.flow_predictor, 'last_predictions'):
                        prediction_data['model1'] = self.flow_predictor.last_predictions.get('model1', 0.0)
                        prediction_data['model3'] = self.flow_predictor.last_predictions.get('model3', 0.0)
                        prediction_data['model4'] = self.flow_predictor.last_predictions.get('model4', 0.0)
                        prediction_data['final'] = prediction_data['model3']  # 使用model3作为最终预测
                    # 记录数据点
                    self.data_recorder.add_data_point(
                        continuous_pressure=self.continuous_pressure,
                        dispersed_pressure=self.dispersed_pressure,
                        droplet_diameter=self.current_size if self.current_size > 0 else 0.0,
                        continuous_flow=self.cont_flow,
                        dispersed_flow=self.disp_flow,
                        model1_pred=prediction_data['model1'],
                        model3_pred=prediction_data['model3'],
                        model4_pred=prediction_data['model4'],
                        final_pred=prediction_data['final']
                    )
                    
                    last_record_time = current_time
                
                # 控制循环频率
                time.sleep(0.05)  # 50ms检查间隔
            except Exception as e:
                print(f"数据记录循环异常: {str(e)}")
                time.sleep(0.1)

    def toggle_data_recording(self):
        """切换数据记录状态"""
        if not self.recording_enabled:
            # 开始记录
            self.recording_enabled = True
            
            # 如果未启用智能接管，自动启动液滴检测
            if not self.pid_enabled:
                self.start_droplet_detection_only()
            
            # 开始数据记录
            self.data_recorder.start_recording()
            # 更新按钮文本
            self.data_record_btn.config(text="停止数据记录")
            print("数据记录已启动")
            
        else:
            #停止记录
            self.recording_enabled = False
            
            # 停止数据记录并保存文件
            saved_file = self.data_recorder.stop_recording()
            
            # 如果之前自动启动了液滴检测，现在停止它
            if not self.pid_enabled:self.stop_droplet_detection_only()
            
            # 更新按钮文本
            self.data_record_btn.config(text="数据记录")
            
            if saved_file:
                messagebox.showinfo("记录完成", f"数据已保存到:\n{saved_file}")
            else:
                messagebox.showwarning("记录停止", "数据记录已停止，但未保存文件")

    def start_droplet_detection_only(self):
        """仅启动液滴检测（不启动PID控制）"""
        # 这里不需要特殊操作，因为video_processing_loop会检查recording_enabled状态
        print("液滴检测已启动（仅用于数据记录）")

    def stop_droplet_detection_only(self):
        """停止仅用于记录的液滴检测"""
        print("液滴检测已停止")

    def connect_to_flow_sensor(self):
        """连接到流量传感器（可选连接）"""
        success = self.flow_controller.connect()
        if success:
            print("流量传感器连接成功")
        else:
            print("流量传感器未连接，流量显示将为0")
        return success

    def flow_reading_loop(self):
        """独立的流量读取循环"""
        while self.flow_running:
            try:# 尝试读取流量数据
                cont_flow, disp_flow = self.flow_controller.read_flows()
                self.cont_flow = cont_flow
                self.disp_flow = disp_flow
                # 更新UI显示（使用after确保在主线程中执行）
                self.after(0, self.update_flow_display)
                
            except Exception as e:
                print(f"流量读取循环异常: {str(e)}")
            
            # 控制读取频率
            time.sleep(0.5)

    def update_flow_display(self):
        """更新流量显示（在主线程中执行）"""
        try:
            self.cont_flow_label.config(text=f"{self.cont_flow:.2f} μL/min")
            self.disp_flow_label.config(text=f"{self.disp_flow:.2f} μL/min")
        except Exception as e:
            print(f"更新流量显示失败: {str(e)}")

    def pressure_control_loop(self):
        """独立的压力控制循环"""
        try:
            while self.pressure_running:
                if self.pid_enabled and hasattr(self, 'current_diameters') and self.current_diameters:    
                    # 连续相PID计算（反比关系）
                    cont_output = self.continuous_pid.compute(
                        self.target_size, self.current_size
                    )
                    # 分散相PID计算（正比关系）
                    disp_output = self.dispersed_pid.compute(
                        self.target_size, self.current_size
                    )    
                    # 更新压力值
                    self.continuous_pressure = np.clip(
                        self.continuous_pressure - cont_output,
                        *self.pressure_limits
                    )
                    self.dispersed_pressure = np.clip(
                        self.dispersed_pressure + disp_output,
                        *self.pressure_limits    )
                    
                    # 发送控制指令
                    self.send_pressure_command()

                if self.pid_enabled and self.flow_predictor:
                    try:
                        prediction = self.flow_predictor.predict_with_timing()
                        if prediction:
                            self.after(0, lambda: self.update_prediction_display(prediction))
                    except Exception as e:
                        print(f"流量预测失败: {str(e)}")
                
                # 控制循环频率（10Hz）
                time.sleep(0.1)
                
        except Exception as e:
            print(f"压力控制循环异常: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_close(self):
        """窗口关闭事件处理"""
        print("\n正在关闭程序...")
        # 停止控制循环
        self.video_running = False
        self.pressure_running = False
        self.flow_running = False
        self.data_recording_running = False
        
        #停止数据记录
        if self.recording_enabled:
            self.data_recorder.stop_recording()
    
        # 确保视频源释放
        self.video_source.release()
    
        # 等待线程完成（增加超时时间）
        if self.video_thread.is_alive():
            print("等待视频线程结束...")
            self.video_thread.join(timeout=1.0)
        if self.pressure_thread.is_alive():
            print("等待压力线程结束...")
            self.pressure_thread.join(timeout=1.0)
        if hasattr(self, 'flow_thread') and self.flow_thread.is_alive():
            print("等待流量线程结束...")
            self.flow_thread.join(timeout=1.0)
        if hasattr(self, 'data_recording_thread') and self.data_recording_thread.is_alive():
            print("等待数据记录线程结束...")
            self.data_recording_thread.join(timeout=1.0)
            
        # 关闭压力控制器
        self.pressure_controller.close()
        self.flow_controller.close()
        
        # 确保分析器释放资源
        if hasattr(self, 'analyzer'):
            self.analyzer.release()
    
        # 最后销毁窗口
        self.destroy()
        print("程序已完全关闭")
            
    def video_processing_loop(self):
        """独立的视频处理循环"""
        last_update_time = time.time()
        frame_count = 0
        
        try:
            while self.video_running:
                # 获取输入帧
                try:
                    frame = self.video_source.get_frame()
                    frame_count += 1
                except RuntimeError as e:
                    print(f"输入源异常: {str(e)}")
                    if self.input_type == "camera":
                        time.sleep(0.5)  # 摄像头异常时稍作等待
                        continue
                    else:
                        # 如果是视频文件结束，则尝试重置
                        if self.video_source.loop:
                            self.video_source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                
                # 执行液滴检测（仅在启用PID、训练模式或数据记录模式时）
                if self.pid_enabled or getattr(self, 'training_mode', False) or self.recording_enabled:
                    start_process = time.time()
                    # 获取ROI区域
                    roi = self.video_panel.get_roi()
                
                    # 根据ROI处理帧
                    diameters = self.analyzer._process_frame(frame, roi)
                    
                    process_time = time.time() - start_process
                    
                    if diameters:
                        self.current_diameters = diameters
                        self.current_size = np.mean(diameters)
                    else:
                        self.current_diameters = []
                        self.current_size = 0.0
                    self.last_diameter_update = time.time()    
                    # 向预测器添加数据点
                    if self.current_size > 0:
                        self.flow_predictor.add_data_point(
                            self.continuous_pressure,
                            self.dispersed_pressure,
                            self.current_size
                        )
                else:
                    diameters = None
                    process_time =0
                    self.current_diameters = None
                    self.current_size = 0.0
                # 执行液滴预测
                if ((self.pid_enabled or self.recording_enabled) and 
                    hasattr(self, 'current_size') and self.current_size > 0 and self.flow_predictor):
                    try:
                        prediction = self.flow_predictor.predict(
                            self.continuous_pressure,
                            self.dispersed_pressure,
                            self.current_size
                        )        
                        self.after(0, lambda: self.update_prediction_display(prediction))
                    except Exception as e:
                        print(f"流量预测失败: {str(e)}")
                # 更新性能监控数据
                current_time = time.time()
                time_diff = current_time - last_update_time
                if time_diff >= 1.0:  # 每秒更新一次性能数据
                    self.fps = frame_count / time_diff
                    self.processing_latency = process_time * 1000  # ms    
                    frame_count = 0
                    last_update_time = current_time
                    
                    # 更新UI显示
                    self.after(0, self.update_display)
                # 更新实时显示- 使用after避免在非主线程访问tkinter变量
                display_frame = self._annotate_frame(frame.copy(), diameters)
                self.after(0, lambda f=display_frame: self._update_video_if_enabled(f))
                
                # 添加小延迟以避免过度占用CPU
                time.sleep(0.01)
                
        except Exception as e:
            print(f"视频处理循环异常: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保释放视频资源
            self.video_source.release()

    def _update_video_if_enabled(self, frame):
        """在主线程中检查是否需要更新视频显示"""
        try:
            if self.show_video.get():
                self.video_panel.update_frame(frame)
        except Exception as e:
            print(f"更新视频显示失败: {str(e)}")

    def update_prediction_display(self, prediction):
        """更新预测结果显示"""
        # 计算生成频率
        frequency_text = self.calculate_generation_frequency(prediction)
        
        if prediction['is_consistent']:
            self.predicted_flow.set(f"预测流量: {prediction['prediction']:.2f} μL/min")
            self.prediction_status.set("状态: 正常")
            self.generation_frequency.set(frequency_text)
            self.predicted_flow_label.config(foreground="green")
            self.prediction_status_label.config(foreground="green")
            self.generation_frequency_label.config(foreground="green")
        else:
            self.predicted_flow.set(f"预测流量: {prediction['prediction']:.2f} μL/min")
            self.prediction_status.set("状态: 检测失效")
            self.generation_frequency.set("生成频率: 计算失败")
            self.predicted_flow_label.config(foreground="red")
            self.prediction_status_label.config(foreground="red")
            self.generation_frequency_label.config(foreground="red")

    def calculate_generation_frequency(self, prediction):
        """计算液滴生成频率"""
        try:
            if (prediction['is_consistent'] and 
                hasattr(self, 'current_size') and 
                self.current_size > 0):
                
                # 使用预测的分散相流量（μL/min）
                dispersed_flow = prediction['prediction']  # μL/min
                
                # 计算单个液滴体积（μL）
                diameter_um = self.current_size  # μm
                radius_cm = diameter_um * 1e-4 / 2  # 转换为cm
                volume_cm3 = (4/3) * np.pi * (radius_cm ** 3)  # cm³
                volume_ul = volume_cm3 * 1000  # 转换为μL
                
                # 计算频率（滴/秒）
                frequency_per_min = dispersed_flow / volume_ul  # 滴/分钟
                frequency_per_sec = frequency_per_min / 60  # 滴/秒
                
                return f"生成频率: {frequency_per_sec:.2f}/s"
            else:
                return "生成频率: 计算失败"
        except Exception as e:
            print(f"频率计算失败: {str(e)}")
            return "生成频率: 计算失败"



    def open_settings(self):
        """打开设置窗口"""
        SettingsWindow(self)
    
    def reload_config(self, new_config):
        """重新加载配置"""
        print("重新加载配置...")# 保存新配置到文件
        try:
            with open("config.yaml",'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, allow_unicode=True)
        except Exception as e:
            print(f"保存配置失败: {str(e)}")

        # 更新压力控制器配置
        if 'pressure_server' in new_config:
            self.pressure_controller.config = new_config['pressure_server']
            
            # 尝试重新连接
            self.connect_to_pressure_server()
        # 更新系统配置
        if 'system' in new_config:
            self.pressure_limits = tuple(new_config['system']['pressure_limits'])
            print(f"更新压力范围: {self.pressure_limits}")
        
        # 更新输入源配置
        if 'input' in new_config or 'hardware' in new_config:# 重新初始化视频源
            self.video_source.reload_config()  # 使用新方法重新加载
            self.input_type = self.video_source.input_type
            print(f"更新输入源: {self.input_type}")
        
        # 更新处理配置
        if 'processing' in new_config:
            # 重新初始化分析器
            self.analyzer.config['processing'] = new_config['processing']
            print("更新处理配置")

    def connect_to_pressure_server(self):
        """连接到压力服务器（带状态更新）"""
        success = self.pressure_controller.connect()
        status = "已连接" if success else "连接失败"
        color = "green" if success else "red"
        self.comm_status.config(text=f"通信状态: {status}", foreground=color)
        return success

    def create_widgets(self):
        """创建GUI界面组件"""
        # 创建主框架（左右分栏）
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        # 右侧视频面板
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ========================
        # 左侧控制区域
        # ========================
        
        # 显示设置
        display_frame = ttk.LabelFrame(left_panel, text="设置")
        display_frame.pack(padx=5, pady=5, fill=tk.X)
        ttk.Checkbutton(
            display_frame,
            text="显示实时视频",
            variable=self.show_video
        ).pack(pady=(5, 0))
        
        # 添加设置按钮到显示设置区域
        settings_btn = ttk.Button(
            display_frame, 
            text="系统配置", 
            command=self.open_settings,
            width=10
        )
        settings_btn.pack(pady=(0, 5))

        training_btn = ttk.Button(
            display_frame, 
            text="数据收集", 
            command=self.toggle_training_mode,
            width=10
        )
        training_btn.pack(pady=(0, 5))

        # 新增数据记录按钮
        self.data_record_btn = ttk.Button(
            display_frame, 
            text="数据记录", 
            command=self.toggle_data_recording,
            width=10
        )
        self.data_record_btn.pack(pady=(0, 5))

        # 网络连接区域
        network_frame = ttk.LabelFrame(left_panel, text="网络连接")
        network_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # 连接按钮
        self.connect_btn = ttk.Button(
            network_frame,
            text="连接压力服务器",
            command=self.connect_to_pressure_server
        )
        self.connect_btn.pack(pady=5)

        # 流量传感器连接按钮
        self.flow_connect_btn = ttk.Button(
            network_frame,
            text="连接流量传感器",
            command=self.connect_to_flow_sensor
        )
        self.flow_connect_btn.pack(pady=5)
        
        # 参数输入区
        control_frame = ttk.LabelFrame(left_panel, text="控制参数")
        control_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # 目标粒径
        ttk.Label(control_frame, text="目标粒径 (μm):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.target_entry = ttk.Entry(control_frame, width=10)
        self.target_entry.insert(0, "100.0")
        self.target_entry.grid(row=0, column=1, pady=2)
        
        # PID参数
        pid_params = [
            ("Kp", "0.05"), ("Ki", "0.00005"), ("Kd", "0")
        ]
        for i, (label, default) in enumerate(pid_params, start=1):
            ttk.Label(control_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(control_frame, width=10)
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=2)
            setattr(self, f"{label.lower()}_entry", entry)
        
        # 控制按钮
        self.toggle_btn = ttk.Button(
            control_frame, 
            text="启动智能接管", 
            command=self.toggle_pid
        )
        self.toggle_btn.grid(row=4, columnspan=2, pady=5)

        # 压力控制区域
        pressure_frame = ttk.LabelFrame(left_panel, text="压力控制")
        pressure_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # 连续相压力控制
        ttk.Label(pressure_frame, text="连续相压力 (mbar):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.continuous_pressure_entry = ttk.Entry(pressure_frame, width=10)
        self.continuous_pressure_entry.insert(0, "100.0")
        self.continuous_pressure_entry.grid(row=0, column=1, pady=2)
        
        # 分散相压力控制
        ttk.Label(pressure_frame, text="分散相压力 (mbar):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.dispersed_pressure_entry = ttk.Entry(pressure_frame, width=10)
        self.dispersed_pressure_entry.insert(0, "100.0")
        self.dispersed_pressure_entry.grid(row=1, column=1, pady=2)
        # 设置压力按钮
        self.set_pressure_btn = ttk.Button(
            pressure_frame,
            text="设置压力", 
            command=self.set_pressures
        )
        self.set_pressure_btn.grid(row=2, columnspan=2, pady=5)
        
        # 实时状态区域
        status_frame = ttk.LabelFrame(left_panel, text="实时状态")
        status_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # 压力显示
        ttk.Label(status_frame, text="连续相压力:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.continuous_label = ttk.Label(status_frame, text="100.0 mbar")
        self.continuous_label.grid(row=0, column=1, pady=2)
        
        ttk.Label(status_frame, text="分散相压力:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.dispersed_label = ttk.Label(status_frame, text="100.0 mbar")
        self.dispersed_label.grid(row=1, column=1, pady=2)
        
        # 新增流量显示
        ttk.Label(status_frame, text="连续相流量:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.cont_flow_label = ttk.Label(status_frame, text="0.0 μL/min")
        self.cont_flow_label.grid(row=2, column=1, pady=2)
        
        ttk.Label(status_frame, text="分散相流量:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.disp_flow_label = ttk.Label(status_frame, text="0.0 μL/min")
        self.disp_flow_label.grid(row=3, column=1, pady=2)

        ttk.Label(status_frame, text="预测流量:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.predicted_flow_label = ttk.Label(status_frame, textvariable=self.predicted_flow)
        self.predicted_flow_label.grid(row=4, column=1, pady=2)
        
        ttk.Label(status_frame, text="预测状态:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.prediction_status_label = ttk.Label(status_frame, textvariable=self.prediction_status)
        self.prediction_status_label.grid(row=5, column=1, pady=2)

        ttk.Label(status_frame, text="生成频率:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.generation_frequency_label = ttk.Label(status_frame, textvariable=self.generation_frequency)
        self.generation_frequency_label.grid(row=6, column=1, pady=2)


        # 通信状态
        self.comm_status = ttk.Label(status_frame, text="通信状态: 压力控制未连接", foreground="red")
        self.comm_status.grid(row=7, columnspan=2, pady=2)
        
        # ========================
        # 右侧视频区域
        # ========================
        
        # 视频显示区域
        self.video_panel = VideoPanel(self.right_panel)
        self.video_panel.pack(fill=tk.BOTH, expand=True, pady=5)
    def set_pressures(self):
        """设置新的压力值"""
        try:
            # 获取输入的压力值
            new_continuous = float(self.continuous_pressure_entry.get())
            new_dispersed = float(self.dispersed_pressure_entry.get())
            
            # 确保在安全范围内
            min_p, max_p = self.pressure_limits
            if min_p <= new_continuous <= max_p and min_p <= new_dispersed <= max_p:
                self.continuous_pressure = new_continuous
                self.dispersed_pressure = new_dispersed
                
                # 更新UI显示
                self.update_display()
                
                # 发送新的压力命令
                self.send_pressure_command()
            else:
                self.comm_status.config(text="通信状态: 压力超限", foreground="red")
        except ValueError:
            self.comm_status.config(text="通信状态: 输入无效", foreground="red")

    def _toggle_video_display(self):
        """切换视频显示状态"""
        if self.show_video.get():
            self.video_panel.pack(fill=tk.BOTH, expand=True)
        else:
            self.video_panel.pack_forget()

    def toggle_pid(self):
        """切换PID控制状态"""
        self.pid_enabled = not self.pid_enabled
        if self.pid_enabled:
            # 获取参数
            self.target_size = float(self.target_entry.get())
            kp = float(self.kp_entry.get())
            ki = float(self.ki_entry.get())
            kd = float(self.kd_entry.get())
            
            # 重置控制器
            self.continuous_pid = PIDController(
                kp, ki, kd,
                deadband=1,
                output_limits=(-500, 500))
            self.dispersed_pid = PIDController(
                kp, ki, kd,
                deadband=1,
                output_limits=(-500, 500))
            
            self.toggle_btn.config(text="停止智能接管")
            self.comm_status.config(text="通信状态: AI控制", foreground="green")
        else:
            self.toggle_btn.config(text="启动智能接管")
            self.comm_status.config(text="通信状态: 手动模式", foreground="blue")
    
    def toggle_training_mode(self):
        """切换训练模式"""
        if not hasattr(self, 'trainer'):
            from train_module import ModelTrainer
            self.trainer = ModelTrainer(self, self.pressure_controller, self.analyzer)
        if not self.pid_enabled and not getattr(self, 'training_mode', False):
            self.trainer.start_training()
        else:
            if self.pid_enabled:
                messagebox.showwarning("警告", "请先停止PID控制再进入训练模式")
            else:
                messagebox.showwarning("警告", "训练已在进行中")

    def _annotate_frame(self, frame, diameters):
        """在视频帧上添加标注信息"""
        # 确保所有属性都已初始化
        fps_display = getattr(self, 'fps', 0.0)
        latency_display = getattr(self, 'processing_latency', 0.0)
        continuous_pressure = getattr(self, 'continuous_pressure', 100.0)
        dispersed_pressure = getattr(self, 'dispersed_pressure', 100.0)
        
        # 获取图像尺寸
        height, width = frame.shape[:2]
        
        # 左上角显示性能数据
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        if self.pid_enabled or self.recording_enabled:  
            # 在PID或记录模式时显示延迟
            cv2.putText(frame, f"Latency: {latency_display:.1f}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0), 2)
        # 右上角显示压力信息
        cv2.putText(frame, f"Continuous: {continuous_pressure:.1f}mbar", (width - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"Dispersed: {dispersed_pressure:.1f}mbar", (width - 300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # 左下角显示检测结果
        if diameters is not None and diameters:  # 检查diameters不为None且不为空
            avg_size = np.mean(diameters)
            cv2.putText(frame, f"Avg Size: {avg_size:.2f}μm", (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Droplets: {len(diameters)}", (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
        elif self.input_type == "camera" and not self.pid_enabled and not self.recording_enabled:
            cv2.putText(frame, "Camera: Streaming", (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
        
        # 右下角显示状态信息
        status_text = "AI: " + ("ON" if self.pid_enabled else "OFF")
        if self.recording_enabled:
            status_text += " | REC"
        cv2.putText(frame, status_text, 
                   (width - 150, height - 30),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if (self.pid_enabled or self.recording_enabled) else (0, 0, 255), 2)
        
        return frame
    
    def update_display(self):
        """更新界面显示"""
        #获取饱和状态
        try:
            cont_saturated = self.continuous_pid.saturated
            disp_saturated = self.dispersed_pid.saturated
        except AttributeError:
            cont_saturated = False
            disp_saturated = False
        # 更新标签（添加颜色指示）
        cont_text = f"{self.continuous_pressure:.1f} mbar"
        disp_text = f"{self.dispersed_pressure:.1f} mbar"
        if cont_saturated:
            self.continuous_label.config(text=cont_text, foreground="red")
        else:
            self.continuous_label.config(text=cont_text, foreground="black")
    
        if disp_saturated:
            self.dispersed_label.config(text=disp_text, foreground="red")
        else:
            self.dispersed_label.config(text=disp_text, foreground="black")
    
    def send_pressure_command(self):
        """压力值发送接口（使用网络连接）"""
        try:
            # 发送压力命令
            success, response = self.pressure_controller.send_pressure_command(
                self.continuous_pressure,
                self.dispersed_pressure
            )
            
            if success:
                self.comm_status.config(text="通信状态: 已发送", foreground="blue")
            else:
                self.comm_status.config(text="通信状态: 失败", foreground="red")
        except Exception as e:
            self.comm_status.config(text="通信状态: 异常", foreground="red")

if __name__ == "__main__":
    # 初始化数据管道
    data_queue = Queue(maxsize=10)
    
    # 启动控制系统
    app = ControlSystem(data_queue)
    app.mainloop()
