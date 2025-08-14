import csv
import time
import random
import threading
from queue import Queue
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class ModelTrainer:
    def __init__(self, parent, pressure_controller, analyzer):
        self.parent = parent
        self.pressure_controller = pressure_controller
        self.analyzer = analyzer
        self.training_active = False
        self.data_points = []
        self.current_point = 0
        self.total_points = 1800
        
        # 创建UI更新队列
        self.ui_queue = Queue()
    
    def generate_pressure_sequence(self):
            """生成符合要求的压力序列（DP/CP比值在0.5-1.5之间，压力不超过500mbar）"""
            points = []
            current_cp = 200  # 初始连续相压力
            current_dp = 200  # 初始分散相压力
            
            # 定义约束条件
            min_pressure = 50
            max_pressure = 500
            min_ratio = 0.5  # 1/2
            max_ratio = 1.5  # 3/2
            
            while len(points) < self.total_points:
                # 生成随机步进
                cp_step = random.randint(-20, 20)
                dp_step = random.randint(-20, 20)
                
                # 计算新的压力值
                new_cp = current_cp + cp_step
                new_dp = current_dp + dp_step
                
                # 限制压力在允许范围内
                new_cp = max(min_pressure, min(max_pressure, new_cp))
                new_dp = max(min_pressure, min(max_pressure, new_dp))
                
                # 计算压力比值
                ratio = new_dp / new_cp
                
                # 如果比值不在允许范围内，调整分散相压力
                if ratio < min_ratio:
                    # 比值太小，增加分散相压力到最小允许值
                    new_dp = new_cp * min_ratio
                elif ratio > max_ratio:
                    # 比值太大，减少分散相压力到最大允许值
                    new_dp = new_cp * max_ratio
                
                # 再次确保分散相压力在允许范围内
                new_dp = max(min_pressure, min(max_pressure, new_dp))
                
                # 如果调整后的分散相压力导致比值超出范围，调整连续相压力
                final_ratio = new_dp / new_cp
                if final_ratio < min_ratio or final_ratio > max_ratio:
                    # 根据分散相压力反算连续相压力
                    if new_dp / max_ratio > min_pressure:
                        new_cp = new_dp / max_ratio
                    elif new_dp / min_ratio < max_pressure:
                        new_cp = new_dp / min_ratio
                    else:
                        # 如果无法满足条件，保持当前值不变，重新生成
                        continue
                
                # 最终确保连续相压力也在范围内
                new_cp = max(min_pressure, min(max_pressure, new_cp))
                
                # 最后验证是否满足所有条件
                final_ratio = new_dp / new_cp
                if (min_pressure <= new_cp <= max_pressure and 
                    min_pressure <= new_dp <= max_pressure and
                    min_ratio <= final_ratio <= max_ratio):
                    points.append((new_cp, new_dp))
                    current_cp, current_dp = new_cp, new_dp
            
            return points[:self.total_points]

            
    

    
    def collect_data_point(self, cp_pressure, dp_pressure):
        """收集单个数据点 - 使用严格的实时数据"""
        # 设置压力
        success, response = self.pressure_controller.send_pressure_command(cp_pressure, dp_pressure)
        if not success:
            print(f"压力设置失败: {response}")
            return False
        
        # 等待压力稳定
        stabilize_time = 1.5 + random.uniform(-0.5, 0.5)
        time.sleep(stabilize_time)
        
        # 从主程序获取流量数据
        cont_flow = self.parent.cont_flow
        disp_flow = self.parent.disp_flow
        
        # 清空之前的数据，确保只收集新数据
        if hasattr(self.parent, 'current_diameters'):
            self.parent.current_diameters = []
        
        # 收集粒径数据 - 只收集1秒内的实时数据
        diameter_samples = []
        sample_count = 0
        start_time = time.time()
        sample_interval = 0.05  # 50ms采样间隔
        last_sample_time = 0
        
        # 严格1秒数据收集
        while time.time() - start_time < 1.0:
            current_time = time.time()
            
            # 获取当前帧的数据
            if hasattr(self.parent, 'current_diameters') and self.parent.current_diameters:
                # 检查数据是否是新的（通过时间戳判断）
                if hasattr(self.parent, 'last_diameter_update'):
                    if self.parent.last_diameter_update > last_sample_time:
                        # 这是新数据，记录它
                        diameter_samples.extend(self.parent.current_diameters)
                        sample_count += 1
                        last_sample_time = self.parent.last_diameter_update
            
            time.sleep(sample_interval)
        
        # 计算统计数据 - 如果没有数据，明确记录为0
        if diameter_samples and len(diameter_samples) > 0:
            avg_diameter = np.mean(diameter_samples)
            std_diameter = np.std(diameter_samples)
            droplet_count = len(diameter_samples)
        else:
            # 没有检测到液滴，记录为0
            avg_diameter = 0.0
            std_diameter = 0.0
            droplet_count = 0
            self.ui_queue.put(('log', f"警告: 在点 {self.current_point} 的1秒采集时间内未检测到液滴"))
        
        # 保存数据点
        self.data_points.append({
            'timestamp': time.time(),
            'cp_pressure': cp_pressure,
            'dp_pressure': dp_pressure,
            'cp_flow': cont_flow,
            'dp_flow': disp_flow,
            'droplet_size': avg_diameter,  # 如果没有液滴，这里会是0
            'droplet_std': std_diameter,
            'droplet_count': droplet_count,
            'sample_count': sample_count,
            'data_validity': 'valid' if droplet_count > 0 else 'no_droplet'  # 添加数据有效性标记
        })
        
        # 日志记录
        if avg_diameter > 0:
            self.ui_queue.put(('log', 
                f"数据点 {self.current_point}: "
                f"直径={avg_diameter:.1f}±{std_diameter:.1f}μm, "
                f"液滴数={droplet_count}, "
                f"采样次数={sample_count}"
            ))
        else:
            self.ui_queue.put(('log', 
                f"数据点 {self.current_point}: "
                f"未检测到液滴 (记录为0)"
            ))
        
        return True

    
    def training_thread(self):
        """训练数据收集线程"""
        try:
            if not self.parent.training_mode:
                self.parent.training_mode = True
            
            pressure_sequence = self.generate_pressure_sequence()
            
            self.ui_queue.put(('log', "等待系统初始化..."))
            time.sleep(2.0)
            
            for i, (cp, dp) in enumerate(pressure_sequence):
                if not self.training_active:
                    # 确保发送0压力命令
                    try:
                        self.pressure_controller.send_pressure_command(0, 0)
                    except:
                        pass
                    break
                    
                success = self.collect_data_point(cp, dp)
                
                self.current_point = i + 1
                progress = int((i + 1) / self.total_points * 100)
                
                self.ui_queue.put(('progress', progress, 
                    f"收集 {i+1}/{self.total_points} (CP:{cp:.0f}, DP:{dp:.0f})"
                ))
                
                if not success:
                    self.ui_queue.put(('log', f"点 {i+1} 数据收集失败"))
            
            # 训练结束（无论是完成还是被终止）
            self.training_active = False
            self.parent.training_mode = False
            
            # 确保发送0压力命令
            try:
                self.pressure_controller.send_pressure_command(0, 0)
            except:
                pass
            
            # 根据是否被中断决定完成类型
            if self.current_point < self.total_points:
                self.ui_queue.put(('terminated', None))  # 被终止
            else:
                self.ui_queue.put(('complete', None))  # 正常完成
            
        except Exception as e:
            self.ui_queue.put(('log', f"训练线程异常: {str(e)}"))
            self.training_active = False
            self.parent.training_mode = False
            # 确保发送0压力命令
            try:
                self.pressure_controller.send_pressure_command(0, 0)
            except:
                pass
            self.ui_queue.put(('terminated', None))  # 异常终止
    
    def save_training_data(self, is_complete=True):
        """保存训练数据（支持完整和部分数据）"""
        if len(self.data_points) > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            status = "complete" if is_complete else "partial"
            filename = f"training_data_{timestamp}_{status}.csv"
            
            # 统计有效数据点
            valid_points = sum(1 for point in self.data_points if point['droplet_size'] > 0)
            total_points = len(self.data_points)
            
            # 保存CSV文件
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = self.data_points[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data_points)
            
            status_text = "训练完成！" if is_complete else "训练已终止！"
            return (f"{status_text}\n"
                   f"总数据点: {total_points}\n"
                   f"有效数据点: {valid_points} ({valid_points/total_points*100:.1f}%)\n"
                   f"数据已保存到: {filename}")
        else:
            return "没有收集到任何数据"
    
    def start_training(self):
        """启动训练过程"""
        if not self.training_active:
            # 检查系统状态
            if self.parent.pid_enabled:
                messagebox.showwarning("警告", "请先停止PID控制再进入训练模式")
                return
            
            # 确保压力控制器已连接
            if not self.parent.pressure_controller.connected:
                if not self.parent.connect_to_pressure_server():
                    messagebox.showerror("错误", "无法连接到压力服务器")
                    return
            
            # 设置训练模式标志
            self.parent.training_mode = True
            
            self.training_active = True
            self.data_points = []
            self.current_point = 0
            
            # 创建进度窗口
            self.progress_window = TrainingProgressWindow(self.parent, self)
            
            # 启动训练线程
            threading.Thread(
                target=self.training_thread,
                daemon=True
            ).start()

class TrainingProgressWindow(tk.Toplevel):
    """训练进度显示窗口（支持中断保存）"""
    def __init__(self, parent, trainer):
        super().__init__(parent)
        self.title("模型训练进度")
        self.geometry("500x400")
        self.trainer = trainer
        
        # 进度变量
        self.progress_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="准备开始训练...")
        
        # 主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 进度条区域
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(progress_frame, text="收集进度:").pack(side=tk.LEFT)
        ttk.Progressbar(
            progress_frame, 
            orient="horizontal",
            length=350,
            mode="determinate",
            variable=self.progress_var
        ).pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        ttk.Label(main_frame, textvariable=self.status_var, wraplength=450).pack(anchor=tk.W)
        
        # 统计信息框架
        stats_frame = ttk.LabelFrame(main_frame, text="实时统计")
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="等待数据...")
        self.stats_label.pack(padx=5, pady=5)
        
        # 日志区域
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_frame, 
            height=10, 
            state="disabled",
            yscrollcommand=scrollbar.set
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="终止并保存",
            command=self.stop_and_save,
            state=tk.NORMAL
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加紧急停止按钮（不保存）
        self.abort_btn = ttk.Button(
            button_frame,
            text="紧急停止",
            command=self.emergency_stop,
            state=tk.NORMAL
        )
        self.abort_btn.pack(side=tk.LEFT, padx=5)
        
        self.close_btn = ttk.Button(
            button_frame,
            text="关闭",
            command=self.on_close,
            state=tk.DISABLED
        )
        self.close_btn.pack(side=tk.RIGHT, padx=5)
        
        # 启动UI更新定时器
        self.process_ui_queue()
        self.update_statistics()
        
        # 窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def update_statistics(self):
        """更新统计信息显示"""
        if self.trainer.data_points:
            total = len(self.trainer.data_points)
            valid = sum(1 for point in self.trainer.data_points if point['droplet_size'] > 0)
            percentage = (valid / total * 100) if total > 0 else 0
            
            self.stats_label.config(
                text=f"已收集: {total} 个数据点 | "
                     f"有效数据: {valid} 个 ({percentage:.1f}%)"
            )
        
        # 继续更新（如果训练仍在进行）
        if self.trainer.training_active:
            self.after(1000, self.update_statistics)
    
    def on_close(self):
        """窗口关闭处理"""
        if self.trainer.training_active:
            # 如果训练还在进行，提示用户
            result = messagebox.askyesnocancel(
                "确认关闭",
                "训练仍在进行中。\n"
                "是 - 保存已收集的数据并关闭\n"
                "否 - 不保存数据直接关闭\n"
                "取消 - 继续训练"
            )
            
            if result is True:  # 保存并关闭
                self.stop_and_save()
                self.after(1000, self.destroy)
            elif result is False:  # 不保存直接关闭
                self.emergency_stop()
                self.after(500, self.destroy)
            # result is None: 取消，继续训练
        else:
            self.destroy()
    
    def process_ui_queue(self):
        """处理UI更新队列"""
        try:
            while not self.trainer.ui_queue.empty():
                item = self.trainer.ui_queue.get_nowait()
                if item[0] == 'progress':
                    _, progress, message = item
                    self.update_progress(progress, message)
                elif item[0] == 'log':
                    _, message = item
                    self.log_message(message)
                elif item[0] == 'complete':
                    message = self.trainer.save_training_data(is_complete=True)
                    self.complete(message)
                elif item[0] == 'terminated':
                    message = self.trainer.save_training_data(is_complete=False)
                    self.terminated(message)
        except Exception as e:
            print(f"处理UI队列异常: {str(e)}")
        
        # 继续定时检查队列
        if self.trainer.training_active or not self.trainer.ui_queue.empty():
            self.after(50, self.process_ui_queue)
    
    def stop_and_save(self):
        """终止训练并保存数据"""
        if self.trainer.training_active:
            # 先停止训练
            self.trainer.training_active = False
            self.trainer.parent.training_mode = False
            
            # 确保压力控制器重置
            try:
                self.trainer.pressure_controller.send_pressure_command(0, 0)
            except:
                pass
            
            # 更新UI状态
            self.stop_btn.config(state=tk.DISABLED)
            self.abort_btn.config(state=tk.DISABLED)
            self.status_var.set("正在终止训练并保存数据...")
            self.log_message("用户请求终止训练（保存数据）")
            
            # 确保保存操作在主线程执行
            self.after(100, lambda: self._finalize_save())
            
    def _finalize_save(self):
        """最终保存操作"""
        try:
            message = self.trainer.save_training_data(is_complete=False)
            self.complete(message)
        except Exception as e:
            self.log_message(f"保存数据失败: {str(e)}")
            messagebox.showerror("错误", f"保存数据失败: {str(e)}")
        finally:
            self.close_btn.config(state=tk.NORMAL)

    
    def emergency_stop(self):
        """紧急停止（不保存）"""
        if self.trainer.training_active:
            # 弹出确认对话框
            if messagebox.askyesno("确认", "紧急停止将不保存任何数据，确定要继续吗？"):
                # 先停止训练
                self.trainer.training_active = False
                self.trainer.parent.training_mode = False
                
                # 确保压力控制器重置
                try:
                    self.trainer.pressure_controller.send_pressure_command(0, 0)
                except:
                    pass
                
                # 更新UI状态
                self.stop_btn.config(state=tk.DISABLED)
                self.abort_btn.config(state=tk.DISABLED)
                self.status_var.set("训练已紧急停止")
                self.log_message("训练已紧急停止（未保存数据）")
                self.close_btn.config(state=tk.NORMAL)
                
                # 强制释放资源
                self.trainer.data_points = []  # 清空数据
                self.after(100, self.destroy)  # 延迟关闭确保UI更新
    
    def update_progress(self, progress, message):
        """更新进度显示"""
        self.progress_var.set(progress)
        self.status_var.set(message)
    
    def log_message(self, message):
        """添加日志消息"""
        self.log_text.config(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
    
    def complete(self, message):
        """训练正常完成处理"""
        self.status_var.set("训练完成")
        self.log_message(message)
        self.stop_btn.config(state=tk.DISABLED)
        self.abort_btn.config(state=tk.DISABLED)
        self.close_btn.config(state=tk.NORMAL)
        messagebox.showinfo("完成", message)
    
    def terminated(self, message):
        """训练被终止处理"""
        self.status_var.set("训练已终止")
        self.log_message(message)
        self.stop_btn.config(state=tk.DISABLED)
        self.abort_btn.config(state=tk.DISABLED)
        self.close_btn.config(state=tk.NORMAL)
        messagebox.showinfo("已保存", message)
