import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
import os
import time
import threading
from collections import deque

class FlowPredictor:
    def __init__(self, config_path="config.yaml"):
        """初始化流量预测器，从配置文件加载模型路径"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f).get('model_paths', {})
        
        model_dir = self.config.get('model_dir', 'rf_models')
        
        try:
            self.models = {
                'model1': joblib.load(os.path.join(model_dir, self.config.get('model_1'))),
                'model3': joblib.load(os.path.join(model_dir, self.config.get('model_3'))),
                'model4': joblib.load(os.path.join(model_dir, self.config.get('model_4')))
            }
            
            self.scalers = {
                'model1': joblib.load(os.path.join(model_dir, self.config.get('scaler_1'))),
                'model3': joblib.load(os.path.join(model_dir, self.config.get('scaler_3'))),
                'model4': joblib.load(os.path.join(model_dir, self.config.get('scaler_4')))
            }
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")
        
        self.last_predictions = {
            'model1': 0.0,
            'model3': 0.0,
            'model4': 0.0
        }
        
        # 添加数据缓存（假设数据采集频率约为30Hz，缓存1.5秒数据）
        self.data_buffer = {
            'cp_pressure': deque(maxlen=50),
            'dp_pressure': deque(maxlen=50),
            'droplet_size': deque(maxlen=50),
            'timestamps': deque(maxlen=50)
        }
        
        # 预测控制
        self.prediction_interval = 1.0  # 1秒预测一次
        self.last_prediction_time = 0
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()
        
    def add_data_point(self, cp_pressure, dp_pressure, droplet_size):
        """添加新的数据点到缓冲区"""
        current_time = time.time()
        with self.prediction_lock:
            self.data_buffer['cp_pressure'].append(cp_pressure)
            self.data_buffer['dp_pressure'].append(dp_pressure)
            self.data_buffer['droplet_size'].append(droplet_size)
            self.data_buffer['timestamps'].append(current_time)
    
    def get_previous_second_average(self):
        """获取前一秒钟的平均值"""
        current_time = time.time()
        
        with self.prediction_lock:
            if len(self.data_buffer['timestamps']) == 0:
                return None, None, None
            
            # 计算前一秒的时间范围
            start_time = current_time - 2.0  # 前一秒的开始
            end_time = current_time - 1.0    # 前一秒的结束
            
            # 筛选前一秒的数据
            prev_second_indices = []
            for i, timestamp in enumerate(self.data_buffer['timestamps']):
                if start_time <= timestamp <= end_time:
                    prev_second_indices.append(i)
            
            if not prev_second_indices:
                return None, None, None
            
            # 计算前一秒数据的平均值
            cp_values = [self.data_buffer['cp_pressure'][i] for i in prev_second_indices]
            dp_values = [self.data_buffer['dp_pressure'][i] for i in prev_second_indices]
            size_values = [self.data_buffer['droplet_size'][i] for i in prev_second_indices]
            
            return np.mean(cp_values), np.mean(dp_values), np.mean(size_values)
    
    def should_predict(self):
        """判断是否应该进行预测（1秒间隔控制）"""
        current_time = time.time()
        return (current_time - self.last_prediction_time) >= self.prediction_interval
    
    def predict_with_timing(self):
        """带时间控制的预测方法"""
        if not self.should_predict():
            return self.latest_prediction
        
        # 获取前一秒的平均值
        cp_pressure, dp_pressure, droplet_size = self.get_previous_second_average()
        
        if cp_pressure is None:
            return self.latest_prediction
        
        # 执行预测
        prediction = self.predict(cp_pressure, dp_pressure, droplet_size)
        
        # 更新预测时间和结果
        self.last_prediction_time = time.time()
        self.latest_prediction = prediction
        
        return prediction


    def get_prediction_details(self):
        """获取各模型的预测详情"""
        return {
            'model1': self.last_predictions.get('model1', 0.0),
            'model3': self.last_predictions.get('model3', 0.0), 
            'model4': self.last_predictions.get('model4', 0.0)
        }




    def predict(self, cp_pressure, dp_pressure, droplet_size):
        """使用三个模型进行预测"""
        try:
            # 模型1: 仅使用分散相压力
            X1 = np.array([[dp_pressure]])
            X1_scaled = self.scalers['model1'].transform(X1)
            pred1 = self.models['model1'].predict(X1_scaled)[0]
            self.last_predictions['model1'] = pred1
            
            # 模型3: 使用多特征
            features = [
                cp_pressure, dp_pressure, droplet_size,
                dp_pressure - cp_pressure, dp_pressure / (cp_pressure + 1e-6),
                cp_pressure + dp_pressure, cp_pressure * dp_pressure,
                dp_pressure / (cp_pressure + 1e-6)
            ]
            X3 = np.array([features])
            X3_scaled = self.scalers['model3'].transform(X3)
            pred3 = self.models['model3'].predict(X3_scaled)[0]
            self.last_predictions['model3'] = pred3
            
            # 模型4: 使用连续相压力和液滴粒径
            X4 = np.array([[cp_pressure, droplet_size]])
            X4_scaled = self.scalers['model4'].transform(X4)
            pred4 = self.models['model4'].predict(X4_scaled)[0]
            self.last_predictions['model4'] = pred4
            
            # 检查预测一致性
            deviations = {
                'model1': abs(pred1 - pred3) / pred3 if pred3 != 0 else 0,
                'model4': abs(pred4 - pred3) / pred3 if pred3 != 0 else 0
            }
            
            is_consistent = all(dev < 0.6 for dev in deviations.values())

            avg_prediction = (pred1 + pred3 + pred4) / 3 #取平均值

            return {
                'prediction': avg_prediction,  # 使用模型3作为主要预测
                'is_consistent': is_consistent,
                'details': {
                    'model1': pred1,
                    'model3': pred3,
                    'model4': pred4
                }
            }
            
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return {
                'prediction': 0.0,
                'is_consistent': False,
                'details': {
                    'model1': 0.0,
                    'model3': 0.0,
                    'model4': 0.0
                }
            }
