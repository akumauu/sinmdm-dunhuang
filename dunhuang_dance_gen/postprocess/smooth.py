"""
Motion Smoother for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 动作平滑模块

实现生成动作的后处理平滑：
- 高斯滤波
- Savitzky-Golay 滤波
- 速度突变修正
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, Literal


class MotionSmoother:
    """动作后处理平滑器"""
    
    def __init__(
        self,
        method: Literal['savgol', 'gaussian'] = 'savgol',
        window_size: int = 5,
        poly_order: int = 2,
        sigma: float = 1.0,
    ):
        """
        初始化平滑器
        
        Args:
            method: 平滑方法 ('savgol' 或 'gaussian')
            window_size: Savitzky-Golay 窗口大小
            poly_order: Savitzky-Golay 多项式阶数
            sigma: 高斯滤波标准差
        """
        self.method = method
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.poly_order = poly_order
        self.sigma = sigma
    
    def smooth(self, data: np.ndarray) -> np.ndarray:
        """
        平滑动作数据
        
        Args:
            data: 输入数据 (frames, ...) 
                  可以是位置 (frames, 3) 或旋转 (frames, joints, 3)
                  
        Returns:
            平滑后的数据
        """
        if len(data) < self.window_size:
            return data
        
        if self.method == 'savgol':
            return self._savgol_smooth(data)
        elif self.method == 'gaussian':
            return self._gaussian_smooth(data)
        else:
            return data
    
    def _savgol_smooth(self, data: np.ndarray) -> np.ndarray:
        """Savitzky-Golay 滤波"""
        original_shape = data.shape
        
        if data.ndim == 2:
            # (frames, features)
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                smoothed[:, i] = savgol_filter(data[:, i], self.window_size, self.poly_order)
        elif data.ndim == 3:
            # (frames, joints, axes)
            smoothed = np.zeros_like(data)
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    smoothed[:, j, k] = savgol_filter(data[:, j, k], self.window_size, self.poly_order)
        else:
            # 1D
            smoothed = savgol_filter(data, self.window_size, self.poly_order)
        
        return smoothed.astype(data.dtype)
    
    def _gaussian_smooth(self, data: np.ndarray) -> np.ndarray:
        """高斯滤波"""
        if data.ndim == 2:
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                smoothed[:, i] = gaussian_filter1d(data[:, i], self.sigma)
        elif data.ndim == 3:
            smoothed = np.zeros_like(data)
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    smoothed[:, j, k] = gaussian_filter1d(data[:, j, k], self.sigma)
        else:
            smoothed = gaussian_filter1d(data, self.sigma)
        
        return smoothed.astype(data.dtype)
    
    def fix_velocity_spikes(
        self, 
        data: np.ndarray, 
        threshold_factor: float = 3.0
    ) -> np.ndarray:
        """
        修复速度突变
        
        Args:
            data: 输入数据 (frames, ...)
            threshold_factor: 阈值因子（标准差的倍数）
            
        Returns:
            修复后的数据
        """
        result = data.copy()
        
        # 计算帧间差分（速度）
        velocity = np.diff(data, axis=0)
        
        # 对每个维度处理
        if data.ndim == 2:
            for i in range(data.shape[1]):
                result[:, i] = self._fix_1d_spikes(data[:, i], threshold_factor)
        elif data.ndim == 3:
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    result[:, j, k] = self._fix_1d_spikes(data[:, j, k], threshold_factor)
        else:
            result = self._fix_1d_spikes(data, threshold_factor)
        
        return result.astype(data.dtype)
    
    def _fix_1d_spikes(self, data: np.ndarray, threshold_factor: float) -> np.ndarray:
        """修复一维序列中的突变"""
        result = data.copy()
        velocity = np.diff(data)
        
        mean_vel = np.mean(velocity)
        std_vel = np.std(velocity)
        
        if std_vel < 1e-6:
            return result
        
        threshold = threshold_factor * std_vel
        
        for i in range(len(velocity)):
            if abs(velocity[i] - mean_vel) > threshold:
                # 使用线性插值修复
                if i > 0 and i < len(data) - 2:
                    result[i + 1] = (result[i] + result[i + 2]) / 2
        
        return result
    
    def adaptive_smooth(
        self, 
        data: np.ndarray, 
        activity_threshold: float = 0.5
    ) -> np.ndarray:
        """
        自适应平滑：静止区域多平滑，运动区域少平滑
        
        Args:
            data: 输入数据
            activity_threshold: 活动阈值
            
        Returns:
            自适应平滑后的数据
        """
        # 计算活动度（速度的绝对值）
        velocity = np.abs(np.diff(data, axis=0, prepend=data[:1]))
        
        if data.ndim > 1:
            activity = np.mean(velocity, axis=tuple(range(1, data.ndim)))
        else:
            activity = velocity
        
        # 归一化活动度
        activity_norm = activity / (np.max(activity) + 1e-6)
        
        # 二次平滑：低活动区域用更强的平滑
        light_smooth = self.smooth(data)
        
        old_window = self.window_size
        self.window_size = min(self.window_size * 2 + 1, len(data) // 2)
        if self.window_size % 2 == 0:
            self.window_size += 1
        heavy_smooth = self.smooth(data)
        self.window_size = old_window
        
        # 混合
        result = np.zeros_like(data)
        for i in range(len(data)):
            weight = min(activity_norm[i] / activity_threshold, 1.0)
            result[i] = weight * light_smooth[i] + (1 - weight) * heavy_smooth[i]
        
        return result.astype(data.dtype)
