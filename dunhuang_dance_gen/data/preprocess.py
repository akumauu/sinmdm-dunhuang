"""
Preprocess module for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 预处理模块

实现姿态序列的预处理流程：
- 帧率标准化
- 滤波平滑
- 异常帧检测与插值
- 坐标系/尺度标准化
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Optional, Tuple
from .bvh_parser import BVHData


class DunhuangPreprocessor:
    """敦煌舞蹈动作预处理器"""
    
    def __init__(
        self,
        target_fps: float = 30.0,
        smooth_window: int = 5,
        smooth_poly_order: int = 2,
        outlier_threshold: float = 3.0,
        target_height: float = 1.7,  # 目标身高（米）
    ):
        """
        初始化预处理器
        
        Args:
            target_fps: 目标帧率
            smooth_window: Savitzky-Golay 滤波窗口大小
            smooth_poly_order: 多项式阶数
            outlier_threshold: 异常值检测阈值（标准差倍数）
            target_height: 目标标准化身高
        """
        self.target_fps = target_fps
        self.smooth_window = smooth_window
        self.smooth_poly_order = smooth_poly_order
        self.outlier_threshold = outlier_threshold
        self.target_height = target_height
    
    def process(self, data: BVHData, 
                resample: bool = True,
                smooth: bool = True,
                fix_outliers: bool = True,
                normalize_scale: bool = True) -> BVHData:
        """
        执行完整预处理流程
        
        Args:
            data: 输入的 BVH 数据
            resample: 是否重采样到目标帧率
            smooth: 是否应用平滑滤波
            fix_outliers: 是否修复异常帧
            normalize_scale: 是否标准化尺度
            
        Returns:
            处理后的 BVHData
        """
        # 复制数据避免修改原始对象
        positions = data.positions.copy()
        rotations = data.rotations.copy()
        offsets = data.offsets.copy()
        
        # 1. 帧率重采样
        if resample and abs(data.fps - self.target_fps) > 0.1:
            positions, rotations = self._resample(
                positions, rotations, 
                data.fps, self.target_fps
            )
            frame_time = 1.0 / self.target_fps
            num_frames = len(positions)
        else:
            frame_time = data.frame_time
            num_frames = data.num_frames
        
        # 2. 异常帧检测与修复
        if fix_outliers:
            positions, rotations = self._fix_outliers(positions, rotations)
        
        # 3. 平滑滤波
        if smooth:
            positions = self._smooth_sequence(positions)
            rotations = self._smooth_sequence(rotations)
        
        # 4. 尺度标准化
        scale_factor = 1.0
        if normalize_scale:
            offsets, positions, scale_factor = self._normalize_scale(offsets, positions)
        
        # 构建新的数据对象
        return BVHData(
            joint_names=data.joint_names.copy(),
            parent_indices=data.parent_indices.copy(),
            offsets=offsets,
            positions=positions,
            rotations=rotations,
            frame_time=frame_time,
            num_frames=num_frames,
            num_joints=data.num_joints
        )
    
    def _resample(self, positions: np.ndarray, rotations: np.ndarray,
                  source_fps: float, target_fps: float) -> Tuple[np.ndarray, np.ndarray]:
        """重采样到目标帧率"""
        source_frames = len(positions)
        target_frames = int(source_frames * target_fps / source_fps)
        
        # 时间轴
        source_times = np.linspace(0, 1, source_frames)
        target_times = np.linspace(0, 1, target_frames)
        
        # 位置插值
        pos_interp = interp1d(source_times, positions, axis=0, kind='cubic')
        new_positions = pos_interp(target_times).astype(np.float32)
        
        # 旋转插值（对每个关节的每个轴）
        new_rotations = np.zeros((target_frames, rotations.shape[1], 3), dtype=np.float32)
        for j in range(rotations.shape[1]):
            for axis in range(3):
                rot_interp = interp1d(source_times, rotations[:, j, axis], kind='cubic')
                new_rotations[:, j, axis] = rot_interp(target_times)
        
        return new_positions, new_rotations
    
    def _fix_outliers(self, positions: np.ndarray, rotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测并修复异常帧"""
        # 计算帧间速度
        pos_velocity = np.diff(positions, axis=0)
        
        # 检测位置异常
        pos_mean = np.mean(pos_velocity, axis=0)
        pos_std = np.std(pos_velocity, axis=0)
        
        for i in range(len(pos_velocity)):
            # 检查每个轴
            for axis in range(3):
                if pos_std[axis] > 0 and abs(pos_velocity[i, axis] - pos_mean[axis]) > self.outlier_threshold * pos_std[axis]:
                    # 使用线性插值修复
                    if i > 0 and i < len(positions) - 2:
                        positions[i + 1, axis] = (positions[i, axis] + positions[i + 2, axis]) / 2
        
        # 类似地处理旋转
        rot_velocity = np.diff(rotations, axis=0)
        for j in range(rotations.shape[1]):
            for axis in range(3):
                vel = rot_velocity[:, j, axis]
                mean_vel = np.mean(vel)
                std_vel = np.std(vel)
                
                if std_vel > 0:
                    for i in range(len(vel)):
                        if abs(vel[i] - mean_vel) > self.outlier_threshold * std_vel:
                            if i > 0 and i < len(rotations) - 2:
                                rotations[i + 1, j, axis] = (rotations[i, j, axis] + rotations[i + 2, j, axis]) / 2
        
        return positions, rotations
    
    def _smooth_sequence(self, data: np.ndarray) -> np.ndarray:
        """应用 Savitzky-Golay 滤波平滑"""
        if len(data) < self.smooth_window:
            return data
        
        # 确保窗口大小为奇数
        window = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
        
        if data.ndim == 2:
            # 2D 数据 (frames, features)
            smoothed = np.zeros_like(data)
            for i in range(data.shape[1]):
                smoothed[:, i] = savgol_filter(data[:, i], window, self.smooth_poly_order)
            return smoothed.astype(np.float32)
        elif data.ndim == 3:
            # 3D 数据 (frames, joints, axes)
            smoothed = np.zeros_like(data)
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    smoothed[:, j, k] = savgol_filter(data[:, j, k], window, self.smooth_poly_order)
            return smoothed.astype(np.float32)
        
        return data
    
    def _normalize_scale(self, offsets: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """标准化骨架尺度"""
        # 估算当前身高（使用 offset 的 Y 轴累加）
        # 假设 offset[1] 通常是髋关节到脊柱的偏移
        total_height = np.sum(np.abs(offsets[:, 1]))
        
        if total_height > 0.01:  # 避免除零
            scale_factor = self.target_height / total_height
        else:
            scale_factor = 1.0
        
        # 应用缩放
        scaled_offsets = offsets * scale_factor
        scaled_positions = positions * scale_factor
        
        return scaled_offsets.astype(np.float32), scaled_positions.astype(np.float32), scale_factor
    
    def center_root(self, positions: np.ndarray) -> np.ndarray:
        """将根节点起始位置移到原点"""
        offset = positions[0].copy()
        offset[1] = 0  # 保持高度不变
        return (positions - offset).astype(np.float32)
    
    def align_facing(self, positions: np.ndarray, rotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对齐朝向（使初始帧面向 Z+ 方向）"""
        # 简化实现：这里只返回原数据
        # 完整实现需要计算初始朝向并旋转整个序列
        return positions, rotations
