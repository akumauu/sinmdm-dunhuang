"""
Physical Constraints for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 物理约束模块

实现生成动作的物理约束：
- 关节角度限位
- 足部滑动抑制
- 根节点稳定
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class PhysicalConstraints:
    """物理约束处理器"""
    
    # 人体关节默认角度限制（度）
    DEFAULT_JOINT_LIMITS = {
        # 上肢
        'shoulder': {'min': -180, 'max': 180},
        'elbow': {'min': 0, 'max': 160},
        'wrist': {'min': -90, 'max': 90},
        
        # 下肢
        'hip': {'min': -120, 'max': 120},
        'knee': {'min': 0, 'max': 160},
        'ankle': {'min': -45, 'max': 45},
        
        # 躯干
        'spine': {'min': -45, 'max': 45},
        'neck': {'min': -60, 'max': 60},
        
        # 默认
        'default': {'min': -180, 'max': 180}
    }
    
    def __init__(
        self,
        joint_limits: Optional[Dict] = None,
        foot_slide_threshold: float = 0.05,  # 足部滑动阈值
        ground_level: float = 0.0,  # 地面高度
    ):
        """
        初始化物理约束处理器
        
        Args:
            joint_limits: 自定义关节角度限制
            foot_slide_threshold: 足部滑动检测阈值
            ground_level: 地面高度
        """
        self.joint_limits = joint_limits or self.DEFAULT_JOINT_LIMITS
        self.foot_slide_threshold = foot_slide_threshold
        self.ground_level = ground_level
    
    def apply_joint_limits(
        self, 
        rotations: np.ndarray,
        joint_names: List[str],
        soft: bool = True
    ) -> np.ndarray:
        """
        应用关节角度限位
        
        Args:
            rotations: 旋转数据 (frames, joints, 3)
            joint_names: 关节名称列表
            soft: 是否使用软限制（平滑过渡）
            
        Returns:
            限位后的旋转数据
        """
        result = rotations.copy()
        
        for j, name in enumerate(joint_names):
            # 查找匹配的限制
            limits = self._get_limits_for_joint(name)
            
            for axis in range(3):
                if soft:
                    # 软限制：使用 tanh 平滑
                    result[:, j, axis] = self._soft_clamp(
                        result[:, j, axis], 
                        limits['min'], 
                        limits['max']
                    )
                else:
                    # 硬限制：直接裁剪
                    result[:, j, axis] = np.clip(
                        result[:, j, axis],
                        limits['min'],
                        limits['max']
                    )
        
        return result.astype(rotations.dtype)
    
    def _get_limits_for_joint(self, joint_name: str) -> Dict:
        """根据关节名称获取限制"""
        name_lower = joint_name.lower()
        
        for key in self.joint_limits:
            if key in name_lower:
                return self.joint_limits[key]
        
        return self.joint_limits.get('default', {'min': -180, 'max': 180})
    
    def _soft_clamp(self, values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """软限制：在边界附近平滑过渡"""
        range_val = max_val - min_val
        mid = (min_val + max_val) / 2
        
        # 归一化到 [-1, 1]
        normalized = (values - mid) / (range_val / 2)
        
        # 应用 tanh 压缩
        compressed = np.tanh(normalized * 0.8) * 0.95
        
        # 反归一化
        return compressed * (range_val / 2) + mid
    
    def suppress_foot_sliding(
        self,
        positions: np.ndarray,
        foot_positions: np.ndarray,
        foot_contact: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        抑制足部滑动
        
        Args:
            positions: 根节点位置 (frames, 3)
            foot_positions: 足部位置 (frames, 2, 3) 左右脚
            foot_contact: 足部接触标签 (frames, 2) 可选
            
        Returns:
            修正后的根节点位置
        """
        result = positions.copy()
        num_frames = len(positions)
        
        if foot_contact is None:
            # 自动检测接触：足部接近地面时
            foot_contact = foot_positions[:, :, 1] < (self.ground_level + self.foot_slide_threshold * 2)
        
        for i in range(1, num_frames):
            for foot_idx in range(2):
                if foot_contact[i, foot_idx] and foot_contact[i-1, foot_idx]:
                    # 如果脚在接触地面，计算滑动量
                    slide = foot_positions[i, foot_idx, [0, 2]] - foot_positions[i-1, foot_idx, [0, 2]]
                    slide_dist = np.linalg.norm(slide)
                    
                    if slide_dist > self.foot_slide_threshold:
                        # 调整根节点位置以抵消滑动
                        correction = slide * (slide_dist - self.foot_slide_threshold) / slide_dist
                        result[i, [0, 2]] -= correction * 0.5  # 部分补偿
        
        return result.astype(positions.dtype)
    
    def stabilize_root(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        smooth_window: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        稳定根节点运动
        
        Args:
            positions: 根节点位置 (frames, 3)
            rotations: 旋转数据 (frames, joints, 3)
            smooth_window: 平滑窗口
            
        Returns:
            稳定后的位置和旋转
        """
        from scipy.signal import savgol_filter
        
        stable_positions = positions.copy()
        stable_rotations = rotations.copy()
        
        # 平滑根节点 XZ 平面移动
        if len(positions) > smooth_window:
            window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            stable_positions[:, 0] = savgol_filter(positions[:, 0], window, 2)
            stable_positions[:, 2] = savgol_filter(positions[:, 2], window, 2)
        
        # 平滑根节点旋转（尤其是 Y 轴朝向）
        if len(rotations) > smooth_window:
            window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            for axis in range(3):
                stable_rotations[:, 0, axis] = savgol_filter(rotations[:, 0, axis], window, 2)
        
        return stable_positions.astype(positions.dtype), stable_rotations.astype(rotations.dtype)
    
    def enforce_ground_contact(
        self,
        positions: np.ndarray,
        min_height: Optional[float] = None
    ) -> np.ndarray:
        """
        确保不穿透地面
        
        Args:
            positions: 位置数据
            min_height: 最小高度
            
        Returns:
            修正后的位置
        """
        result = positions.copy()
        
        if min_height is None:
            min_height = self.ground_level
        
        # 确保 Y 轴（高度）不低于地面
        result[:, 1] = np.maximum(result[:, 1], min_height)
        
        return result.astype(positions.dtype)
    
    def apply_all(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        joint_names: List[str],
        foot_joint_indices: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用所有物理约束
        
        Args:
            positions: 根节点位置
            rotations: 旋转数据
            joint_names: 关节名称
            foot_joint_indices: 足部关节索引 (左脚, 右脚)
            
        Returns:
            处理后的 (位置, 旋转)
        """
        # 1. 关节角度限位
        rotations = self.apply_joint_limits(rotations, joint_names)
        
        # 2. 稳定根节点
        positions, rotations = self.stabilize_root(positions, rotations)
        
        # 3. 确保不穿透地面
        positions = self.enforce_ground_contact(positions)
        
        return positions, rotations
