"""
Data Validator for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 数据验证器

验证姿态序列数据的有效性和质量
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .bvh_parser import BVHData


@dataclass
class ValidationResult:
    """数据验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: dict
    
    def __str__(self):
        status = "✓ Valid" if self.is_valid else "✗ Invalid"
        msg = [f"Validation: {status}"]
        if self.errors:
            msg.append("Errors:")
            msg.extend([f"  - {e}" for e in self.errors])
        if self.warnings:
            msg.append("Warnings:")
            msg.extend([f"  - {w}" for w in self.warnings])
        return "\n".join(msg)


class DataValidator:
    """数据验证器"""
    
    def __init__(
        self,
        min_frames: int = 30,
        max_velocity: float = 50.0,  # 最大允许速度（单位/帧）
        min_duration: float = 1.0,   # 最小时长（秒）
        max_rotation_change: float = 180.0,  # 单帧最大旋转变化（度）
    ):
        self.min_frames = min_frames
        self.max_velocity = max_velocity
        self.min_duration = min_duration
        self.max_rotation_change = max_rotation_change
    
    def validate(self, data: BVHData) -> ValidationResult:
        """
        验证 BVH 数据
        
        Args:
            data: 要验证的 BVH 数据
            
        Returns:
            ValidationResult: 验证结果
        """
        errors = []
        warnings = []
        stats = {}
        
        # 基本信息检查
        stats['num_frames'] = data.num_frames
        stats['num_joints'] = data.num_joints
        stats['fps'] = data.fps
        stats['duration'] = data.duration
        
        # 1. 帧数检查
        if data.num_frames < self.min_frames:
            errors.append(f"Too few frames: {data.num_frames} (min: {self.min_frames})")
        
        # 2. 时长检查
        if data.duration < self.min_duration:
            errors.append(f"Duration too short: {data.duration:.2f}s (min: {self.min_duration}s)")
        
        # 3. 数据形状检查
        if data.positions is None or len(data.positions) == 0:
            errors.append("No position data")
        
        if data.rotations is None or len(data.rotations) == 0:
            errors.append("No rotation data")
        
        # 4. NaN 检查
        if data.positions is not None and np.any(np.isnan(data.positions)):
            errors.append("NaN values found in positions")
            stats['nan_position_frames'] = np.sum(np.any(np.isnan(data.positions), axis=1))
        
        if data.rotations is not None and np.any(np.isnan(data.rotations)):
            errors.append("NaN values found in rotations")
            stats['nan_rotation_frames'] = np.sum(np.any(np.isnan(data.rotations), axis=(1, 2)))
        
        # 5. Inf 检查
        if data.positions is not None and np.any(np.isinf(data.positions)):
            errors.append("Infinite values found in positions")
        
        if data.rotations is not None and np.any(np.isinf(data.rotations)):
            errors.append("Infinite values found in rotations")
        
        # 6. 速度检查（检测抖动）
        if data.positions is not None and len(data.positions) > 1:
            velocity = np.diff(data.positions, axis=0)
            max_vel = np.max(np.abs(velocity))
            stats['max_velocity'] = float(max_vel)
            
            if max_vel > self.max_velocity:
                warnings.append(f"High velocity detected: {max_vel:.2f} (threshold: {self.max_velocity})")
        
        # 7. 旋转变化检查
        if data.rotations is not None and len(data.rotations) > 1:
            rot_diff = np.abs(np.diff(data.rotations, axis=0))
            max_rot_change = np.max(rot_diff)
            stats['max_rotation_change'] = float(max_rot_change)
            
            if max_rot_change > self.max_rotation_change:
                warnings.append(f"Large rotation change detected: {max_rot_change:.2f}° (threshold: {self.max_rotation_change}°)")
        
        # 8. 骨架完整性检查
        if len(data.joint_names) != data.num_joints:
            errors.append(f"Joint name count mismatch: {len(data.joint_names)} vs {data.num_joints}")
        
        if data.offsets is not None and len(data.offsets) != data.num_joints:
            errors.append(f"Offset count mismatch: {len(data.offsets)} vs {data.num_joints}")
        
        # 9. 父节点有效性检查
        if data.parent_indices is not None:
            for i, parent in enumerate(data.parent_indices):
                if i > 0 and (parent < 0 or parent >= i):
                    if parent != -1:  # -1 是根节点
                        errors.append(f"Invalid parent index for joint {i}: {parent}")
        
        # 统计信息
        if data.positions is not None:
            stats['position_range'] = {
                'min': data.positions.min(axis=0).tolist(),
                'max': data.positions.max(axis=0).tolist()
            }
        
        if data.rotations is not None:
            stats['rotation_range'] = {
                'min': data.rotations.min(),
                'max': data.rotations.max()
            }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def check_sinmdm_compatibility(self, data: BVHData) -> ValidationResult:
        """
        检查数据与 SinMDM 的兼容性
        
        Args:
            data: BVH 数据
            
        Returns:
            ValidationResult: 兼容性检查结果
        """
        errors = []
        warnings = []
        stats = {'compatible': True}
        
        # SinMDM 推荐的帧率
        if abs(data.fps - 30.0) > 1.0 and abs(data.fps - 60.0) > 1.0:
            warnings.append(f"FPS {data.fps:.1f} may not be optimal. SinMDM works best with 30 or 60 FPS")
        
        # 序列长度检查
        if data.num_frames < 64:
            warnings.append(f"Short sequence ({data.num_frames} frames). SinMDM performs better with longer sequences")
        
        # 关节数量检查
        if data.num_joints < 10:
            warnings.append(f"Few joints ({data.num_joints}). Complex skeletons may produce better results")
        
        stats['compatible'] = len(errors) == 0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def compute_quality_score(self, data: BVHData) -> float:
        """
        计算数据质量分数 (0-100)
        
        Args:
            data: BVH 数据
            
        Returns:
            float: 质量分数
        """
        score = 100.0
        
        # 帧数惩罚
        if data.num_frames < 100:
            score -= (100 - data.num_frames) * 0.1
        
        # NaN 惩罚
        if data.positions is not None:
            nan_ratio = np.sum(np.isnan(data.positions)) / data.positions.size
            score -= nan_ratio * 50
        
        if data.rotations is not None:
            nan_ratio = np.sum(np.isnan(data.rotations)) / data.rotations.size
            score -= nan_ratio * 50
        
        # 抖动惩罚
        if data.positions is not None and len(data.positions) > 1:
            velocity = np.diff(data.positions, axis=0)
            jitter = np.std(velocity)
            if jitter > 1.0:
                score -= min(jitter * 2, 20)
        
        return max(0.0, min(100.0, score))
