"""
Post-processing Pipeline for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 后处理集成管线

将平滑、约束、导出串联为完整的后处理流程，
支持参数配置和处理前后对比。
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .smooth import MotionSmoother
from .constraints import PhysicalConstraints


@dataclass
class PostProcessConfig:
    """后处理参数配置"""
    # 平滑参数
    smooth_method: str = 'savgol'       # 'savgol' | 'gaussian' | 'none'
    smooth_window: int = 5              # Savitzky-Golay 窗口大小
    smooth_poly_order: int = 2          # 多项式阶数
    smooth_sigma: float = 1.0           # 高斯标准差
    
    # 速度修正
    fix_velocity_spikes: bool = True    # 是否修正速度突变
    spike_threshold: float = 3.0        # 突变检测阈值（标准差倍数）
    
    # 物理约束
    apply_joint_limits: bool = True     # 是否应用关节限位
    soft_limits: bool = True            # 使用软限制（平滑过渡）
    stabilize_root: bool = True         # 是否稳定根节点
    root_smooth_window: int = 5         # 根节点平滑窗口
    enforce_ground: bool = True         # 是否确保不穿透地面
    ground_level: float = 0.0           # 地面高度


@dataclass
class PostProcessResult:
    """后处理结果"""
    positions: np.ndarray               # 处理后的位置
    rotations: np.ndarray               # 处理后的旋转
    config: PostProcessConfig           # 所用配置
    
    # 对比统计 (处理前后)
    stats_before: Dict[str, float] = field(default_factory=dict)
    stats_after: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """生成处理摘要"""
        lines = ["=== 后处理摘要 ==="]
        lines.append(f"平滑方法: {self.config.smooth_method}")
        lines.append(f"速度修正: {'开启' if self.config.fix_velocity_spikes else '关闭'}")
        lines.append(f"关节限位: {'开启' if self.config.apply_joint_limits else '关闭'}")
        lines.append(f"根节点稳定: {'开启' if self.config.stabilize_root else '关闭'}")
        
        if self.stats_before and self.stats_after:
            lines.append("\n--- 处理前后对比 ---")
            for key in self.stats_before:
                before = self.stats_before[key]
                after = self.stats_after.get(key, 0)
                change = ((after - before) / (before + 1e-8)) * 100
                lines.append(f"  {key}: {before:.4f} → {after:.4f} ({change:+.1f}%)")
        
        return "\n".join(lines)


class PostProcessPipeline:
    """后处理集成管线"""
    
    def __init__(self, config: Optional[PostProcessConfig] = None):
        """
        初始化管线
        
        Args:
            config: 后处理配置，为 None 时使用默认配置
        """
        self.config = config or PostProcessConfig()
    
    def process(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        joint_names: list,
        foot_joint_indices: Optional[Tuple[int, int]] = None
    ) -> PostProcessResult:
        """
        执行完整后处理管线
        
        Args:
            positions: 根节点位置 (frames, 3)
            rotations: 关节旋转 (frames, joints, 3)
            joint_names: 关节名称列表
            foot_joint_indices: 足部关节索引 (左脚, 右脚)
            
        Returns:
            PostProcessResult: 处理结果
        """
        cfg = self.config
        
        # 计算处理前统计
        stats_before = self._compute_stats(positions, rotations, joint_names)
        
        # 复制数据
        pos = positions.copy()
        rot = rotations.copy()
        
        # Step 1: 速度突变修正 (在平滑之前)
        if cfg.fix_velocity_spikes:
            smoother = MotionSmoother(method=cfg.smooth_method)
            rot_flat = rot.reshape(rot.shape[0], -1)  # (frames, joints*3)
            rot_flat = smoother.fix_velocity_spikes(rot_flat, cfg.spike_threshold)
            rot = rot_flat.reshape(rot.shape)
            pos = smoother.fix_velocity_spikes(pos, cfg.spike_threshold)
        
        # Step 2: 平滑滤波
        if cfg.smooth_method != 'none':
            smoother = MotionSmoother(
                method=cfg.smooth_method,
                window_size=cfg.smooth_window,
                poly_order=cfg.smooth_poly_order,
                sigma=cfg.smooth_sigma
            )
            rot = smoother.smooth(rot)
            pos = smoother.smooth(pos)
        
        # Step 3: 物理约束
        constraints = PhysicalConstraints(ground_level=cfg.ground_level)
        
        if cfg.apply_joint_limits:
            rot = constraints.apply_joint_limits(rot, joint_names, soft=cfg.soft_limits)
        
        if cfg.stabilize_root:
            pos, rot = constraints.stabilize_root(pos, rot, smooth_window=cfg.root_smooth_window)
        
        if cfg.enforce_ground:
            pos = constraints.enforce_ground_contact(pos)
        
        # 计算处理后统计
        stats_after = self._compute_stats(pos, rot, joint_names)
        
        return PostProcessResult(
            positions=pos,
            rotations=rot,
            config=cfg,
            stats_before=stats_before,
            stats_after=stats_after
        )
    
    def _compute_stats(
        self, 
        positions: np.ndarray, 
        rotations: np.ndarray, 
        joint_names: list
    ) -> Dict[str, float]:
        """计算动作质量统计指标"""
        stats = {}
        
        # 1. 帧间角速度均值（平滑度指标，越低越平滑）
        rot_velocity = np.diff(rotations, axis=0)
        stats['角速度均值(°/帧)'] = float(np.mean(np.abs(rot_velocity)))
        stats['角速度最大值(°/帧)'] = float(np.max(np.abs(rot_velocity)))
        
        # 2. 关节越界率
        out_of_range = 0
        total = rotations.shape[0] * rotations.shape[1] * 3
        for j, name in enumerate(joint_names):
            name_lower = name.lower()
            if 'elbow' in name_lower or 'knee' in name_lower:
                # 这些关节有明确的单向约束
                oob = np.sum(rotations[:, j, :] < -10) + np.sum(rotations[:, j, :] > 170)
                out_of_range += oob
        if total > 0:
            stats['关节越界率(%)'] = float(out_of_range / total * 100)
        
        # 3. 根节点位移平滑度
        pos_velocity = np.diff(positions, axis=0)
        stats['根节点速度均值'] = float(np.mean(np.abs(pos_velocity)))
        
        # 4. 根节点加速度（抖动指标）
        if len(positions) > 2:
            pos_accel = np.diff(pos_velocity, axis=0)
            stats['根节点加速度均值'] = float(np.mean(np.abs(pos_accel)))
        
        return stats
