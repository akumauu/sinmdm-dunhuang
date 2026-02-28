"""
Motion Evaluation Metrics for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - 生成质量评估模块

提供客观量化指标：
- 帧间平滑度 (Angular Velocity)
- 关节越界率 (Joint Limit Violation Rate)
- 足部滑动距离 (Foot Sliding Distance)
- 与原始动作的分布相似性 (Distribution Similarity)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvalReport:
    """评估报告"""
    motion_name: str = ""
    num_frames: int = 0
    duration_sec: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """输出 Markdown 格式报告"""
        lines = [
            f"## 评估报告: {self.motion_name}",
            f"- **帧数**: {self.num_frames}",
            f"- **时长**: {self.duration_sec:.2f} 秒",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
        ]
        for key, val in self.metrics.items():
            lines.append(f"| {key} | {val:.4f} |")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            'motion_name': self.motion_name,
            'num_frames': self.num_frames,
            'duration_sec': self.duration_sec,
            **self.metrics
        }


class MotionEvaluator:
    """动作质量评估器"""
    
    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps: 帧率
        """
        self.fps = fps
    
    def evaluate(
        self,
        rotations: np.ndarray,
        positions: np.ndarray,
        joint_names: List[str],
        motion_name: str = "unknown",
        reference_rotations: Optional[np.ndarray] = None
    ) -> EvalReport:
        """
        评估生成动作质量
        
        Args:
            rotations: 生成的旋转数据 (frames, joints, 3)
            positions: 根节点位置 (frames, 3)
            joint_names: 关节名称
            motion_name: 动作名称
            reference_rotations: 参考（原始）旋转数据 (可选)
            
        Returns:
            EvalReport: 评估报告
        """
        metrics = {}
        num_frames = rotations.shape[0]
        duration = num_frames / self.fps
        
        # 1. 帧间平滑度
        smoothness = self.compute_smoothness(rotations)
        metrics.update(smoothness)
        
        # 2. 关节越界率
        violation = self.compute_joint_violation(rotations, joint_names)
        metrics.update(violation)
        
        # 3. 根节点运动统计
        root_stats = self.compute_root_motion_stats(positions)
        metrics.update(root_stats)
        
        # 4. 与参考动作对比
        if reference_rotations is not None:
            similarity = self.compute_distribution_similarity(
                rotations, reference_rotations
            )
            metrics.update(similarity)
        
        return EvalReport(
            motion_name=motion_name,
            num_frames=num_frames,
            duration_sec=duration,
            metrics=metrics
        )
    
    def compute_smoothness(self, rotations: np.ndarray) -> Dict[str, float]:
        """
        计算帧间平滑度
        
        基于关节角速度和角加速度：
        - 角速度越低，动作越平滑
        - 角加速度越低，动作越连贯
        """
        # 角速度: 帧间差分
        angular_velocity = np.diff(rotations, axis=0)  # (F-1, J, 3)
        
        # 角加速度: 二阶差分
        angular_accel = np.diff(angular_velocity, axis=0)  # (F-2, J, 3)
        
        return {
            '角速度均值(°/帧)': float(np.mean(np.abs(angular_velocity))),
            '角速度标准差(°/帧)': float(np.std(angular_velocity)),
            '角加速度均值(°/帧²)': float(np.mean(np.abs(angular_accel))),
            '最大角速度(°/帧)': float(np.max(np.abs(angular_velocity))),
        }
    
    def compute_joint_violation(
        self, 
        rotations: np.ndarray, 
        joint_names: List[str]
    ) -> Dict[str, float]:
        """
        计算关节越界率
        
        检查关键关节（肘、膝）是否超出物理合理范围
        """
        total_checks = 0
        violations = 0
        
        JOINT_LIMITS = {
            'elbow': (0, 160),
            'knee': (0, 160),
            'shoulder': (-180, 180),
            'hip': (-120, 120),
            'neck': (-60, 60),
            'spine': (-45, 45),
        }
        
        for j, name in enumerate(joint_names):
            name_lower = name.lower()
            for joint_key, (lo, hi) in JOINT_LIMITS.items():
                if joint_key in name_lower:
                    for axis in range(3):
                        vals = rotations[:, j, axis]
                        total_checks += len(vals)
                        violations += np.sum((vals < lo) | (vals > hi))
        
        rate = (violations / max(total_checks, 1)) * 100
        return {
            '关节越界率(%)': float(rate),
            '越界帧数': int(violations),
            '检查总数': int(total_checks),
        }
    
    def compute_root_motion_stats(self, positions: np.ndarray) -> Dict[str, float]:
        """计算根节点运动统计"""
        velocity = np.diff(positions, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        metrics = {
            '根节点平均速度': float(np.mean(speed)),
            '根节点最大速度': float(np.max(speed)),
            '根节点总位移': float(np.linalg.norm(positions[-1] - positions[0])),
        }
        
        # 抖动指标: 加速度
        if len(positions) > 2:
            accel = np.diff(velocity, axis=0)
            jerk = np.linalg.norm(accel, axis=1)
            metrics['根节点平均抖动'] = float(np.mean(jerk))
        
        return metrics
    
    def compute_distribution_similarity(
        self,
        generated: np.ndarray,
        reference: np.ndarray
    ) -> Dict[str, float]:
        """
        对比生成与参考动作的分布相似性
        
        使用每个关节旋转的均值/方差距离
        """
        # 展平为 (frames, features)
        gen_flat = generated.reshape(generated.shape[0], -1)
        ref_flat = reference.reshape(reference.shape[0], -1)
        
        # 均值距离
        mean_diff = np.mean(np.abs(gen_flat.mean(axis=0) - ref_flat.mean(axis=0)))
        
        # 方差距离
        std_diff = np.mean(np.abs(gen_flat.std(axis=0) - ref_flat.std(axis=0)))
        
        return {
            '均值距离': float(mean_diff),
            '方差距离': float(std_diff),
        }
    
    def batch_evaluate(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray]],
        joint_names: List[str],
        motion_name: str = "batch"
    ) -> List[EvalReport]:
        """
        批量评估多个生成样本
        
        Args:
            samples: [(rotations, positions), ...] 列表
            joint_names: 关节名称
            motion_name: 动作名称
            
        Returns:
            评估报告列表
        """
        reports = []
        for i, (rot, pos) in enumerate(samples):
            report = self.evaluate(
                rot, pos, joint_names, 
                motion_name=f"{motion_name}_sample{i}"
            )
            reports.append(report)
        return reports
    
    def summary_table(self, reports: List[EvalReport]) -> str:
        """生成汇总对比表格"""
        if not reports:
            return "无评估数据"
        
        # 收集所有指标名
        all_keys = list(reports[0].metrics.keys())
        
        lines = ["## 评估汇总表"]
        header = "| 样本 | " + " | ".join(all_keys) + " |"
        sep = "|---" * (len(all_keys) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        
        for r in reports:
            vals = [f"{r.metrics.get(k, 0):.4f}" for k in all_keys]
            lines.append(f"| {r.motion_name} | " + " | ".join(vals) + " |")
        
        # 平均值
        avg_vals = []
        for k in all_keys:
            avg = np.mean([r.metrics.get(k, 0) for r in reports])
            avg_vals.append(f"{avg:.4f}")
        lines.append(f"| **平均** | " + " | ".join(avg_vals) + " |")
        
        return "\n".join(lines)
