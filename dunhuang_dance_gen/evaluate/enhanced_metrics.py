"""
Enhanced Motion Evaluation - 增强版动作序列评估模块
适配 SinMDM 论文标准的 BVH 级评估指标

SinMDM (ICLR 2024) 使用的原版指标 (via GANimator):
  - Coverage: GT 动作中被生成动作覆盖的比例
  - Global Diversity: 基于 Patched NN 的全局多样性
  - Local Diversity: 基于 Per-window NN 的局部多样性
  - Inter/Intra Diversity Distance: 样本间/样本内多样性

本模块实现了纯 NumPy 版本 (无需 ganimator_eval_kernel)，
并新增了以下针对动作序列质量的专业指标：
  - 足部滑动距离 (Foot Skating)
  - 关节角度物理可行性 (Joint Angle Plausibility)
  - 帧间平滑度 (Jerk / Smoothness)
  - 运动节奏一致性 (Motion Rhythm Consistency)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MotionQualityReport:
    """动作质量评估报告"""
    motion_name: str
    num_frames: int
    num_joints: int
    fps: float
    
    # 平滑性指标
    smoothness: Dict[str, float] = field(default_factory=dict)
    
    # 物理可行性指标
    physical_plausibility: Dict[str, float] = field(default_factory=dict)
    
    # 足部指标
    foot_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 多样性指标 (需要多个样本)
    diversity: Dict[str, float] = field(default_factory=dict)
    
    # 与参考动作的相似度
    similarity: Dict[str, float] = field(default_factory=dict)
    
    # 敦煌舞风格特征
    style_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        lines = [
            f"### {self.motion_name}",
            f"基本信息: {self.num_frames} 帧, {self.num_joints} 关节, {self.fps} FPS",
            "",
        ]
        
        for section_name, metrics in [
            ("平滑性指标", self.smoothness),
            ("物理可行性", self.physical_plausibility),
            ("足部运动指标", self.foot_metrics),
            ("多样性指标", self.diversity),
            ("与参考动作相似度", self.similarity),
            ("敦煌舞风格特征", self.style_metrics),
        ]:
            if metrics:
                lines.append(f"**{section_name}**")
                lines.append("")
                lines.append("| 指标 | 值 |")
                lines.append("|------|-----|")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        lines.append(f"| {k} | {v:.4f} |")
                    else:
                        lines.append(f"| {k} | {v} |")
                lines.append("")
        
        return "\n".join(lines)


class EnhancedMotionEvaluator:
    """增强版动作评估器 — 学术论文级指标"""
    
    # 敦煌舞 22 关节的参考关节限位 (单位: 度)
    JOINT_LIMITS = {
        'elbow':   (-5, 160),     # 肘关节屈曲
        'knee':    (-5, 160),     # 膝关节屈曲
        'hip':     (-120, 120),   # 髋关节各方向
        'spine':   (-60, 60),     # 脊柱各方向
        'neck':    (-80, 80),     # 颈部
        'shoulder':(-180, 180),   # 肩关节（灵活度大）
        'ankle':   (-50, 70),     # 脚踝
        'wrist':   (-90, 90),     # 手腕
    }
    
    # 足部关节名称匹配
    FOOT_KEYWORDS = ['foot', 'toe', 'Foot', 'Toe']
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps
    
    # ================================================================
    # 1. 平滑性指标 (Smoothness Metrics)
    # ================================================================
    def compute_smoothness(self, rotations: np.ndarray) -> Dict[str, float]:
        """
        计算动作平滑性指标
        
        Args:
            rotations: (T, J, 3) 或 (T, J*3) 旋转数据
        
        Returns:
            Dict 包含:
            - jerk: 急动度 (三阶导数，越小越平滑)
            - angular_velocity_mean: 角速度均值
            - angular_velocity_std: 角速度标准差
            - angular_acceleration_mean: 角加速度均值
        """
        if rotations.ndim == 3:
            rot_flat = rotations.reshape(rotations.shape[0], -1)
        else:
            rot_flat = rotations
        
        metrics = {}
        
        # 一阶: 角速度
        vel = np.diff(rot_flat, axis=0) / self.dt
        metrics['角速度均值(°/s)'] = float(np.mean(np.abs(vel)))
        metrics['角速度标准差'] = float(np.std(vel))
        metrics['角速度最大值'] = float(np.max(np.abs(vel)))
        
        # 二阶: 角加速度
        if rot_flat.shape[0] > 2:
            acc = np.diff(vel, axis=0) / self.dt
            metrics['角加速度均值(°/s²)'] = float(np.mean(np.abs(acc)))
            metrics['角加速度标准差'] = float(np.std(acc))
        
        # 三阶: 急动度 (Jerk) — 越小越平滑，论文核心指标
        if rot_flat.shape[0] > 3:
            jerk = np.diff(acc, axis=0) / self.dt
            metrics['急动度Jerk(°/s³)'] = float(np.mean(np.abs(jerk)))
        
        return metrics
    
    # ================================================================
    # 2. 物理可行性指标 (Physical Plausibility)
    # ================================================================
    def compute_physical_plausibility(self, rotations: np.ndarray, 
                                      joint_names: List[str]) -> Dict[str, float]:
        """
        计算关节角度物理可行性
        
        检查每个关节的旋转是否在人体生理范围内
        """
        metrics = {}
        
        total_frames = rotations.shape[0]
        total_violations = 0
        joint_violation_counts = {}
        
        for j, name in enumerate(joint_names):
            name_lower = name.lower()
            
            # 匹配关节类别
            limit = None
            for keyword, (lo, hi) in self.JOINT_LIMITS.items():
                if keyword in name_lower:
                    limit = (lo, hi)
                    break
            
            if limit is None:
                continue
            
            lo, hi = limit
            if rotations.ndim == 3 and j < rotations.shape[1]:
                joint_rot = rotations[:, j, :]
            elif rotations.ndim == 2:
                idx = j * 3
                if idx + 3 <= rotations.shape[1]:
                    joint_rot = rotations[:, idx:idx+3]
                else:
                    continue
            else:
                continue
            
            # 计算每帧是否越界
            violations = np.sum(
                np.any((joint_rot < lo) | (joint_rot > hi), axis=1)
            )
            total_violations += violations
            if violations > 0:
                joint_violation_counts[name] = int(violations)
        
        total_checks = total_frames * len(joint_names)
        violation_rate = total_violations / max(total_checks, 1) * 100
        
        metrics['关节越界率(%)'] = float(violation_rate)
        metrics['越界总帧数'] = int(total_violations)
        metrics['越界关节数'] = len(joint_violation_counts)
        
        # 列出 top-3 越界最多的关节
        sorted_violations = sorted(joint_violation_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (name, count) in enumerate(sorted_violations[:3]):
            metrics[f'越界TOP{i+1}:{name}'] = count
        
        return metrics
    
    # ================================================================
    # 3. 足部指标 (Foot Metrics) — 基于简化 FK 的滑动检测
    # ================================================================
    
    # 骨骼层级定义 (parent → child) — 敦煌舞 22 关节
    SKELETON_CHAINS = {
        'left_leg':  [0, 11, 12, 13, 17],   # Hips → L_Hip → L_Knee → L_Ankle → L_Foot
        'right_leg': [0, 14, 15, 16, 18],    # Hips → R_Hip → R_Knee → R_Ankle → R_Foot
    }
    
    # 各段骨骼默认方向和长度
    BONE_DEFAULTS = {
        (0, 11): (np.array([-0.1, -1, 0]), 8.0),    # Hips → L_Hip
        (11, 12): (np.array([0, -1, 0]), 15.0),      # L_Hip → L_Knee
        (12, 13): (np.array([0, -1, 0]), 15.0),      # L_Knee → L_Ankle
        (13, 17): (np.array([0, -0.2, 0.5]), 5.0),   # L_Ankle → L_Foot
        (0, 14): (np.array([0.1, -1, 0]), 8.0),      # Hips → R_Hip
        (14, 15): (np.array([0, -1, 0]), 15.0),       # R_Hip → R_Knee
        (15, 16): (np.array([0, -1, 0]), 15.0),       # R_Knee → R_Ankle
        (16, 18): (np.array([0, -0.2, 0.5]), 5.0),   # R_Ankle → R_Foot
    }
    
    def _simple_fk_foot(self, rotations: np.ndarray, 
                         positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        简化前向运动学: 从旋转计算左右足部世界坐标
        
        Returns:
            left_foot_pos: (T, 3) 左脚世界坐标
            right_foot_pos: (T, 3) 右脚世界坐标
        """
        T = rotations.shape[0]
        n_joints = rotations.shape[1] if rotations.ndim == 3 else rotations.shape[1] // 3
        
        rot_data = rotations.reshape(T, -1, 3) if rotations.ndim == 2 else rotations
        
        results = {}
        for side, chain in self.SKELETON_CHAINS.items():
            foot_pos = np.zeros((T, 3))
            
            for t in range(T):
                # 根节点位置
                if positions is not None and positions.ndim == 2 and len(positions) > t:
                    current_pos = positions[t, :3].copy()
                else:
                    current_pos = np.zeros(3)
                
                # 沿运动链累加
                for k in range(len(chain) - 1):
                    parent, child = chain[k], chain[k+1]
                    bone_key = (parent, child)
                    
                    if bone_key in self.BONE_DEFAULTS:
                        default_dir, bone_len = self.BONE_DEFAULTS[bone_key]
                    else:
                        default_dir = np.array([0, -1, 0])
                        bone_len = 10.0
                    
                    # 使用旋转角度调制方向 (简化: Euler 直接调制)
                    if child < n_joints:
                        angle_rad = np.deg2rad(rot_data[t, child] * 0.1)
                        # 简单旋转: 用三个欧拉角各自偏转
                        direction = default_dir.copy()
                        direction[0] += np.sin(angle_rad[2]) * 0.3   # Z旋转影响X
                        direction[2] += np.sin(angle_rad[0]) * 0.3   # X旋转影响Z
                    else:
                        direction = default_dir
                    
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    current_pos = current_pos + direction * bone_len
                
                foot_pos[t] = current_pos
            
            results[side] = foot_pos
        
        return results.get('left_leg', np.zeros((T, 3))), results.get('right_leg', np.zeros((T, 3)))
    
    def compute_foot_metrics(self, positions: np.ndarray, 
                              joint_names: List[str],
                              rotations: Optional[np.ndarray] = None,
                              ground_level: float = 0.0,
                              contact_threshold: float = 3.0) -> Dict[str, float]:
        """
        计算足部相关指标 (含 FK 滑动检测)
        
        - 足部滑动距离: 足部在地面接触时的水平位移
        - 地面穿透率: 足部低于地面的帧比例
        - 接触比例: 足部处于地面接触状态的帧比例
        """
        metrics = {}
        
        # 如果有旋转数据，用简化 FK 计算足部位置
        if rotations is not None:
            left_foot, right_foot = self._simple_fk_foot(rotations, positions)
            
            # --- 接触检测 ---
            left_contact = left_foot[:, 1] < (ground_level + contact_threshold)
            right_contact = right_foot[:, 1] < (ground_level + contact_threshold)
            
            total_frames = len(left_contact)
            left_contact_ratio = np.sum(left_contact) / total_frames * 100
            right_contact_ratio = np.sum(right_contact) / total_frames * 100
            
            metrics['左脚接触比例(%)'] = float(left_contact_ratio)
            metrics['右脚接触比例(%)'] = float(right_contact_ratio)
            
            # --- 滑动距离 ---
            def _skating_distance(foot_pos, contact_mask):
                """计算接触状态下的水平滑动距离"""
                xz = foot_pos[:, [0, 2]]  # 水平面 (X, Z)
                disp = np.sqrt(np.sum(np.diff(xz, axis=0) ** 2, axis=1))
                # 只在接触帧计算
                contact_disp = disp[contact_mask[1:]]  # diff 少一帧
                if len(contact_disp) > 0:
                    return float(np.mean(contact_disp)), float(np.sum(contact_disp))
                return 0.0, 0.0
            
            left_avg, left_total = _skating_distance(left_foot, left_contact)
            right_avg, right_total = _skating_distance(right_foot, right_contact)
            
            metrics['左脚滑动(平均)'] = left_avg
            metrics['右脚滑动(平均)'] = right_avg
            metrics['足部滑动距离(总)'] = left_total + right_total
            
            # --- 地面穿透 ---
            left_penetrate = np.sum(left_foot[:, 1] < ground_level)
            right_penetrate = np.sum(right_foot[:, 1] < ground_level)
            total_penetrate = left_penetrate + right_penetrate
            metrics['地面穿透帧数'] = int(total_penetrate)
            metrics['地面穿透率(%)'] = float(total_penetrate / (2 * total_frames) * 100)
            
            return metrics
        
        # 降级: 仅有根节点位置
        if positions is not None and positions.ndim == 2 and positions.shape[1] >= 3:
            below_ground = np.sum(positions[:, 1] < ground_level)
            metrics['地面穿透帧数'] = int(below_ground)
            metrics['地面穿透率(%)'] = float(below_ground / positions.shape[0] * 100)
        else:
            metrics['足部指标'] = '数据不足'
        
        return metrics
    
    # ================================================================
    # 4. 多样性指标 (Diversity Metrics) — SinMDM 论文标准
    # ================================================================
    def compute_diversity(self, samples: List[np.ndarray]) -> Dict[str, float]:
        """
        计算生成样本间的多样性
        
        对应 SinMDM 论文中的:
        - Inter-sample Diversity: 不同样本之间的平均距离
        - Intra-sample Diversity: 同一样本内不同片段的平均距离
        
        Args:
            samples: 多个生成样本的旋转数据列表，每个 shape = (T, features)
        """
        metrics = {}
        
        if len(samples) < 2:
            metrics['样本间多样性(Inter)'] = 0.0
            return metrics
        
        # 展平为 (T, F)
        flat_samples = []
        for s in samples:
            if s.ndim == 3:
                flat_samples.append(s.reshape(s.shape[0], -1))
            else:
                flat_samples.append(s)
        
        # Inter-sample diversity: 所有样本对之间的平均帧距离
        inter_dists = []
        for i in range(len(flat_samples)):
            for j in range(i + 1, len(flat_samples)):
                s1, s2 = flat_samples[i], flat_samples[j]
                min_len = min(len(s1), len(s2))
                if min_len > 0:
                    dist = np.sqrt(np.sum((s1[:min_len] - s2[:min_len]) ** 2, axis=1))
                    inter_dists.append(np.mean(dist))
        
        if inter_dists:
            metrics['样本间多样性(Inter)'] = float(np.mean(inter_dists))
            metrics['样本间多样性标准差'] = float(np.std(inter_dists))
        
        # Intra-sample diversity: 单个样本内不同窗口的距离
        window_size = 15  # SinMDM 默认 tmin=15
        intra_dists = []
        
        for s in flat_samples:
            if len(s) < 2 * window_size:
                continue
            
            n_windows = min(10, len(s) // window_size)
            starts = np.random.choice(len(s) - window_size, size=min(n_windows * 2, len(s) - window_size), replace=False)
            
            for k in range(0, len(starts) - 1, 2):
                w1 = s[starts[k]:starts[k] + window_size]
                w2 = s[starts[k+1]:starts[k+1] + window_size]
                dist = np.sqrt(np.sum((w1 - w2) ** 2, axis=1))
                intra_dists.append(np.mean(dist))
        
        if intra_dists:
            metrics['样本内多样性(Intra)'] = float(np.mean(intra_dists))
        
        return metrics
    
    # ================================================================
    # 5. 覆盖率 (Coverage) — SinMDM 核心指标
    # ================================================================
    def compute_coverage(self, generated_rot: np.ndarray, reference_rot: np.ndarray,
                          window_size: int = 15) -> Dict[str, float]:
        """
        计算覆盖率: 参考动作中有多少比例的运动模式被生成动作覆盖
        
        使用滑动窗口最近邻匹配，距离使用逐帧平均 L2 范数 (与 SinMDM 一致)
        阈值根据参考动作自身的窗口间距离自适应设定
        """
        gen = generated_rot.reshape(generated_rot.shape[0], -1) if generated_rot.ndim == 3 else generated_rot
        ref = reference_rot.reshape(reference_rot.shape[0], -1) if reference_rot.ndim == 3 else reference_rot
        
        metrics = {}
        
        if len(ref) < window_size or len(gen) < window_size:
            metrics['覆盖率(%)'] = 0.0
            return metrics
        
        # 提取窗口 — 每个窗口是 (window_size, features) 展平为 (window_size * features,)
        n_ref = len(ref) - window_size + 1
        n_gen = len(gen) - window_size + 1
        n_feats = ref.shape[1]
        
        ref_windows = np.array([ref[i:i + window_size] for i in range(n_ref)])  # (N_ref, W, F)
        gen_windows = np.array([gen[i:i + window_size] for i in range(n_gen)])  # (N_gen, W, F)
        
        # 计算逐帧平均 L2 距离 (每个 window pair)
        # 为了效率，先展平 (N, W*F)，再用逐帧距离的等效计算
        ref_flat = ref_windows.reshape(n_ref, -1)  # (N_ref, W*F)
        gen_flat = gen_windows.reshape(n_gen, -1)  # (N_gen, W*F)
        
        # 自适应阈值: 基于参考动作内相邻窗口的距离中位数
        step = max(1, n_ref // 20)
        self_dists = []
        for i in range(0, n_ref - step, step):
            d = np.sqrt(np.sum((ref_flat[i] - ref_flat[i + step]) ** 2)) / window_size
            self_dists.append(d)
        
        if self_dists:
            adaptive_threshold = np.median(self_dists) * 1.5  # 1.5x 中位自距离
        else:
            adaptive_threshold = 50.0  # fallback
        
        # 计算每个参考窗口到最近生成窗口的距离 (逐帧归一化)
        covered = 0
        nearest_dists = []
        
        batch_size = 50
        for start in range(0, n_ref, batch_size):
            end = min(start + batch_size, n_ref)
            batch = ref_flat[start:end]  # (B, W*F)
            
            # (B, 1, W*F) - (1, N_gen, W*F) -> (B, N_gen)
            dists = np.sqrt(np.sum(
                (batch[:, np.newaxis, :] - gen_flat[np.newaxis, :, :]) ** 2,
                axis=2
            )) / window_size  # 逐帧归一化
            
            min_dists = np.min(dists, axis=1)
            nearest_dists.extend(min_dists.tolist())
            covered += np.sum(min_dists < adaptive_threshold)
        
        total_windows = n_ref
        metrics['覆盖率(%)'] = float(covered / total_windows * 100)
        metrics['平均最近邻距离'] = float(np.mean(nearest_dists))
        metrics['最近邻距离中位数'] = float(np.median(nearest_dists))
        metrics['自适应阈值'] = float(adaptive_threshold)
        
        return metrics
    
    # ================================================================
    # 6. 分布距离 (Distribution Distance)
    # ================================================================
    def compute_distribution_distance(self, gen_rot: np.ndarray, ref_rot: np.ndarray) -> Dict[str, float]:
        """
        计算生成与参考动作之间的分布距离
        
        指标:
        - 均值距离 (FMD-like)
        - 方差距离
        - KL 散度近似
        """
        gen = gen_rot.reshape(gen_rot.shape[0], -1) if gen_rot.ndim == 3 else gen_rot
        ref = ref_rot.reshape(ref_rot.shape[0], -1) if ref_rot.ndim == 3 else ref_rot
        
        metrics = {}
        
        # 均值距离
        gen_mean = np.mean(gen, axis=0)
        ref_mean = np.mean(ref, axis=0)
        metrics['分布均值距离(FMD)'] = float(np.sqrt(np.mean((gen_mean - ref_mean) ** 2)))
        
        # 方差距离
        gen_std = np.std(gen, axis=0)
        ref_std = np.std(ref, axis=0)
        metrics['分布方差距离'] = float(np.sqrt(np.mean((gen_std - ref_std) ** 2)))
        
        # 逐维度 KL 散度近似 (假设高斯分布)
        eps = 1e-8
        kl_per_dim = 0.5 * (
            np.log((ref_std + eps) / (gen_std + eps)) +
            (gen_std ** 2 + (gen_mean - ref_mean) ** 2) / (2 * (ref_std + eps) ** 2) - 0.5
        )
        metrics['KL散度(近似)'] = float(np.mean(np.clip(kl_per_dim, 0, 100)))
        
        return metrics
    
    # ================================================================
    # 综合评估
    # ================================================================
    def evaluate(self, rotations: np.ndarray, positions: np.ndarray,
                 joint_names: List[str], motion_name: str = "unknown",
                 reference_rotations: Optional[np.ndarray] = None,
                 additional_samples: Optional[List[np.ndarray]] = None) -> MotionQualityReport:
        """
        执行全面评估
        
        Args:
            rotations: (T, J, 3) 旋转数据
            positions: (T, 3) 根节点位置
            joint_names: 关节名称列表
            motion_name: 动作名称
            reference_rotations: 参考动作旋转（用于相似度/覆盖率计算）
            additional_samples: 其他生成样本的旋转数据（用于多样性计算）
        """
        report = MotionQualityReport(
            motion_name=motion_name,
            num_frames=rotations.shape[0],
            num_joints=rotations.shape[1] if rotations.ndim == 3 else len(joint_names),
            fps=self.fps,
        )
        
        # 1. 平滑性
        report.smoothness = self.compute_smoothness(rotations)
        
        # 2. 物理可行性
        report.physical_plausibility = self.compute_physical_plausibility(rotations, joint_names)
        
        # 3. 足部指标 (含 FK 滑动检测)
        report.foot_metrics = self.compute_foot_metrics(positions, joint_names, rotations=rotations)
        
        # 4. 与参考动作相似度
        if reference_rotations is not None:
            report.similarity = self.compute_distribution_distance(rotations, reference_rotations)
            report.similarity.update(self.compute_coverage(rotations, reference_rotations))
        
        # 5. 多样性
        if additional_samples:
            all_samples = [rotations] + additional_samples
            report.diversity = self.compute_diversity(all_samples)
        
        # 6. 敦煌舞风格特征
        try:
            from .style_features import DunhuangStyleExtractor
            style_extractor = DunhuangStyleExtractor(fps=self.fps)
            style_profile = style_extractor.extract(rotations, motion_name)
            report.style_metrics = style_profile.to_dict()
        except Exception as e:
            report.style_metrics = {'风格提取错误': str(e)}
        
        return report
